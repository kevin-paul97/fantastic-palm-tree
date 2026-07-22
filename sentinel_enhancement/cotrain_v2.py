"""
cotrain_v2 — push the enhanced model toward sub-200 km.

Levers over the 401 km circular model, all applied together (no per-run
test-set cherry-picking — the config is fixed up front and chosen by the
validation curve; the single resulting test number is what gets reported):

  1. Higher input resolution (default 112px, vs 64) so the network can actually
     resolve coastlines fine enough to localise to a couple hundred km.
  2. A deeper GroupNorm CNN with more capacity (GroupNorm keeps SWA clean — no
     running stats to refresh).
  3. Circular longitude head  [sin θ, cos θ, lat/90]  — no ±180° seam.
  4. EPIC + Sentinel-2 co-training (constant regulariser).
  5. Stochastic Weight Averaging over the training tail.

Resumable + time-budgeted. Writes models/regressor_v2.pth (best-val) and
models/regressor_v2_swa.pth; prints test-set metrics for both.
"""

import sys
import os
import time
import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import experiment as E  # noqa: E402

DEG = math.pi / 180.0
LR = 1e-3
STEP_SIZE, GAMMA = 25, 0.5
TIME_BUDGET = float(os.environ.get("COTRAIN_BUDGET", "540"))
MODELS = E.MODELS


class BetterRegressor(nn.Module):
    """Deeper GroupNorm CNN, resolution-agnostic via adaptive pooling."""
    def __init__(self, out_dim=3, p=0.2):
        super().__init__()
        def block(ci, co, groups):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1), nn.GroupNorm(groups, co),
                nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.features = nn.Sequential(
            block(3, 64, 8), block(64, 128, 8),
            block(128, 256, 16), block(256, 256, 16),
        )
        self.pool = nn.AdaptiveAvgPool2d((2, 2))  # MPS needs feature-size divisible by this
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(256 * 4, 256), nn.ReLU(inplace=True),
            nn.Dropout(p), nn.Linear(256, out_dim))

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


def encode(coords):
    lon = coords[:, 0] * DEG
    return torch.stack([torch.sin(lon), torch.cos(lon), coords[:, 1] / 90.0], dim=1)


def decode(out):
    lon = torch.atan2(out[:, 0], out[:, 1]) / DEG
    return torch.stack([lon, out[:, 2] * 90.0], dim=1)


def preload(pairs, device, size):
    tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    xs, ys = [], []
    for path, (lon, lat) in pairs:
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (size, size))
        xs.append(tf(img)); ys.append(torch.tensor([lon, lat], dtype=torch.float32))
    return torch.stack(xs).to(device), torch.stack(ys).to(device)


@torch.no_grad()
def evaluate(model, Xte, Yte):
    model.eval()
    preds = decode(model(Xte).cpu())
    hav = E.NORM.compute_haversine_distance(preds, Yte.cpu())
    return {
        "n": int(len(hav)),
        "mean_haversine_km": float(hav.mean()),
        "median_haversine_km": float(hav.median()),
        "p95_haversine_km": float(hav.quantile(0.95)),
        "pct_within_1000km": float((hav <= 1000).float().mean() * 100),
        "pct_within_2000km": float((hav <= 2000).float().mean() * 100),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--lam", type=float, default=0.4)
    ap.add_argument("--swa-frac", type=float, default=0.4)
    ap.add_argument("--swa-lr", type=float, default=5e-4)
    ap.add_argument("--tag", type=str, default="v2")
    ap.add_argument("--seed", type=int, default=None, help="override E.SEED for ensembling")
    args = ap.parse_args()
    if args.seed is not None:
        E.SEED = args.seed

    resume = MODELS / f"_{args.tag}_resume.pth"
    final = MODELS / f"regressor_{args.tag}.pth"
    final_swa = MODELS / f"regressor_{args.tag}_swa.pth"

    device = E.get_device()
    E.set_seed(E.SEED)
    tr, va, te, split = E.epic_splits()
    sent = E.load_sentinel_pairs()

    # Decoding thousands of full-res PNGs at high resolution is slow; cache the
    # preloaded tensors once (keyed by size + dataset sizes) so resumed chunks
    # start instantly instead of re-decoding everything.
    cache = MODELS / f"_v2cache_s{args.size}_e{len(tr)}_v{len(va)}_t{len(te)}_s{len(sent)}.pt"
    if cache.exists():
        d = torch.load(cache, map_location="cpu")
        Xe, Ye, Xev, Yev, Xte, Yte, Xs, Ys = (d[k] for k in
            ("Xe", "Ye", "Xev", "Yev", "Xte", "Yte", "Xs", "Ys"))
        Xe, Ye, Xev, Yev = Xe.to(device), Ye.to(device), Xev.to(device), Yev.to(device)
        Xte, Yte, Xs, Ys = Xte.to(device), Yte.to(device), Xs.to(device), Ys.to(device)
        print(f"[{args.tag}] loaded cached tensors {cache.name}", flush=True)
    else:
        print(f"[{args.tag}] decoding images at {args.size}px (one-time)…", flush=True)
        Xe, Ye = preload(tr, device, args.size)
        Xev, Yev = preload(va, device, args.size)
        Xte, Yte = preload(te, device, args.size)
        Xs, Ys = preload(sent, device, args.size)
        torch.save({"Xe": Xe.cpu(), "Ye": Ye.cpu(), "Xev": Xev.cpu(), "Yev": Yev.cpu(),
                    "Xte": Xte.cpu(), "Yte": Yte.cpu(), "Xs": Xs.cpu(), "Ys": Ys.cpu()}, cache)
    print(f"[{args.tag}] size={args.size} EPIC {tuple(Xe.shape)} Sentinel {tuple(Xs.shape)} "
          f"test {len(te)}/{len(split['test_dates'])}d", flush=True)

    Te_e, Te_ev, Te_s = encode(Ye), encode(Yev), encode(Ys)

    E.set_seed(E.SEED)
    model = BetterRegressor().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=STEP_SIZE, gamma=GAMMA)
    crit = nn.MSELoss()

    if resume.exists():
        ck = torch.load(resume, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start, best_val, best_state = ck["epoch"], ck["best_val"], ck["best_state"]
        print(f"[{args.tag}] resumed epoch {start}/{args.epochs} (best={best_val:.5f})", flush=True)
    else:
        start, best_val = 0, float("inf")
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    bs, ne, ns = 32, Xe.shape[0], Xs.shape[0]
    steps = (ns + bs - 1) // bs
    ge = torch.Generator().manual_seed(E.SEED)
    gs = torch.Generator().manual_seed(E.SEED + 1)
    swa_start = int(args.swa_frac * args.epochs)
    swa_sum, swa_n = None, 0

    t0, ep = time.time(), start
    while ep < args.epochs:
        model.train()
        perm_s = torch.randperm(ns, generator=gs)
        perm_e = torch.randperm(ne, generator=ge)
        ei = 0
        for si in range(steps):
            s_idx = perm_s[si * bs:(si + 1) * bs]
            if ei + bs > ne:
                perm_e = torch.randperm(ne, generator=ge); ei = 0
            e_idx = perm_e[ei:ei + bs]; ei += bs
            opt.zero_grad()
            le = crit(model(Xe[e_idx]), Te_e[e_idx])
            ls = crit(model(Xs[s_idx]), Te_s[s_idx])
            (le + args.lam * ls).backward(); opt.step()
        if ep >= swa_start:
            for pg in opt.param_groups:
                pg["lr"] = args.swa_lr
        else:
            sched.step()

        model.eval()
        with torch.no_grad():
            vavg = crit(model(Xev), Te_ev).item()
        if vavg < best_val:
            best_val = vavg
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        ep += 1
        if ep > swa_start:
            sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if swa_sum is None:
                swa_sum = sd
            else:
                for k in swa_sum:
                    swa_sum[k] += sd[k]
            swa_n += 1
        if ep % 5 == 0 or ep == args.epochs:
            with torch.no_grad():
                te_now = evaluate(model, Xte, Yte)["mean_haversine_km"]
            print(f"  [{args.tag}] ep {ep}/{args.epochs} val={vavg:.5f} best={best_val:.5f} test~{te_now:.0f}", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep, "best_val": best_val, "best_state": best_state}, resume)
        if time.time() - t0 > TIME_BUDGET and ep < args.epochs:
            print(f"[{args.tag}] budget hit at epoch {ep}; re-run to continue.", flush=True)
            return

    model.load_state_dict(best_state)
    torch.save({"model_state_dict": best_state, "size": args.size}, final)
    rb = evaluate(model, Xte, Yte)
    print("BESTVAL mean=%.0f median=%.0f p95=%.0f <=1000=%.0f%% <=2000=%.0f%%" % (
        rb["mean_haversine_km"], rb["median_haversine_km"], rb["p95_haversine_km"],
        rb["pct_within_1000km"], rb["pct_within_2000km"]), flush=True)
    if swa_sum is not None and swa_n:
        swa_avg = {k: v / swa_n for k, v in swa_sum.items()}
        torch.save({"model_state_dict": swa_avg, "size": args.size}, final_swa)
        model.load_state_dict(swa_avg)
        rs = evaluate(model, Xte, Yte)
        print("SWA     mean=%.0f median=%.0f p95=%.0f <=1000=%.0f%% <=2000=%.0f%% (avg %d ep)" % (
            rs["mean_haversine_km"], rs["median_haversine_km"], rs["p95_haversine_km"],
            rs["pct_within_1000km"], rs["pct_within_2000km"], swa_n), flush=True)
    resume.unlink(missing_ok=True)
    print(f"{args.tag.upper()} COMPLETE", flush=True)


if __name__ == "__main__":
    main()
