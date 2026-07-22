"""
Co-train EPIC + Sentinel-2 with a CIRCULAR longitude head.

The plateau in the plain model is entirely a longitude problem near the ±180°
dateline: regressing raw longitude puts a discontinuity right where the hardest
(featureless mid-Pacific) frames live, so the model catastrophically flips
hemispheres. The fix is to predict longitude on the circle:

    target = [ sin(lon), cos(lon), lat/90 ]           (3 outputs, all ~[-1,1])
    decode: lon = atan2(sin, cos),  lat = 90 * out[2]

No seam at ±180°, so dateline frames degrade gracefully instead of exploding.
Same EPIC+Sentinel co-training and SWA as cotrain.py.

Writes models/regressor_enhanced_circular.pth (+ _swa). Evaluated by decoding
back to (lon, lat) and computing the identical Haversine metric.
"""

import sys
import time
import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import experiment as E  # noqa: E402
from models import LocationRegressor  # noqa: E402

LR = 1e-3
STEP_SIZE, GAMMA = 20, 0.5
TIME_BUDGET = float(__import__("os").environ.get("COTRAIN_BUDGET", "560"))
MODELS = E.MODELS
RESUME = MODELS / "_circular_resume.pth"
FINAL = MODELS / "regressor_enhanced_circular.pth"
FINAL_SWA = MODELS / "regressor_enhanced_circular_swa.pth"
DEG = math.pi / 180.0


def preload(pairs, device):
    ds = E.PairDataset(pairs)
    xs, ys = zip(*[ds[i] for i in range(len(ds))])
    return torch.stack(xs).to(device), torch.stack(ys).to(device)


def encode(coords):
    """[lon,lat] deg -> [sin(lon), cos(lon), lat/90]."""
    lon = coords[:, 0] * DEG
    lat = coords[:, 1]
    return torch.stack([torch.sin(lon), torch.cos(lon), lat / 90.0], dim=1)


def decode(out):
    """[sin,cos,lat/90] -> [lon,lat] deg."""
    lon = torch.atan2(out[:, 0], out[:, 1]) / DEG
    lat = out[:, 2] * 90.0
    return torch.stack([lon, lat], dim=1)


def make_model(device):
    return LocationRegressor(input_channels=3, conv_channels=[64, 128, 256],
                             kernel_size=3, pool_size=4, activation="tanh",
                             hidden_dim=128, output_dim=3, dropout_rate=0.2).to(device)


@torch.no_grad()
def evaluate(model, Xte, Yte, device):
    model.eval()
    out = model(Xte).cpu()
    preds = decode(out)
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
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--swa-frac", type=float, default=0.35)
    ap.add_argument("--swa-lr", type=float, default=5e-4)
    args = ap.parse_args()

    device = E.get_device()
    E.set_seed()
    tr, va, te, _ = E.epic_splits()
    Xe, Ye = preload(tr, device)
    Xev, Yev = preload(va, device)
    Xte, Yte = preload(te, device)
    sent = E.load_sentinel_pairs()
    Xs, Ys = preload(sent, device)
    print(f"circular co-train: EPIC {tuple(Xe.shape)}  Sentinel {tuple(Xs.shape)}", flush=True)

    E.set_seed()
    model = make_model(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=STEP_SIZE, gamma=GAMMA)
    crit = nn.MSELoss()

    Te_e, Te_ev, Te_s = encode(Ye), encode(Yev), encode(Ys)

    if RESUME.exists():
        ck = torch.load(RESUME, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start, best_val, best_state = ck["epoch"], ck["best_val"], ck["best_state"]
    else:
        start, best_val = 0, float("inf")
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    bs = 32
    ne, ns = Xe.shape[0], Xs.shape[0]
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
        if ep % 10 == 0 or ep == args.epochs:
            print(f"  [circular] epoch {ep}/{args.epochs}  val={vavg:.5f}  best={best_val:.5f}", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep,
                    "best_val": best_val, "best_state": best_state}, RESUME)
        if time.time() - t0 > TIME_BUDGET and ep < args.epochs:
            print(f"[circular] budget hit at epoch {ep}; re-run to continue.", flush=True)
            return

    model.load_state_dict(best_state)
    torch.save({"model_state_dict": best_state}, FINAL)
    r = evaluate(model, Xte, Yte, device)
    print("BESTVAL  mean=%.0f median=%.0f p95=%.0f" % (
        r["mean_haversine_km"], r["median_haversine_km"], r["p95_haversine_km"]), flush=True)
    if swa_sum is not None and swa_n:
        swa_avg = {k: v / swa_n for k, v in swa_sum.items()}
        torch.save({"model_state_dict": swa_avg}, FINAL_SWA)
        model.load_state_dict(swa_avg)
        rs = evaluate(model, Xte, Yte, device)
        print("SWA      mean=%.0f median=%.0f p95=%.0f (avg %d ep)" % (
            rs["mean_haversine_km"], rs["median_haversine_km"], rs["p95_haversine_km"], swa_n), flush=True)
    RESUME.unlink(missing_ok=True)
    print("CIRCULAR COMPLETE", flush=True)


if __name__ == "__main__":
    main()
