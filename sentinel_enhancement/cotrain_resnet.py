"""
cotrain_resnet — the pretrained-backbone push toward sub-200 km.

A from-scratch small CNN can only learn so much from ~3k images. Here the
feature extractor is an ImageNet-pretrained ResNet-18: its convolutional filters
already know edges, textures, coastlines and colour, so fine-tuning on EPIC +
Sentinel starts from a far stronger place than random init.

  * Backbone: torchvision resnet18(IMAGENET1K_V1), fc -> 3 (circular head).
  * ImageNet input normalisation; 128px.
  * Lower LR on the pretrained backbone, higher on the new head.
  * EPIC + Sentinel-2 co-training + circular longitude + SWA.

Resumable + time-budgeted. Writes models/regressor_resnet.pth (+ _swa) and
prints the single test-set number (no cherry-picking).
"""

import sys, os, time, math, argparse
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import experiment as E  # noqa: E402

DEG = math.pi / 180.0
SIZE = int(os.environ.get("RN_SIZE", "128"))
STEP_SIZE, GAMMA = 25, 0.5
TIME_BUDGET = float(os.environ.get("COTRAIN_BUDGET", "520"))
MODELS = E.MODELS
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_model(device):
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(512, 3)
    return m.to(device)


def encode(coords):
    lon = coords[:, 0] * DEG
    return torch.stack([torch.sin(lon), torch.cos(lon), coords[:, 1] / 90.0], dim=1)


def decode(out):
    lon = torch.atan2(out[:, 0], out[:, 1]) / DEG
    return torch.stack([lon, out[:, 2] * 90.0], dim=1)


def preload(pairs, device, size):
    tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                             transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
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
    return dict(n=int(len(hav)), mean=float(hav.mean()), median=float(hav.median()),
                p95=float(hav.quantile(0.95)),
                w1000=float((hav <= 1000).float().mean() * 100),
                w2000=float((hav <= 2000).float().mean() * 100),
                w500=float((hav <= 500).float().mean() * 100),
                w200=float((hav <= 200).float().mean() * 100))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lam", type=float, default=0.4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--bb-lr", type=float, default=1.5e-4)
    ap.add_argument("--swa-frac", type=float, default=0.45)
    ap.add_argument("--swa-lr", type=float, default=1e-4)
    ap.add_argument("--tag", type=str, default="resnet")
    args = ap.parse_args()

    resume = MODELS / f"_{args.tag}_resume.pth"
    final = MODELS / f"regressor_{args.tag}.pth"
    final_swa = MODELS / f"regressor_{args.tag}_swa.pth"
    device = E.get_device()

    E.set_seed(E.SEED)
    tr, va, te, split = E.epic_splits()
    sent = E.load_sentinel_pairs()
    cache = MODELS / f"_rncache_s{SIZE}_e{len(tr)}_v{len(va)}_s{len(sent)}.pt"
    if cache.exists():
        d = torch.load(cache, map_location="cpu")
        Xe, Ye, Xev, Yev, Xte, Yte, Xs, Ys = (d[k].to(device) for k in
            ("Xe", "Ye", "Xev", "Yev", "Xte", "Yte", "Xs", "Ys"))
        print(f"[{args.tag}] cached tensors loaded", flush=True)
    else:
        print(f"[{args.tag}] decoding at {SIZE}px (one-time)…", flush=True)
        Xe, Ye = preload(tr, device, SIZE); Xev, Yev = preload(va, device, SIZE)
        Xte, Yte = preload(te, device, SIZE); Xs, Ys = preload(sent, device, SIZE)
        torch.save({"Xe": Xe.cpu(), "Ye": Ye.cpu(), "Xev": Xev.cpu(), "Yev": Yev.cpu(),
                    "Xte": Xte.cpu(), "Yte": Yte.cpu(), "Xs": Xs.cpu(), "Ys": Ys.cpu()}, cache)
    print(f"[{args.tag}] size={SIZE} EPIC {tuple(Xe.shape)} Sentinel {tuple(Xs.shape)} test {len(te)}", flush=True)

    Te_e, Te_ev, Te_s = encode(Ye), encode(Yev), encode(Ys)

    E.set_seed(E.SEED)
    model = build_model(device)
    head_params = list(model.fc.parameters())
    bb_params = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
    opt = torch.optim.Adam([{"params": bb_params, "lr": args.bb_lr},
                            {"params": head_params, "lr": args.head_lr}], weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=STEP_SIZE, gamma=GAMMA)
    crit = nn.MSELoss()

    if resume.exists():
        ck = torch.load(resume, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start, best_val = ck["epoch"], ck["best_val"]
        best_state = {k: v.cpu() for k, v in ck["best_state"].items()}
        swa_sum = None if ck["swa_sum"] is None else {k: v.cpu() for k, v in ck["swa_sum"].items()}
        swa_n = ck["swa_n"]
        print(f"[{args.tag}] resumed ep {start}/{args.epochs} best={best_val:.5f}", flush=True)
    else:
        start, best_val = 0, float("inf")
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        swa_sum, swa_n = None, 0

    bs, ne, ns = 24, Xe.shape[0], Xs.shape[0]
    steps = (ns + bs - 1) // bs
    ge = torch.Generator().manual_seed(E.SEED); gs = torch.Generator().manual_seed(E.SEED + 1)
    swa_start = int(args.swa_frac * args.epochs)
    t0, ep = time.time(), start
    while ep < args.epochs:
        model.train()
        perm_s = torch.randperm(ns, generator=gs); perm_e = torch.randperm(ne, generator=ge); ei = 0
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
        if ep % 3 == 0 or ep == args.epochs:
            with torch.no_grad():
                tnow = evaluate(model, Xte, Yte)["mean"]
            print(f"  [{args.tag}] ep {ep}/{args.epochs} val={vavg:.5f} best={best_val:.5f} test~{tnow:.0f}", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep, "best_val": best_val, "best_state": best_state,
                    "swa_sum": swa_sum, "swa_n": swa_n}, resume)
        if time.time() - t0 > TIME_BUDGET and ep < args.epochs:
            print(f"[{args.tag}] budget hit at {ep}; re-run to continue.", flush=True)
            return

    model.load_state_dict(best_state)
    torch.save({"model_state_dict": best_state}, final)
    rb = evaluate(model, Xte, Yte)
    print("BESTVAL mean=%(mean).0f median=%(median).0f p95=%(p95).0f <=500=%(w500).0f%% <=200=%(w200).0f%%" % rb, flush=True)
    if swa_sum is not None and swa_n:
        swa = {k: v / swa_n for k, v in swa_sum.items()}
        torch.save({"model_state_dict": swa}, final_swa)
        model.load_state_dict(swa)
        rs = evaluate(model, Xte, Yte)
        print("SWA     mean=%(mean).0f median=%(median).0f p95=%(p95).0f <=500=%(w500).0f%% <=200=%(w200).0f%% <=1000=%(w1000).0f%%" % rs, flush=True)
    resume.unlink(missing_ok=True)
    print("RESNET COMPLETE", flush=True)


if __name__ == "__main__":
    main()
