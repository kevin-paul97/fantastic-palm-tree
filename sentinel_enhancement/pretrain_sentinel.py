"""
Fast, resumable pre-training of the conv feature extractor on the Sentinel-2
auxiliary geolocation task (image -> scene-center lon/lat).

Preloads all Sentinel images into device memory so epochs are fast. Saves the
best-val model to models/_sentinel_pretrained.pth for run_finetune.py --mode
enhanced to transfer from. Resumable + time-budgeted for the kill window.
"""

import sys
import time
import random
import argparse
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import experiment as E  # noqa: E402

LR = 1e-3
STEP_SIZE, GAMMA = 25, 0.5
TIME_BUDGET = 150.0

MODELS = E.MODELS
FINAL = MODELS / "_sentinel_pretrained.pth"
RESUME = MODELS / "_sentinel_resume.pth"


def preload(pairs, device):
    ds = E.PairDataset(pairs)
    xs, ys = zip(*[ds[i] for i in range(len(ds))])
    return torch.stack(xs).to(device), torch.stack(ys).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    args = ap.parse_args()

    device = E.get_device()
    E.set_seed()
    pairs = E.load_sentinel_pairs()
    if len(pairs) < 100:
        print(f"only {len(pairs)} sentinel pairs — need >=100", flush=True); sys.exit(1)
    rng = random.Random(E.SEED); rng.shuffle(pairs)
    n_val = max(32, int(len(pairs) * 0.1))
    va, tr = pairs[:n_val], pairs[n_val:]
    Xtr, Ytr = preload(tr, device)
    Xva, Yva = preload(va, device)
    print(f"[sentinel] train {tuple(Xtr.shape)}  val {tuple(Xva.shape)}", flush=True)

    E.set_seed()
    model = E.make_model(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=STEP_SIZE, gamma=GAMMA)
    crit = nn.MSELoss()

    if RESUME.exists():
        ck = torch.load(RESUME, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start, best_val, best_state = ck["epoch"], ck["best_val"], ck["best_state"]
        print(f"[sentinel] resumed at epoch {start}/{args.epochs} (best={best_val:.5f})", flush=True)
    else:
        start, best_val = 0, float("inf")
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    bs, n = 32, Xtr.shape[0]
    g = torch.Generator().manual_seed(E.SEED)
    t0, ep = time.time(), start
    while ep < args.epochs:
        model.train()
        perm = torch.randperm(n, generator=g)
        tot = nb = 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = crit(E.NORM.normalize(model(Xtr[idx])), E.NORM.normalize(Ytr[idx]))
            loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        model.eval()
        with torch.no_grad():
            vavg = crit(E.NORM.normalize(model(Xva)), E.NORM.normalize(Yva)).item()
        if vavg < best_val:
            best_val = vavg
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        ep += 1
        if ep % 10 == 0 or ep == args.epochs:
            print(f"  [sentinel] epoch {ep}/{args.epochs}  train={tot/max(1,nb):.5f}  "
                  f"val={vavg:.5f}  best={best_val:.5f}", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep,
                    "best_val": best_val, "best_state": best_state}, RESUME)
        if time.time() - t0 > TIME_BUDGET and ep < args.epochs:
            print(f"[sentinel] time budget hit at epoch {ep}; re-run to continue.", flush=True)
            return

    torch.save({"model_state_dict": best_state}, FINAL)
    print(f"SENTINEL PRETRAIN COMPLETE -> {FINAL} (best_val={best_val:.5f}, {n} train imgs)", flush=True)


if __name__ == "__main__":
    main()
