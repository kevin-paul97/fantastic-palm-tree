"""
Resumable, time-budgeted fine-tuning of the ENHANCED model.

Background jobs in this environment are killed after a few minutes, so this
script trains in short chunks: it checkpoints every epoch and exits cleanly
before a soft time budget, resuming from where it left off on the next run.

The schedule is IDENTICAL to the cached baseline (same lr, epochs, optimizer,
scheduler, seed) — the ONLY difference is that the conv feature extractor is
initialised from Sentinel-2 pre-training. That isolates the Copernicus effect.

Run repeatedly until it prints "ENHANCED FINE-TUNE COMPLETE".
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import experiment as E  # noqa: E402

TARGET_EPOCHS = 60
LR = 1e-3            # identical to baseline
STEP_SIZE, GAMMA = 20, 0.5
TIME_BUDGET = 130.0  # seconds per invocation, then checkpoint + exit

MODELS = E.MODELS
RESUME = MODELS / "_enhanced_resume.pth"
PRE_CKPT = MODELS / "_sentinel_pretrained.pth"
FINAL = MODELS / "regressor_enhanced.pth"


def preload(pairs, device):
    """Decode every image once into a single on-device tensor stack."""
    ds = E.PairDataset(pairs)
    xs, ys = [], []
    for i in range(len(ds)):
        x, y = ds[i]
        xs.append(x); ys.append(y)
    X = torch.stack(xs).to(device)
    Y = torch.stack(ys).to(device)
    return X, Y


def main():
    device = E.get_device()
    E.set_seed()
    tr, va, te, _ = E.epic_splits()

    print("preloading images into memory…", flush=True)
    Xtr, Ytr = preload(tr, device)
    Xva, Yva = preload(va, device)
    print(f"preloaded train {tuple(Xtr.shape)}  val {tuple(Xva.shape)}", flush=True)

    model = E.make_model(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=STEP_SIZE, gamma=GAMMA)
    crit = nn.MSELoss()

    if RESUME.exists():
        ck = torch.load(RESUME, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start = ck["epoch"]
        best_val = ck["best_val"]
        best_state = ck["best_state"]
        print(f"resumed at epoch {start}/{TARGET_EPOCHS} (best_val={best_val:.5f})", flush=True)
    else:
        if not PRE_CKPT.exists():
            print("ERROR: missing Sentinel pre-trained cache", flush=True)
            sys.exit(1)
        pre = E.make_model(device)
        E.load_checkpoint_into(pre, PRE_CKPT, device)
        n = E.transfer_conv_weights(model, pre.state_dict())
        print(f"init: transferred {n} conv tensors from Sentinel pre-training", flush=True)
        start = 0
        best_val = float("inf")
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    bs = 32
    n = Xtr.shape[0]
    g = torch.Generator(device="cpu").manual_seed(E.SEED)

    t0 = time.time()
    ep = start
    while ep < TARGET_EPOCHS:
        model.train()
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        nb = 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            x, y = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            loss = crit(E.NORM.normalize(model(x)), E.NORM.normalize(y))
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            vavg = crit(E.NORM.normalize(model(Xva)), E.NORM.normalize(Yva)).item()
        tot = tot / max(1, nb)
        if vavg < best_val:
            best_val = vavg
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        ep += 1
        if ep % 10 == 0 or ep == TARGET_EPOCHS:
            print(f"  [enhanced] epoch {ep}/{TARGET_EPOCHS}  train={tot:.5f}  "
                  f"val={vavg:.5f}  best={best_val:.5f}", flush=True)

        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep,
                    "best_val": best_val, "best_state": best_state}, RESUME)

        if time.time() - t0 > TIME_BUDGET and ep < TARGET_EPOCHS:
            print(f"time budget hit at epoch {ep}; checkpointed. Re-run to continue.", flush=True)
            return

    # done — save the best-val model in the standard checkpoint format
    torch.save({"model_state_dict": best_state}, FINAL)
    print(f"ENHANCED FINE-TUNE COMPLETE -> {FINAL} (best_val={best_val:.5f})", flush=True)


if __name__ == "__main__":
    main()
