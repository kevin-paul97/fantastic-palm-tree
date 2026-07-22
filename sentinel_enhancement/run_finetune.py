"""
Train the baseline OR the enhanced model through IDENTICAL code so the ONLY
difference is the convolutional initialisation:

  --mode baseline  -> random init (Kaiming/Xavier)
  --mode enhanced  -> conv init transferred from Sentinel-2 pre-training

Everything else (data, epochs, lr, optimizer, scheduler, seed, minibatch order,
best-val model selection) is shared. Images are pre-loaded once into device
memory so an epoch is fast. Resumable + time-budgeted because background jobs in
this environment get killed after a few minutes; re-run until it prints
"<MODE> COMPLETE".
"""

import sys
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import experiment as E  # noqa: E402

TARGET_EPOCHS = 60
LR = 1e-3
STEP_SIZE, GAMMA = 20, 0.5
TIME_BUDGET = 150.0

MODELS = E.MODELS
PRE_CKPT = MODELS / "_sentinel_pretrained.pth"


def preload(pairs, device):
    ds = E.PairDataset(pairs)
    xs, ys = zip(*[ds[i] for i in range(len(ds))])
    return torch.stack(xs).to(device), torch.stack(ys).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "enhanced"], required=True)
    ap.add_argument("--freeze-conv", action="store_true",
                    help="freeze conv layers, train only the FC head (enhanced only)")
    args = ap.parse_args()
    mode = args.mode
    tag = mode + ("_frozen" if args.freeze_conv else "")
    resume_path = MODELS / f"_{tag}_resume.pth"
    final_path = MODELS / f"regressor_{mode}.pth"

    device = E.get_device()
    E.set_seed()
    tr, va, te, _ = E.epic_splits()
    Xtr, Ytr = preload(tr, device)
    Xva, Yva = preload(va, device)
    print(f"[{mode}] preloaded train {tuple(Xtr.shape)}  val {tuple(Xva.shape)}", flush=True)

    E.set_seed()  # identical init RNG state for both modes
    model = E.make_model(device)
    resuming = resume_path.exists()

    if not resuming and mode == "enhanced":
        if not PRE_CKPT.exists():
            print("ERROR: missing Sentinel pre-trained cache", flush=True); sys.exit(1)
        pre = E.make_model(device)
        E.load_checkpoint_into(pre, PRE_CKPT, device)
        k = E.transfer_conv_weights(model, pre.state_dict())
        print(f"[enhanced] transferred {k} conv tensors from Sentinel pre-training", flush=True)
    elif not resuming:
        print("[baseline] random conv init", flush=True)

    if args.freeze_conv:
        for cl in model.conv_layers:
            for p in cl.parameters():
                p.requires_grad = False
        print("[frozen] conv layers frozen — training FC head only", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=STEP_SIZE, gamma=GAMMA)
    crit = nn.MSELoss()

    if resuming:
        ck = torch.load(resume_path, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start, best_val, best_state = ck["epoch"], ck["best_val"], ck["best_state"]
        print(f"[{tag}] resumed at epoch {start}/{TARGET_EPOCHS} (best={best_val:.5f})", flush=True)
    else:
        start, best_val = 0, float("inf")
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    bs, n = 32, Xtr.shape[0]
    g = torch.Generator().manual_seed(E.SEED)
    t0, ep = time.time(), start
    while ep < TARGET_EPOCHS:
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
        if ep % 10 == 0 or ep == TARGET_EPOCHS:
            print(f"  [{mode}] epoch {ep}/{TARGET_EPOCHS}  train={tot/max(1,nb):.5f}  "
                  f"val={vavg:.5f}  best={best_val:.5f}", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep,
                    "best_val": best_val, "best_state": best_state}, resume_path)
        if time.time() - t0 > TIME_BUDGET and ep < TARGET_EPOCHS:
            print(f"[{mode}] time budget hit at epoch {ep}; checkpointed. Re-run to continue.", flush=True)
            return

    torch.save({"model_state_dict": best_state}, final_path)
    print(f"{mode.upper()} COMPLETE -> {final_path} (best_val={best_val:.5f})", flush=True)


if __name__ == "__main__":
    main()
