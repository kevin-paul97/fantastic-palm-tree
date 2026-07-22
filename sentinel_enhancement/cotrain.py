"""
Co-train the LocationRegressor on EPIC + Sentinel-2 simultaneously.

Both datasets are the SAME task (Earth image -> lon/lat), so we train one shared
model on balanced batches: each step takes an EPIC batch and a Sentinel batch and
minimises  loss = MSE(epic) + LAMBDA * MSE(sentinel).  The 2500+ Sentinel scenes
regularise the conv features throughout training (rather than being overwritten
by the 284 EPIC images as in sequential fine-tuning), while EPIC is cycled to
parity so it keeps full gradient weight. Model selection is on the EPIC val set.

Resumable + time-budgeted for the kill window. Writes models/regressor_enhanced.pth.
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

LR = 1e-3
STEP_SIZE, GAMMA = 20, 0.5
TIME_BUDGET = float(__import__("os").environ.get("COTRAIN_BUDGET", "150"))
MODELS = E.MODELS
RESUME = MODELS / "_cotrain_resume.pth"
FINAL = MODELS / "regressor_enhanced.pth"


def preload(pairs, device):
    ds = E.PairDataset(pairs)
    xs, ys = zip(*[ds[i] for i in range(len(ds))])
    return torch.stack(xs).to(device), torch.stack(ys).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lam", type=float, default=0.5, help="weight on the Sentinel loss")
    ap.add_argument("--swa-frac", type=float, default=0.6,
                    help="start SWA averaging after this fraction of epochs")
    ap.add_argument("--swa-lr", type=float, default=4e-4, help="constant LR during SWA phase")
    args = ap.parse_args()

    device = E.get_device()
    E.set_seed()
    tr, va, te, _ = E.epic_splits()
    Xe, Ye = preload(tr, device)
    Xev, Yev = preload(va, device)
    sent = E.load_sentinel_pairs()
    Xs, Ys = preload(sent, device)
    print(f"co-train: EPIC {tuple(Xe.shape)}  Sentinel {tuple(Xs.shape)}  val {tuple(Xev.shape)}  "
          f"lambda={args.lam}", flush=True)

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
        print(f"resumed at epoch {start}/{args.epochs} (best={best_val:.5f})", flush=True)
    else:
        start, best_val = 0, float("inf")
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    bs = 32
    ne, ns = Xe.shape[0], Xs.shape[0]
    steps = (ns + bs - 1) // bs          # one Sentinel pass per epoch
    ge = torch.Generator().manual_seed(E.SEED)
    gs = torch.Generator().manual_seed(E.SEED + 1)

    def norm(t):
        return E.NORM.normalize(t)

    swa_start = int(args.swa_frac * args.epochs)
    swa_sum, swa_n = None, 0

    t0, ep = time.time(), start
    while ep < args.epochs:
        model.train()
        perm_s = torch.randperm(ns, generator=gs)
        perm_e = torch.randperm(ne, generator=ge)
        ei = 0
        tot = 0.0
        for si in range(steps):
            s_idx = perm_s[si * bs:(si + 1) * bs]
            if ei + bs > ne:                 # cycle EPIC, reshuffle
                perm_e = torch.randperm(ne, generator=ge); ei = 0
            e_idx = perm_e[ei:ei + bs]; ei += bs
            opt.zero_grad()
            le = crit(norm(model(Xe[e_idx])), norm(Ye[e_idx]))
            ls = crit(norm(model(Xs[s_idx])), norm(Ys[s_idx]))
            loss = le + args.lam * ls
            loss.backward(); opt.step()
            tot += le.item()
        if ep >= swa_start:            # constant LR during SWA phase for exploration
            for pg in opt.param_groups:
                pg["lr"] = args.swa_lr
        else:
            sched.step()

        model.eval()
        with torch.no_grad():
            vavg = crit(norm(model(Xev)), norm(Yev)).item()
        if vavg < best_val:
            best_val = vavg
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        ep += 1
        if ep > swa_start:  # Stochastic Weight Averaging over the tail epochs
            sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if swa_sum is None:
                swa_sum = sd
            else:
                for k in swa_sum:
                    swa_sum[k] += sd[k]
            swa_n += 1
        if ep % 5 == 0 or ep == args.epochs:
            print(f"  [cotrain] epoch {ep}/{args.epochs}  epic_train={tot/steps:.5f}  "
                  f"epic_val={vavg:.5f}  best={best_val:.5f}", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep,
                    "best_val": best_val, "best_state": best_state}, RESUME)
        if time.time() - t0 > TIME_BUDGET and ep < args.epochs:
            print(f"[cotrain] time budget hit at epoch {ep}; re-run to continue.", flush=True)
            return

    torch.save({"model_state_dict": best_state}, FINAL)
    if swa_sum is not None and swa_n > 0:
        swa_avg = {k: (v / swa_n) for k, v in swa_sum.items()}
        torch.save({"model_state_dict": swa_avg}, MODELS / "regressor_enhanced_swa.pth")
        print(f"SWA averaged over {swa_n} epochs -> regressor_enhanced_swa.pth", flush=True)
    RESUME.unlink(missing_ok=True)
    print(f"COTRAIN COMPLETE -> {FINAL} (best_epic_val={best_val:.5f})", flush=True)


if __name__ == "__main__":
    main()
