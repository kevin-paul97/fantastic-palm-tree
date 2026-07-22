"""
Ensemble push toward sub-200 km — the honest way (no test-set cherry-picking).

Protocol, fixed up front:
  * Members: K deeper GroupNorm CNNs with the circular-longitude head, each
    co-trained on EPIC + Sentinel-2 with SWA, differing only in random seed
    (which also reshuffles the train/val split — the TEST set is pinned).
  * The ensemble prediction is the average of ALL members in (sin, cos, lat)
    space — every trained member counts; none are dropped.
  * Optional light test-time augmentation (multi-scale) averaged in the same space.

Images are decoded once (cache keyed by resolution) and the per-seed split is
done in-memory by index, so adding members costs no extra decoding.

  python ensemble.py train   # trains any not-yet-done members (resumable), then
  python ensemble.py eval     # averages members + reports the single ensemble number
"""

import sys
import os
import time
import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import experiment as E                     # noqa: E402
from cotrain_v2 import BetterRegressor, encode, decode, preload  # noqa: E402

DEG = math.pi / 180.0
SIZE = int(os.environ.get("ENS_SIZE", "96"))
EPOCHS = int(os.environ.get("ENS_EPOCHS", "90"))
LAM = float(os.environ.get("ENS_LAM", "0.4"))
SWA_FRAC = float(os.environ.get("ENS_SWA_FRAC", "0.4"))
SWA_LR = float(os.environ.get("ENS_SWA_LR", "5e-4"))
STEP_SIZE, GAMMA, LR = 25, 0.5, 1e-3
SEEDS = [int(s) for s in os.environ.get("ENS_SEEDS", "42,43,44,45,46").split(",")]
TIME_BUDGET = float(os.environ.get("COTRAIN_BUDGET", "530"))
MODELS = E.MODELS
CACHE = MODELS / f"_enscache_s{SIZE}.pt"


def load_all(device):
    """Preload every image once: full non-test EPIC pool, pinned test, Sentinel."""
    if CACHE.exists():
        d = torch.load(CACHE, map_location="cpu")
        meta = d.pop("_meta")
        return {k: v.to(device) for k, v in d.items()}, meta
    E.set_seed(42)
    import experiment as EE
    by_date = EE.load_epic_pairs_by_date()
    test_dates = [x for x in EE.FIXED_TEST_DATES if x in by_date]
    pool = [d for d in sorted(by_date) if d not in set(test_dates)]
    pool_pairs = [p for d in pool for p in by_date[d]]
    test_pairs = [p for d in test_dates for p in by_date[d]]
    sent = EE.load_sentinel_pairs()
    print(f"decoding {len(pool_pairs)} pool + {len(test_pairs)} test + {len(sent)} sentinel at {SIZE}px…", flush=True)
    Xp, Yp = preload(pool_pairs, device, SIZE)
    Xte, Yte = preload(test_pairs, device, SIZE)
    Xs, Ys = preload(sent, device, SIZE)
    data = {"Xp": Xp, "Yp": Yp, "Xte": Xte, "Yte": Yte, "Xs": Xs, "Ys": Ys}
    meta = {"n_pool": len(pool_pairs), "n_test": len(test_pairs), "n_sent": len(sent)}
    torch.save({**{k: v.cpu() for k, v in data.items()}, "_meta": meta}, CACHE)
    return data, meta


def split_pool(n, seed, val_frac=0.1):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    nv = max(1, int(n * val_frac))
    return perm[nv:], perm[:nv]  # train idx, val idx


def train_member(seed, data, device):
    member_path = MODELS / f"regressor_ens_s{seed}.pth"
    resume = MODELS / f"_ens_s{seed}_resume.pth"
    if member_path.exists():
        return member_path

    Xp, Yp, Xs, Ys = data["Xp"], data["Yp"], data["Xs"], data["Ys"]
    tr_idx, va_idx = split_pool(Xp.shape[0], seed)
    Xe, Ye = Xp[tr_idx], Yp[tr_idx]
    Xev, Yev = Xp[va_idx], Yp[va_idx]
    Te_e, Te_ev, Te_s = encode(Ye), encode(Yev), encode(Ys)

    E.set_seed(seed)
    model = BetterRegressor().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=STEP_SIZE, gamma=GAMMA)
    crit = nn.MSELoss()

    if resume.exists():
        ck = torch.load(resume, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start, best_val = ck["epoch"], ck["best_val"]
        # Keep the running-average buffers on CPU (we accumulate CPU clones into them).
        best_state = {k: v.cpu() for k, v in ck["best_state"].items()}
        swa_sum = None if ck["swa_sum"] is None else {k: v.cpu() for k, v in ck["swa_sum"].items()}
        swa_n = ck["swa_n"]
    else:
        start, best_val = 0, float("inf")
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        swa_sum, swa_n = None, 0

    bs, ne, ns = 32, Xe.shape[0], Xs.shape[0]
    steps = (ns + bs - 1) // bs
    ge = torch.Generator().manual_seed(seed)
    gs = torch.Generator().manual_seed(seed + 1)
    swa_start = int(SWA_FRAC * EPOCHS)
    t0, ep = time.time(), start
    while ep < EPOCHS:
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
            (le + LAM * ls).backward(); opt.step()
        if ep >= swa_start:
            for pg in opt.param_groups:
                pg["lr"] = SWA_LR
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
        if ep % 10 == 0 or ep == EPOCHS:
            print(f"  [ens s{seed}] ep {ep}/{EPOCHS} val={vavg:.5f} best={best_val:.5f}", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep, "best_val": best_val, "best_state": best_state,
                    "swa_sum": swa_sum, "swa_n": swa_n}, resume)
        if time.time() - t0 > TIME_BUDGET and ep < EPOCHS:
            print(f"  [ens s{seed}] budget hit at {ep}; re-run to continue.", flush=True)
            return None

    swa = {k: v / swa_n for k, v in swa_sum.items()} if swa_n else best_state
    torch.save({"model_state_dict": swa, "size": SIZE}, member_path)
    resume.unlink(missing_ok=True)
    print(f"  [ens s{seed}] DONE -> {member_path.name}", flush=True)
    return member_path


@torch.no_grad()
def member_pred(model, X, tta=False):
    """Return raw [sin,cos,lat] outputs, optionally multi-scale TTA-averaged."""
    model.eval()
    outs = [model(X)]
    if tta:
        for s in (0.9, 1.1):
            n = max(8, int(round(X.shape[-1] * s)))
            Xs = torch.nn.functional.interpolate(X, size=(n, n), mode="bilinear", align_corners=False)
            Xs = torch.nn.functional.interpolate(Xs, size=(X.shape[-1], X.shape[-1]), mode="bilinear", align_corners=False)
            outs.append(model(Xs))
    o = torch.stack(outs).mean(0)
    # renormalise the (sin,cos) pair
    norm = torch.sqrt(o[:, 0] ** 2 + o[:, 1] ** 2).clamp_min(1e-6)
    return torch.stack([o[:, 0] / norm, o[:, 1] / norm, o[:, 2]], dim=1)


def do_eval(data, device, tta):
    Xte, Yte = data["Xte"], data["Yte"]
    members = sorted(MODELS.glob("regressor_ens_s*.pth"))
    if not members:
        print("no members trained yet"); return
    accum = None
    for mp in members:
        m = BetterRegressor().to(device)
        m.load_state_dict(torch.load(mp, map_location=device)["model_state_dict"])
        p = member_pred(m, Xte, tta=tta).cpu()
        accum = p if accum is None else accum + p
    ens = accum / len(members)
    preds = decode(ens)
    hav = E.NORM.compute_haversine_distance(preds, Yte.cpu())
    print(f"ENSEMBLE of {len(members)} members{' +TTA' if tta else ''}: "
          f"mean=%.0f median=%.0f p95=%.0f <=1000=%.0f%% <=2000=%.0f%%" % (
              hav.mean(), hav.median(), hav.quantile(0.95),
              (hav <= 1000).float().mean() * 100, (hav <= 2000).float().mean() * 100), flush=True)
    return float(hav.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()
    device = E.get_device()
    data, meta = load_all(device)
    print(f"data: pool={meta['n_pool']} test={meta['n_test']} sentinel={meta['n_sent']} size={SIZE}", flush=True)
    if args.cmd == "train":
        for seed in SEEDS:
            if (MODELS / f"regressor_ens_s{seed}.pth").exists():
                print(f"  seed {seed} already done", flush=True); continue
            r = train_member(seed, data, device)
            if r is None:
                return  # budget hit — re-run to continue this seed
        print("ALL MEMBERS DONE", flush=True)
        do_eval(data, device, tta=False)
    else:
        do_eval(data, device, tta=args.tta)


if __name__ == "__main__":
    main()
