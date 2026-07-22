"""
Final scoreboard + journey, written to the portfolio metrics file.

Evaluates on the pinned EPIC test split:
  original          shipped .bin (2-out raw longitude)
  baseline          EPIC-only from scratch (2-out)
  circular          single circular co-train + SWA (3-out)
  ensemble          averaged ensemble of all regressor_ens_s*.pth members
"""
import sys, os, json, time
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import experiment as E                        # noqa: E402
import cotrain_circular as C                  # noqa: E402
import ensemble as ENS                        # noqa: E402
from cotrain_v2 import BetterRegressor, decode  # noqa: E402

PORTFOLIO = Path.home() / "Development/portfolio/public/satellite-metrics.json"


def metrics_from_hav(hav):
    return dict(n=int(len(hav)),
                mean_haversine_km=float(hav.mean()),
                median_haversine_km=float(hav.median()),
                p95_haversine_km=float(hav.quantile(0.95)),
                pct_within_1000km=float((hav <= 1000).float().mean() * 100),
                pct_within_2000km=float((hav <= 2000).float().mean() * 100))


def main():
    device = E.get_device(); E.set_seed()
    tr, va, te, split = E.epic_splits()
    dl = E.DataLoader(E.PairDataset(te), batch_size=32, shuffle=False)
    models = {}

    ship = Path.home() / "Development/portfolio/public/models/regressor_weights.bin"
    if ship.exists():
        m = E.make_model(device); E.load_bin_into(m, ship, device)
        models["original"] = E.evaluate(m, dl, device)
    if (E.MODELS / "regressor_baseline.pth").exists():
        m = E.make_model(device); E.load_checkpoint_into(m, E.MODELS / "regressor_baseline.pth", device)
        models["baseline"] = E.evaluate(m, dl, device)

    # single circular (3-out) at 64px
    Xte64, Yte64 = C.preload(te, device)
    for p in ["regressor_enhanced_circular_swa.pth", "regressor_enhanced_circular.pth"]:
        if (E.MODELS / p).exists():
            m = C.make_model(device); m.load_state_dict(torch.load(E.MODELS / p, map_location=device)["model_state_dict"])
            models["circular"] = C.evaluate(m, Xte64, Yte64, device); break

    # ensemble of BetterRegressor members at 64px
    data, meta = ENS.load_all(device)
    Xte, Yte = data["Xte"], data["Yte"]
    members = sorted(E.MODELS.glob("regressor_ens_s*.pth"))
    if members:
        accum = None
        for mp in members:
            mm = BetterRegressor().to(device)
            mm.load_state_dict(torch.load(mp, map_location=device)["model_state_dict"])
            p = ENS.member_pred(mm, Xte, tta=False).cpu()
            accum = p if accum is None else accum + p
        preds = decode(accum / len(members))
        hav = E.NORM.compute_haversine_distance(preds, Yte.cpu())
        models["ensemble"] = metrics_from_hav(hav)
        models["ensemble"]["members"] = len(members)
        models["enhanced"] = dict(models["ensemble"])  # alias for the page scoreboard

    out = {"models": models, "split": split,
           "generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
           "enhanced_kind": "ensemble of circular co-train members"}
    (E.RESULTS / "comparison.json").write_text(json.dumps(out, indent=2))
    PORTFOLIO.write_text(json.dumps(out, indent=2))
    for k in ("original", "baseline", "circular", "ensemble"):
        if k in models:
            r = models[k]
            extra = f" ({r['members']} members)" if k == "ensemble" else ""
            print(f"  {k:<9} mean={r['mean_haversine_km']:6.0f}  median={r['median_haversine_km']:6.0f}  "
                  f"p95={r['p95_haversine_km']:6.0f}  <=2000={r['pct_within_2000km']:.0f}%{extra}")
    print(f"published -> {PORTFOLIO}")


if __name__ == "__main__":
    main()
