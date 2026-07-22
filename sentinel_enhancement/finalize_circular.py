"""Final comparison including the circular-longitude enhanced model."""
import sys, json, os
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import experiment as E                      # noqa: E402
import cotrain_circular as C                # noqa: E402
import torch                                # noqa: E402

def main():
    dev = E.get_device(); E.set_seed()
    tr, va, te, split = E.epic_splits()
    dl = E.DataLoader(E.PairDataset(te), batch_size=32, shuffle=False)
    Xte, Yte = C.preload(te, dev)
    print(f"TEST set: {len(te)} images / {len(split['test_dates'])} dates {split['test_dates']}")
    res = {"split": split, "models": {}}

    # original (shipped .bin, 2-output raw)
    ship = Path.home()/"Development/portfolio/public/models/regressor_weights.bin"
    if ship.exists():
        m = E.make_model(dev); E.load_bin_into(m, ship, dev)
        res["models"]["original"] = E.evaluate(m, dl, dev)
    # baseline (2-output, EPIC only)
    if (E.MODELS/"regressor_baseline.pth").exists():
        m = E.make_model(dev); E.load_checkpoint_into(m, E.MODELS/"regressor_baseline.pth", dev)
        res["models"]["baseline"] = E.evaluate(m, dl, dev)
    # enhanced (circular, 3-output) — prefer SWA
    for path in ["regressor_enhanced_circular_swa.pth", "regressor_enhanced_circular.pth"]:
        p = E.MODELS/path
        if p.exists():
            m = C.make_model(dev)
            sd = torch.load(p, map_location=dev)["model_state_dict"]
            m.load_state_dict(sd)
            r = C.evaluate(m, Xte, Yte, dev)
            r["p25_haversine_km"] = 0.0
            res["models"]["enhanced"] = r
            res["enhanced_kind"] = "circular-longitude co-train + SWA"
            break

    E.RESULTS.mkdir(exist_ok=True)
    (E.RESULTS/"comparison.json").write_text(json.dumps(res, indent=2))
    for k in ("original","baseline","enhanced"):
        if k in res["models"]:
            r = res["models"][k]
            print(f"  {k:<10} mean={r['mean_haversine_km']:6.0f}  median={r['median_haversine_km']:6.0f}  "
                  f"p95={r['p95_haversine_km']:6.0f}  <=1000={r['pct_within_1000km']:.0f}%  <=2000={r['pct_within_2000km']:.0f}%")
    if "original" in res["models"] and "enhanced" in res["models"]:
        o=res["models"]["original"]["mean_haversine_km"]; e=res["models"]["enhanced"]["mean_haversine_km"]
        print(f"\n  Enhanced vs original: {o:.0f} -> {e:.0f} km  ({100*(o-e)/o:+.1f}%)  "
              f"{'*** ENHANCED WINS ***' if e<o else 'original still ahead'}")

if __name__ == "__main__":
    main()
