"""
Evaluate original (shipped .bin), baseline (cached), and enhanced (fine-tuned)
on the identical held-out EPIC test split, then write comparison.json/.md.
"""

import os
import sys
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import experiment as E  # noqa: E402


def main():
    device = E.get_device()
    E.set_seed()
    tr, va, te, split_info = E.epic_splits()
    test_loader = E.DataLoader(E.PairDataset(te), batch_size=32, shuffle=False, num_workers=0)
    print(f"Test set: {len(te)} images / {len(split_info['test_dates'])} dates", flush=True)

    results = {"split": split_info, "models": {}}

    ship_bin = Path(os.environ.get(
        "PORTFOLIO_BIN",
        str(Path.home() / "Development/portfolio/public/models/regressor_weights.bin")))
    if ship_bin.exists():
        m = E.make_model(device)
        E.load_bin_into(m, ship_bin, device)
        results["models"]["original"] = E.evaluate(m, test_loader, device)
        results["original_source"] = str(ship_bin)

    base_ckpt = E.MODELS / "regressor_baseline.pth"
    if base_ckpt.exists():
        m = E.make_model(device)
        E.load_checkpoint_into(m, base_ckpt, device)
        results["models"]["baseline"] = E.evaluate(m, test_loader, device)

    enh_ckpt = E.MODELS / "regressor_enhanced.pth"
    if enh_ckpt.exists():
        m = E.make_model(device)
        E.load_checkpoint_into(m, enh_ckpt, device)
        results["models"]["enhanced"] = E.evaluate(m, test_loader, device)

    E.RESULTS.mkdir(exist_ok=True)
    (E.RESULTS / "comparison.json").write_text(json.dumps(results, indent=2))
    E.write_markdown(results)

    for k in ("original", "baseline", "enhanced"):
        if k in results["models"]:
            r = results["models"][k]
            print(f"  {k:<10} mean={r['mean_haversine_km']:7.0f} km  "
                  f"median={r['median_haversine_km']:6.0f} km  "
                  f"<=1000km={r['pct_within_1000km']:.0f}%", flush=True)
    print(f"Wrote {E.RESULTS/'comparison.json'} and comparison.md", flush=True)


if __name__ == "__main__":
    main()
