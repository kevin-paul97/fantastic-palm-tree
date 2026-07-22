"""
Publish experiment results to the portfolio and recommend a deployment.

- Copies results/comparison.json -> portfolio/public/satellite-metrics.json
  (with a generated timestamp) so the /satellite page shows real numbers.
- Prints a summary table and a recommendation on which weights to ship.
"""

import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMP = HERE / "results" / "comparison.json"
PORTFOLIO_METRICS = Path.home() / "Development/portfolio/public/satellite-metrics.json"


def main():
    if not COMP.exists():
        print("no comparison.json — run experiment.py first")
        return
    data = json.loads(COMP.read_text())
    data["generated"] = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    PORTFOLIO_METRICS.write_text(json.dumps(data, indent=2))
    print(f"published -> {PORTFOLIO_METRICS}")

    m = data["models"]
    print("\nModel                         mean_km  median_km  <=1000km")
    for k in ("original", "baseline", "enhanced"):
        if k in m:
            r = m[k]
            print(f"  {k:<26} {r['mean_haversine_km']:7.0f}  {r['median_haversine_km']:8.0f}  "
                  f"{r['pct_within_1000km']:6.0f}%")

    if "baseline" in m and "enhanced" in m:
        b = m["baseline"]["mean_haversine_km"]
        e = m["enhanced"]["mean_haversine_km"]
        print(f"\nSentinel effect (clean, identical data): "
              f"{b:.0f} -> {e:.0f} km  ({100*(b-e)/b:+.1f}%)")

    # recommendation: ship whichever trained model beats the original, else keep original
    ship = None
    if "enhanced" in m:
        cands = {k: m[k]["mean_haversine_km"] for k in ("enhanced", "baseline") if k in m}
        best_trained = min(cands, key=cands.get)
        orig = m.get("original", {}).get("mean_haversine_km", float("inf"))
        if cands[best_trained] < orig:
            ship = best_trained
    print(f"\nRECOMMENDATION: {'ship models/regressor_' + ship + '.pth' if ship else 'keep original shipped weights'}")


if __name__ == "__main__":
    main()
