"""Step 5: Statistical Reporting (NO LLM, runs in ~1 second).

Aggregates evaluated graph records across seeds, runs statistical tests
(t-tests, Holm family-wise correction, sign tests for QA),
and writes results.json and summary.md.

Usage:
    python step5_report.py --config config_kaggle.yaml --out runs/exp1
"""
import argparse
import json
from pathlib import Path

import yaml

from run_experiment import aggregate, write_summary


def main():
    ap = argparse.ArgumentParser(description="Step 5: Statistical Reporting (no LLM)")
    ap.add_argument("--config", default="config_kaggle.yaml", help="config file")
    ap.add_argument("--out", required=True, help="output directory")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    ex = cfg["experiment"]
    graph_dir = Path(a.out) / "graphs"

    if not graph_dir.exists():
        print(f"ERROR: Evaluated graphs directory {graph_dir}/ not found. Run step4_evaluate.py first.")
        return

    by_p, skipped = {}, []
    for p in ex["p_out_values"]:
        records = []
        for seed in range(ex["seed_start"], ex["seed_start"] + ex["n_graphs"]):
            f = graph_dir / f"pout{p}_seed{seed}.json"
            if f.exists():
                records.append(json.loads(f.read_text()))
            else:
                skipped.append({"p_out": p, "seed": seed})
        if records:
            by_p[str(p)] = aggregate(records)
            print(f"  p_out={p}: aggregated {len(records)} graphs")

    if not by_p:
        print("ERROR: No evaluated graph records found matching the config seeds and p_out values.")
        return

    out = Path(a.out)
    (out / "results.json").write_text(json.dumps(by_p, indent=2))
    write_summary(out / "summary.md", cfg, by_p, skipped)

    print(f"\nStep 5 Complete! Reports written to:")
    print(f"  {out / 'summary.md'}")
    print(f"  {out / 'results.json'}")
    if skipped:
        print(f"Missing graphs (skipped): {len(skipped)}")


if __name__ == "__main__":
    main()
