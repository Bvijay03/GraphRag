"""Step 4: Aggregate results and write summary.md (NO LLM, runs in 1 second).

Loads evaluated graph JSONs from step3, aggregates across graphs,
runs statistical tests (t-tests, Holm correction, sign tests),
and writes results.json + summary.md.

Usage:
    python step4_report.py --config config.yaml --out runs/exp1
"""
import argparse
import json
from pathlib import Path

import yaml

from run_experiment import aggregate, write_summary


def main():
    ap = argparse.ArgumentParser(description="Step 4: aggregate + write report (no LLM)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", required=True, help="same output dir as step1/step2/step3")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    ex = cfg["experiment"]
    graph_dir = Path(a.out) / "graphs"

    if not graph_dir.exists():
        print("ERROR: No evaluated graphs found. Run step3_evaluate.py first.")
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

    out = Path(a.out)
    (out / "results.json").write_text(json.dumps(by_p, indent=2))
    write_summary(out / "summary.md", cfg, by_p, skipped)

    print(f"\nDone! Report written to:")
    print(f"  {out / 'summary.md'}")
    print(f"  {out / 'results.json'}")
    if skipped:
        print(f"Missing graphs (skipped): {len(skipped)}")


if __name__ == "__main__":
    main()
