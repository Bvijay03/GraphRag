"""Step 3: Evaluate retention + QA (NO LLM needed, runs locally in seconds).

Loads prepared graphs (step1) and summaries (step2), computes:
  - Per-node retention tables (L1, L2, within/cut stratum breakdown)
  - Intervention stats (DiD: topology lift vs control lift)
  - Multi-hop QA accuracy using OracleReaderAnswerer (deterministic, no LLM)
Saves one full result JSON per graph under <out>/graphs/.

Usage:
    python step3_evaluate.py --config config.yaml --out runs/exp1
"""
import argparse
import json
import math
from pathlib import Path

import yaml

from analysis import node_table
from communities import cut_triples
from interventions import intervention_stats
from metrics import retained_l2
from qa import (OracleReaderAnswerer, accuracy, evaluate_questions,
                make_questions)


def _clean(x):
    """JSON-safe: NaN -> None."""
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def rebuild_inputs(inputs_json):
    """Convert JSON-loaded inputs back to tuple format."""
    out = {}
    for c_str, v in inputs_json.items():
        c = int(c_str) if c_str.isdigit() else c_str
        out[c] = {
            "within": [tuple(t) for t in v["within"]],
            "cut": [tuple(t) for t in v["cut"]],
            "input": [(tuple(t), stratum) for t, stratum in v["input"]],
            "dropped": [tuple(t) for t in v["dropped"]],
        }
    return out


def rebuild_summaries(summaries_json):
    """Convert JSON-loaded summaries back to the format analysis expects."""
    out = {}
    for cond, communities in summaries_json.items():
        out[cond] = {}
        for c_str, data in communities.items():
            c = int(c_str) if c_str.isdigit() else c_str
            out[cond][c] = data
    return out


def evaluate_one(prep_file, sum_file):
    """Combine prepared data + summaries -> full evaluation result."""
    prep = json.loads(prep_file.read_text())
    sums = json.loads(sum_file.read_text())

    meta = prep["nodes"]
    triples = [tuple(t) for t in prep["triples"]]
    l0 = {k: v for k, v in prep["level0"].items()}
    inputs = rebuild_inputs(prep["inputs"])
    runs = rebuild_summaries(sums["summaries"])

    # Rebuild questions from the prepared data
    questions = prep["questions"]
    # Convert question hops back to tuples
    for q in questions:
        q["hops"] = [tuple(h) for h in q["hops"]]

    # Use OracleReaderAnswerer (deterministic, no LLM needed)
    answerer = OracleReaderAnswerer()
    qa_records = {c: evaluate_questions(questions, l0, runs[c], inputs, answerer)
                  for c in runs}

    tokens = [r["prompt_tokens"] for c in runs for r in runs[c].values()]
    return _clean({
        "p_out": prep["p_out"],
        "seed": prep["seed"],
        "backend": sums["backend"],
        "model": sums["model"],
        "n_nodes": prep["n_nodes"],
        "n_triples": prep["n_triples"],
        "n_communities_l0": prep["n_communities_l0"],
        "n_communities_l1": prep["n_communities_l1"],
        "n_cut_triples": prep["n_cut_triples"],
        "max_prompt_tokens": max(tokens) if tokens else 0,
        "n_near_limit": sum(r["near_limit"] for c in runs for r in runs[c].values()),
        "n_questions": prep["n_questions"],
        "tables": {c: node_table(inputs, runs[c], meta, l0) for c in runs},
        "intervention": intervention_stats(inputs, runs) if len(runs) == 3 else None,
        "qa_records": qa_records,
        "qa": {c: accuracy(qa_records[c]) for c in qa_records},
        "marked": {c: {str(k): v["marked"] for k, v in runs[c].items()} for c in runs},
        "summaries": {c: {str(k): v["summary"] for k, v in runs[c].items()} for c in runs},
    })


def main():
    ap = argparse.ArgumentParser(description="Step 3: evaluate retention + QA (no LLM)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", required=True, help="same output dir as step1/step2")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    prep_dir = Path(a.out) / "prepared"
    sum_dir = Path(a.out) / "summaries"
    graph_dir = Path(a.out) / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    if not prep_dir.exists():
        print("ERROR: No prepared data found. Run step1_prepare.py first.")
        return
    if not sum_dir.exists():
        print("ERROR: No summaries found. Run step2_summarize.py first.")
        return

    prep_files = sorted(prep_dir.glob("pout*_seed*.json"))
    done, missing = 0, 0
    for prep_file in prep_files:
        sum_file = sum_dir / prep_file.name
        out_file = graph_dir / prep_file.name

        if out_file.exists():
            print(f"  {prep_file.name}: already evaluated, skipping")
            done += 1
            continue

        if not sum_file.exists():
            print(f"  {prep_file.name}: NO SUMMARY FOUND, skipping")
            missing += 1
            continue

        print(f"  {prep_file.name}: evaluating...", end=" ", flush=True)
        result = evaluate_one(prep_file, sum_file)
        out_file.write_text(json.dumps(result))
        done += 1
        print("done")

    print(f"\nDone! Evaluated {done} graphs. Missing summaries: {missing}.")
    print(f"Results saved in {graph_dir}/")
    if missing:
        print(f"Run step2_summarize.py to generate the {missing} missing summaries.")


if __name__ == "__main__":
    main()
