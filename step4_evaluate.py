"""Step 4: Evaluate Retention & Oracle QA (NO LLM, runs in ~2 seconds).

Combines community inputs (step 2) with whatever prompt condition summaries
exist (step 3: default, topology, control). Computes:
  - Node-level retention tables (L1, L2, within vs cut stratum)
  - Intervention stats (DiD lift, if topology and control exist)
  - Multi-hop QA accuracy using deterministic OracleReaderAnswerer
Saves full evaluated records to <out>/graphs/.

Usage:
    python step4_evaluate.py --config config_kaggle.yaml --out runs/exp1
"""
import argparse
import json
import math
from pathlib import Path

import yaml

from analysis import node_table
from interventions import intervention_stats
from qa import OracleReaderAnswerer, accuracy, evaluate_questions


def _clean(x):
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def rebuild_inputs(inputs_json):
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


def evaluate_graph(comm_data, available_runs):
    meta = comm_data["nodes"]
    l0 = {k: v for k, v in comm_data["level0"].items()}
    inputs = rebuild_inputs(comm_data["inputs"])
    questions = comm_data["questions"]
    for q in questions:
        q["hops"] = [tuple(h) for h in q["hops"]]

    # Run OracleReaderAnswerer (fast, deterministic, zero LLM calls)
    answerer = OracleReaderAnswerer()
    qa_records = {c: evaluate_questions(questions, l0, available_runs[c], inputs, answerer)
                  for c in available_runs}

    first_run = next(iter(available_runs.values()))
    first_summary_val = next(iter(first_run.values()))

    tokens = [r["prompt_tokens"] for c in available_runs for r in available_runs[c].values()]
    has_all_conditions = all(c in available_runs for c in ("default", "topology", "control"))

    return _clean({
        "p_out": comm_data["p_out"],
        "seed": comm_data["seed"],
        "backend": comm_data.get("backend", "ollama"),
        "model": comm_data.get("model", "qwen3:8b"),
        "conditions_evaluated": list(available_runs.keys()),
        "n_nodes": comm_data["n_nodes"],
        "n_triples": comm_data["n_triples"],
        "n_communities_l0": comm_data["n_communities_l0"],
        "n_communities_l1": comm_data["n_communities_l1"],
        "n_cut_triples": comm_data["n_cut_triples"],
        "max_prompt_tokens": max(tokens) if tokens else 0,
        "n_near_limit": sum(r.get("near_limit", False) for c in available_runs for r in available_runs[c].values()),
        "n_questions": comm_data["n_questions"],
        "tables": {c: node_table(inputs, available_runs[c], meta, l0) for c in available_runs},
        "intervention": intervention_stats(inputs, available_runs) if has_all_conditions else None,
        "qa_records": qa_records,
        "qa": {c: accuracy(qa_records[c]) for c in qa_records},
        "marked": {c: {str(k): v["marked"] for k, v in available_runs[c].items()} for c in available_runs},
        "summaries": {c: {str(k): v["summary"] for k, v in available_runs[c].items()} for c in available_runs},
    })


def main():
    ap = argparse.ArgumentParser(description="Step 4: Evaluate Retention & Oracle QA (no LLM)")
    ap.add_argument("--config", default="config_kaggle.yaml", help="config file")
    ap.add_argument("--out", required=True, help="output directory")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    comm_dir = Path(a.out) / "communities"
    sum_base = Path(a.out) / "summaries"
    graph_dir = Path(a.out) / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    if not comm_dir.exists():
        print(f"ERROR: {comm_dir}/ not found. Run step2_communities.py first.")
        return
    if not sum_base.exists():
        print(f"ERROR: {sum_base}/ not found. Run step3_summarize.py first.")
        return

    comm_files = sorted(comm_dir.glob("pout*_seed*.json"))
    print(f"Evaluating {len(comm_files)} graphs from {comm_dir}/...")

    done, skipped = 0, 0
    for f in comm_files:
        out_f = graph_dir / f.name

        # Gather whichever prompt conditions have been generated for this graph
        available_runs = {}
        for cond in ("default", "topology", "control"):
            cond_f = sum_base / cond / f.name
            if cond_f.exists():
                data = json.loads(cond_f.read_text())
                # Rebuild community dict
                comm_sums = {}
                for c_str, s_data in data["summaries"].items():
                    c = int(c_str) if c_str.isdigit() else c_str
                    comm_sums[c] = s_data
                available_runs[cond] = comm_sums

        if not available_runs:
            print(f"  {f.name}: NO summaries found in any condition under {sum_base}/, skipping")
            skipped += 1
            continue

        comm_data = json.loads(f.read_text())
        res = evaluate_graph(comm_data, available_runs)
        out_f.write_text(json.dumps(res))
        done += 1
        cond_str = ", ".join(available_runs.keys())
        qa_str = " | ".join(f"QA {c}: {res['qa'][c]['acc']*100:.1f}%" for c in available_runs)
        print(f"  {f.name} [{cond_str}]: evaluated ({qa_str})")

    print(f"\nStep 4 Complete! Evaluated {done} graphs (skipped {skipped}) into {graph_dir}/")


if __name__ == "__main__":
    main()
