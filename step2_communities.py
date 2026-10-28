"""Step 2: Community Detection & Stratification (NO LLM, runs in ~2 seconds).

Loads raw graphs from step 1, runs Leiden hierarchical clustering (L0, L1),
splits triples into within-community and cut strata, and generates
multi-hop evaluation questions. Saves to <out>/communities/.

Usage:
    python step2_communities.py --config config_kaggle.yaml --out runs/exp1
"""
import argparse
import json
from pathlib import Path

import yaml

from communities import build_inputs, cut_triples, detect_hierarchy
from qa import make_questions


def process_communities(graph_raw, experiment_cfg):
    triples = [tuple(t) for t in graph_raw["triples"]]
    meta = graph_raw["nodes"]
    seed = graph_raw["seed"]
    p_out = graph_raw["p_out"]

    l0, l1 = detect_hierarchy(triples, seed)
    inputs = build_inputs(triples, l0, seed, experiment_cfg["max_triples"])
    questions = make_questions(triples, l0, seed, n=experiment_cfg["qa_n"], node_meta=meta)

    # Convert inputs to JSON-safe format
    inputs_json = {}
    for c, v in inputs.items():
        inputs_json[str(c)] = {
            "within": [list(t) for t in v["within"]],
            "cut": [list(t) for t in v["cut"]],
            "input": [[list(t), stratum] for t, stratum in v["input"]],
            "dropped": [list(t) for t in v["dropped"]],
        }

    return {
        "p_out": p_out,
        "seed": seed,
        "n_nodes": len(meta),
        "n_triples": len(triples),
        "n_communities_l0": len(set(l0.values())),
        "n_communities_l1": len(set(l1.values())),
        "n_cut_triples": len(cut_triples(triples, l0)),
        "n_questions": len(questions),
        "triples": [list(t) for t in triples],
        "nodes": meta,
        "level0": {str(k): v for k, v in l0.items()},
        "level1": {str(k): v for k, v in l1.items()},
        "inputs": inputs_json,
        "questions": questions,
    }


def main():
    ap = argparse.ArgumentParser(description="Step 2: Community Detection & Stratification (no LLM)")
    ap.add_argument("--config", default="config_kaggle.yaml", help="config file")
    ap.add_argument("--out", required=True, help="output directory")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    ex = cfg["experiment"]

    raw_dir = Path(a.out) / "graphs_raw"
    comm_dir = Path(a.out) / "communities"
    comm_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        print(f"ERROR: Raw graphs directory {raw_dir}/ not found. Run step1_graph.py first.")
        return

    raw_files = sorted(raw_dir.glob("pout*_seed*.json"))
    print(f"Processing Leiden communities for {len(raw_files)} graphs...")

    done, skipped = 0, 0
    for f in raw_files:
        out_f = comm_dir / f.name
        if out_f.exists():
            print(f"  {f.name}: already processed, skipping")
            skipped += 1
            continue

        raw = json.loads(f.read_text())
        res = process_communities(raw, ex)
        out_f.write_text(json.dumps(res))
        done += 1
        print(f"  {f.name}: {res['n_communities_l0']} communities, {res['n_cut_triples']} cut triples")

    print(f"\nStep 2 Complete! Processed {done} graphs (skipped {skipped}) in {comm_dir}/")


if __name__ == "__main__":
    main()
