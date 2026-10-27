"""Step 1: Prepare all graphs (NO LLM needed, runs locally in seconds).

Builds the SBM graph, runs Leiden community detection, splits triples into
within/cut strata, and generates multi-hop questions. Saves one JSON per
graph seed under <out>/prepared/.

Usage:
    python step1_prepare.py --config config.yaml --out runs/exp1
"""
import argparse
import json
from pathlib import Path

import yaml

from communities import build_inputs, cut_triples, detect_hierarchy
from kg_gen import build_graph
from qa import make_questions


def prepare_graph(graph_cfg, experiment_cfg, p_out, seed):
    """Build one graph and save everything EXCEPT summaries."""
    g = dict(graph_cfg, p_out=p_out, seed=seed)
    try:
        data = build_graph(g)
    except ValueError:
        return None  # disconnected graph

    triples = [tuple(t) for t in data["triples"]]
    meta = data["nodes"]
    l0, l1 = detect_hierarchy(triples, seed)
    inputs = build_inputs(triples, l0, seed, experiment_cfg["max_triples"])
    questions = make_questions(triples, l0, seed, n=experiment_cfg["qa_n"],
                              node_meta=meta)

    # Convert inputs to JSON-safe format (tuples -> lists)
    inputs_json = {}
    for c, v in inputs.items():
        inputs_json[c] = {
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
        "level0": l0,
        "level1": l1,
        "inputs": inputs_json,
        "questions": questions,
    }


def main():
    ap = argparse.ArgumentParser(description="Step 1: prepare graphs (no LLM)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--n-graphs", type=int, help="override experiment.n_graphs")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    ex = cfg["experiment"]
    if a.n_graphs:
        ex["n_graphs"] = a.n_graphs

    out = Path(a.out) / "prepared"
    out.mkdir(parents=True, exist_ok=True)

    # Save config
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "config_used.yaml").write_text(yaml.safe_dump(cfg))

    skipped = []
    total = 0
    for p in ex["p_out_values"]:
        for seed in range(ex["seed_start"], ex["seed_start"] + ex["n_graphs"]):
            f = out / f"pout{p}_seed{seed}.json"
            if f.exists():
                print(f"  p_out={p} seed={seed}: already prepared, skipping")
                total += 1
                continue
            rec = prepare_graph(cfg["graph"], ex, p, seed)
            if rec is None:
                skipped.append({"p_out": p, "seed": seed})
                print(f"  p_out={p} seed={seed}: disconnected, skipped")
                continue
            f.write_text(json.dumps(rec))
            total += 1
            print(f"  p_out={p} seed={seed}: prepared ({rec['n_triples']} triples, "
                  f"{rec['n_communities_l0']} communities)")

    print(f"\nDone! Prepared {total} graphs in {out}/")
    if skipped:
        print(f"Skipped (disconnected): {skipped}")


if __name__ == "__main__":
    main()
