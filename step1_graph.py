"""Step 1: Generate Knowledge Graphs (NO LLM, runs in ~1 second).

Generates synthetic graphs under the Stochastic Block Model (SBM) with
degree-matched planted bridges and hubs. Saves raw graph structures (nodes,
edges, roles, degrees, betweenness) to <out>/graphs_raw/.

Usage:
    python step1_graph.py --config config_kaggle.yaml --out runs/exp1
"""
import argparse
import json
from pathlib import Path

import yaml

from kg_gen import build_graph


def main():
    ap = argparse.ArgumentParser(description="Step 1: Generate raw graphs (no LLM)")
    ap.add_argument("--config", default="config_kaggle.yaml", help="config file")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--n-graphs", type=int, help="override experiment.n_graphs")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    ex = cfg["experiment"]
    if a.n_graphs:
        ex["n_graphs"] = a.n_graphs

    out = Path(a.out) / "graphs_raw"
    out.mkdir(parents=True, exist_ok=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "config_used.yaml").write_text(yaml.safe_dump(cfg))

    total = 0
    skipped = []
    print(f"Generating graphs into {out}/")
    for p in ex["p_out_values"]:
        for seed in range(ex["seed_start"], ex["seed_start"] + ex["n_graphs"]):
            f = out / f"pout{p}_seed{seed}.json"
            if f.exists():
                print(f"  p_out={p} seed={seed}: already exists, skipping")
                total += 1
                continue

            g_cfg = dict(cfg["graph"], p_out=p, seed=seed)
            try:
                data = build_graph(g_cfg)
            except ValueError:
                print(f"  p_out={p} seed={seed}: disconnected, skipped")
                skipped.append({"p_out": p, "seed": seed})
                continue

            rec = {
                "p_out": p,
                "seed": seed,
                "n_nodes": len(data["nodes"]),
                "n_triples": len(data["triples"]),
                "nodes": data["nodes"],
                "triples": data["triples"],
            }
            f.write_text(json.dumps(rec))
            total += 1
            print(f"  p_out={p} seed={seed}: generated ({rec['n_nodes']} nodes, {rec['n_triples']} triples)")

    print(f"\nStep 1 Complete! {total} graphs saved in {out}/")
    if skipped:
        print(f"Skipped disconnected seeds: {skipped}")


if __name__ == "__main__":
    main()
