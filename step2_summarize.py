"""Step 2: Run LLM summaries (THIS is the only step that needs a GPU).

Loads prepared graphs from step1, sends community triples to the LLM under
3 prompt conditions (default, topology, control), and saves the raw
summaries. One JSON per graph seed under <out>/summaries/.

Usage (on Kaggle or GPU machine):
    python step2_summarize.py --config config.yaml --out runs/exp1
    python step2_summarize.py --config config.yaml --out runs/exp1 --backend ollama
"""
import argparse
import json
import time
from pathlib import Path

import yaml

from communities import build_inputs
from interventions import run_conditions
from llm import OllamaLLM, MockLLM
from planted_mock import MarkerResponsiveMock
from summarize import summarize_level0


def build_llm(llm_cfg, meta, seed):
    b = llm_cfg["backend"]
    if b == "mock":
        return MockLLM(llm_cfg["mock_keep_fraction"], seed)
    if b == "planted":
        return MarkerResponsiveMock(meta, seed=seed, **llm_cfg["planted"])
    if b == "ollama":
        return OllamaLLM(llm_cfg["model"], llm_cfg["num_ctx"], llm_cfg["temperature"],
                         llm_cfg["seed"], llm_cfg["think"])
    if b == "template":
        return None
    raise ValueError(f"unknown backend: {b}")


def rebuild_inputs(inputs_json):
    """Convert JSON-loaded inputs back to the tuple format the code expects."""
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


def summarize_one(prep_file, llm_cfg, experiment_cfg):
    """Load a prepared graph and generate summaries under all conditions."""
    prep = json.loads(prep_file.read_text())
    meta = prep["nodes"]
    inputs = rebuild_inputs(prep["inputs"])

    llm = build_llm(llm_cfg, meta, prep["seed"])
    limit = experiment_cfg["context_limit"] or (
        llm_cfg["num_ctx"] if llm_cfg["backend"] == "ollama" else None)

    if llm_cfg["backend"] == "template":
        runs = {"default": summarize_level0(None, inputs, meta, use_template=True)}
    else:
        runs = run_conditions(llm, inputs, meta, experiment_cfg["k_marked"],
                              prep["seed"], limit)

    # Convert summaries to JSON-safe format
    result = {}
    for cond, community_summaries in runs.items():
        result[cond] = {}
        for comm_id, summary_data in community_summaries.items():
            result[cond][str(comm_id)] = summary_data

    return {
        "p_out": prep["p_out"],
        "seed": prep["seed"],
        "backend": llm_cfg["backend"],
        "model": llm_cfg["model"],
        "summaries": result,
    }


def main():
    ap = argparse.ArgumentParser(description="Step 2: LLM summaries (needs GPU)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", required=True, help="same output dir as step1")
    ap.add_argument("--backend", help="override llm.backend")
    ap.add_argument("--model", help="override llm.model (e.g. qwen3:8b)")
    ap.add_argument("--n-graphs", type=int, help="override experiment.n_graphs")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    if a.backend:
        cfg["llm"]["backend"] = a.backend
    if a.model:
        cfg["llm"]["model"] = a.model
    ex = cfg["experiment"]
    if a.n_graphs:
        ex["n_graphs"] = a.n_graphs

    prep_dir = Path(a.out) / "prepared"
    sum_dir = Path(a.out) / "summaries"
    sum_dir.mkdir(parents=True, exist_ok=True)

    if not prep_dir.exists():
        print(f"ERROR: No prepared data found at {prep_dir}/")
        print("Run step1_prepare.py first.")
        return

    files = sorted(prep_dir.glob("pout*_seed*.json"))
    print(f"Found {len(files)} prepared graphs")
    print(f"Backend: {cfg['llm']['backend']}, Model: {cfg['llm']['model']}")
    print(f"Each graph needs ~18 LLM calls (6 communities x 3 conditions)")
    print()

    done, skipped = 0, 0
    for i, prep_file in enumerate(files):
        out_file = sum_dir / prep_file.name
        if out_file.exists():
            print(f"  [{i+1}/{len(files)}] {prep_file.name}: already done, skipping")
            skipped += 1
            continue

        t0 = time.time()
        print(f"  [{i+1}/{len(files)}] {prep_file.name}: summarizing...", end=" ", flush=True)
        result = summarize_one(prep_file, cfg["llm"], ex)
        elapsed = time.time() - t0
        out_file.write_text(json.dumps(result))
        done += 1
        print(f"done ({elapsed:.1f}s)")

    print(f"\nDone! Summarized {done} graphs, skipped {skipped} already-done.")
    print(f"Summaries saved in {sum_dir}/")


if __name__ == "__main__":
    main()
