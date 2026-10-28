"""Step 3: Run LLM Summarization — Run One Prompt Condition at a Time (GPU Accelerated).

Loads prepared communities from step 2, prompts the LLM for each community,
and saves the output. You can run one condition at a time:
    --condition default   (Prompt 1: standard GraphRAG summary)
    --condition topology  (Prompt 2: topology-marked prompt)
    --condition control   (Prompt 3: random-control marked prompt)
    --condition all       (runs all 3 in sequence)

Usage:
    # Run only Prompt 1 (default) on Kaggle:
    python step3_summarize.py --config config_kaggle.yaml --out runs/exp1 --condition default --model qwen3:8b

    # Run Prompt 2 (topology):
    python step3_summarize.py --config config_kaggle.yaml --out runs/exp1 --condition topology --model qwen3:8b
"""
import argparse
import json
import time
from pathlib import Path

import yaml

from llm import MockLLM, OllamaLLM
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


def summarize_condition(comm_data, condition, llm_cfg, experiment_cfg):
    meta = comm_data["nodes"]
    inputs = rebuild_inputs(comm_data["inputs"])
    seed = comm_data["seed"]
    llm = build_llm(llm_cfg, meta, seed)
    limit = experiment_cfg["context_limit"] or (
        llm_cfg["num_ctx"] if llm_cfg["backend"] == "ollama" else None)

    if llm_cfg["backend"] == "template":
        summaries = summarize_level0(None, inputs, meta, use_template=True)
    else:
        summaries = summarize_level0(llm, inputs, meta, condition=condition,
                                     k=experiment_cfg["k_marked"], seed=seed,
                                     context_limit=limit)

    # Convert summaries to JSON-safe format
    summaries_json = {}
    for c, data in summaries.items():
        summaries_json[str(c)] = {
            "summary": data["summary"],
            "prompt_tokens": data["prompt_tokens"],
            "near_limit": data["near_limit"],
            "n_input": data["n_input"],
            "marked": data["marked"],
        }

    return {
        "p_out": comm_data["p_out"],
        "seed": seed,
        "condition": condition,
        "backend": llm_cfg["backend"],
        "model": llm_cfg["model"],
        "summaries": summaries_json,
    }


def main():
    ap = argparse.ArgumentParser(description="Step 3: Run LLM Summarization per Condition")
    ap.add_argument("--config", default="config_kaggle.yaml", help="config file")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--condition", choices=["default", "topology", "control", "all"],
                    default="default", help="which prompt condition to run")
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

    comm_dir = Path(a.out) / "communities"
    if not comm_dir.exists():
        print(f"ERROR: No communities directory at {comm_dir}/. Run step2_communities.py first.")
        return

    comm_files = sorted(comm_dir.glob("pout*_seed*.json"))
    if not comm_files:
        print(f"ERROR: No prepared community files found in {comm_dir}/")
        return

    conditions = ["default", "topology", "control"] if a.condition == "all" else [a.condition]

    print(f"{'='*70}")
    print(f"Step 3: LLM Summarization")
    print(f"Backend: {cfg['llm']['backend']} | Model: {cfg['llm']['model']}")
    print(f"Conditions to run: {conditions}")
    print(f"Found {len(comm_files)} graph community files")
    print(f"{'='*70}\n")

    for cond in conditions:
        cond_dir = Path(a.out) / "summaries" / cond
        cond_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- Running Condition: [{cond.upper()}] (saving to {cond_dir}/) ---")

        done, skipped = 0, 0
        for i, f in enumerate(comm_files):
            out_f = cond_dir / f.name
            if out_f.exists():
                print(f"  [{i+1}/{len(comm_files)}] {f.name}: already done, skipping")
                skipped += 1
                continue

            t0 = time.time()
            print(f"  [{i+1}/{len(comm_files)}] {f.name} ({cond}): summarizing...", end=" ", flush=True)
            comm_data = json.loads(f.read_text())
            res = summarize_condition(comm_data, cond, cfg["llm"], ex)
            out_f.write_text(json.dumps(res))
            elapsed = time.time() - t0
            done += 1
            print(f"done ({elapsed:.1f}s)")

        print(f"Condition [{cond}] finished: {done} summarized, {skipped} skipped.")

    print(f"\nStep 3 Complete! Summaries saved under {Path(a.out) / 'summaries'}/")


if __name__ == "__main__":
    main()
