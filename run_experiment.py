"""M9: experiment driver. Loops over p_out values and graph seeds, runs the pipeline
(graph -> communities -> inputs -> summaries under 3 prompt conditions -> retention tables ->
questions), writes one JSON file per graph (resumable), then aggregates.

Usage:
    python run_experiment.py --config config.yaml --out runs/exp1
    python run_experiment.py --backend ollama --out runs/real1          # on the GPU machine
Backends: mock (random 50%), planted (known effect, for testing), template (no LLM), ollama.
"""
import argparse
import copy
import hashlib
import importlib.metadata as md
import json
import math
import platform
import sys
import time
from pathlib import Path

import yaml

from analysis import analyse, node_table
from communities import build_inputs, cut_triples, detect_hierarchy
from interventions import (CONDITIONS, analyse_interventions, intervention_stats,
                           run_conditions)
from kg_gen import build_graph
from llm import MockLLM, OllamaLLM
from planted_mock import MarkerResponsiveMock
from qa import (LLMAnswerer, OracleReaderAnswerer, accuracy, compare_accuracy,
                evaluate_questions, make_questions)
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


def _clean(x):
    """JSON-safe: NaN -> None."""
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def run_graph(cfg, p_out, seed, oracle_qa=False):
    """Run the whole pipeline for one graph. Returns a record, or None if the graph is
    disconnected (the seed is skipped)."""
    g = dict(cfg["graph"], p_out=p_out, seed=seed)
    try:
        data = build_graph(g)
    except ValueError:
        return None
    ex, lc = cfg["experiment"], cfg["llm"]
    triples = [tuple(t) for t in data["triples"]]
    meta = data["nodes"]
    l0, l1 = detect_hierarchy(triples, seed)
    inputs = build_inputs(triples, l0, seed, ex["max_triples"])
    llm = build_llm(lc, meta, seed)
    limit = ex["context_limit"] or (lc["num_ctx"] if lc["backend"] == "ollama" else None)

    if lc["backend"] == "template":
        runs = {"default": summarize_level0(None, inputs, meta, use_template=True)}
    else:
        runs = run_conditions(llm, inputs, meta, ex["k_marked"], seed, limit)

    questions = make_questions(triples, l0, seed, n=ex["qa_n"], node_meta=meta)
    answerer = OracleReaderAnswerer() if (oracle_qa or lc["backend"] != "ollama") else LLMAnswerer(llm)
    qa_records = {c: evaluate_questions(questions, l0, runs[c], inputs, answerer) for c in runs}

    tokens = [r["prompt_tokens"] for c in runs for r in runs[c].values()]
    return _clean({
        "p_out": p_out, "seed": seed, "backend": lc["backend"], "model": lc["model"],
        "n_nodes": len(meta), "n_triples": len(triples),
        "n_communities_l0": len(set(l0.values())), "n_communities_l1": len(set(l1.values())),
        "n_cut_triples": len(cut_triples(triples, l0)),
        "max_prompt_tokens": max(tokens) if tokens else 0,
        "n_near_limit": sum(r["near_limit"] for c in runs for r in runs[c].values()),
        "n_questions": len(questions),
        "tables": {c: node_table(inputs, runs[c], meta, l0) for c in runs},
        "intervention": intervention_stats(inputs, runs) if len(runs) == 3 else None,
        "qa_records": qa_records,
        "qa": {c: accuracy(qa_records[c]) for c in qa_records},
        "marked": {c: {k: v["marked"] for k, v in runs[c].items()} for c in runs},
        "summaries": {c: {k: v["summary"] for k, v in runs[c].items()} for c in runs},
    })


def _pooled_qa(records, a, b):
    ra = [dict(r, id=(i, r["id"])) for i, rec in enumerate(records) for r in rec["qa_records"][a]]
    rb = [dict(r, id=(i, r["id"])) for i, rec in enumerate(records) for r in rec["qa_records"][b]]
    return compare_accuracy(ra, rb)


def aggregate(records):
    """Aggregate the per-graph records of ONE p_out value."""
    tables = [r["tables"]["default"] for r in records]
    out = {"n_graphs": len(records),
           "mean_cut_fraction": sum(r["n_cut_triples"] / r["n_triples"] for r in records)
                                / len(records),
           "n_near_limit": sum(r["n_near_limit"] for r in records),
           "max_prompt_tokens": max(r["max_prompt_tokens"] for r in records),
           "retention": analyse(tables)}
    if all(r["intervention"] for r in records):
        out["interventions"] = analyse_interventions([r["intervention"] for r in records])
        out["qa"] = {"accuracy": {c: sum(r["qa"][c]["acc"] for r in records) / len(records)
                                  for c in CONDITIONS},
                     "default_vs_topology": _pooled_qa(records, "default", "topology"),
                     "default_vs_control": _pooled_qa(records, "default", "control"),
                     "topology_vs_control": _pooled_qa(records, "topology", "control")}
    else:
        out["qa"] = {"accuracy": {"default": sum(r["qa"]["default"]["acc"] for r in records)
                                  / len(records)}}
    return _clean(out)


def write_summary(path, cfg, by_p, skipped):
    lc = cfg["llm"]
    L = [f"# Experiment summary", "",
         f"Backend: **{lc['backend']}**; model: {lc['model']}; num_ctx: {lc['num_ctx']}; "
         f"temperature {lc['temperature']}; thinking off: {not lc['think']}.", "",
         f"Skipped (disconnected) seeds: {skipped}.", ""]
    for p, a in by_p.items():
        L += [f"## p_out = {p}", "",
              f"- graphs: {a['n_graphs']}; mean cut-triple fraction: {a['mean_cut_fraction']:.3f}",
              f"- prompts near the context limit: {a['n_near_limit']} "
              f"(max prompt tokens {a['max_prompt_tokens']})", "",
              "### Retention (L2 primary; L1 diagnostic only, biased against bridges)", "",
              "| test | mean over graphs | p | p (Holm) |", "| --- | --- | --- | --- |"]
        for lv in ("l2", "l1"):
            for m in ("diff", "rho_g", "prho_g", "rho_l", "prho_l"):
                r = a["retention"][f"{lv}:{m}"]
                f = lambda v: "n/a" if v is None else f"{v:.4f}"
                L.append(f"| {lv} {m} | {f(r['mean'])} | {f(r['p'])} | {f(r.get('p_holm'))} |")
        # Mechanism table: where does the loss happen — cut triples or within-community?
        def _fmt(v):
            return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.3f}"
        L += ["", "### Mechanism: L2 retention by stratum and role (mean over graphs)", "",
              "Partition loss = cut-triple stratum; compression loss = within-community stratum.",
              "A bridge-vs-hub gap in the *cut* row points to the partition as the cause.", "",
              "| stratum | bridges | hubs | diff (bridge - hub) |",
              "| --- | --- | --- | --- |"]
        for stratum in ("within", "cut"):
            b_val = a["retention"].get(f"l2:{stratum}_b", {}).get("mean")
            h_val = a["retention"].get(f"l2:{stratum}_h", {}).get("mean")
            if (b_val is not None and h_val is not None
                    and not math.isnan(b_val) and not math.isnan(h_val)):
                diff_val = b_val - h_val
            else:
                diff_val = float("nan")
            L.append(f"| {stratum} | {_fmt(b_val)} | {_fmt(h_val)} | {_fmt(diff_val)} |")
        if "interventions" in a:
            i = a["interventions"]
            L += ["", "### Interventions (lift = retention gain over the default prompt)", "",
                  "| quantity | mean | p | p (Holm) |", "| --- | --- | --- | --- |"]
            for k in ("lift_topology", "lift_control", "did"):
                L.append(f"| {k} | {f(i[k]['mean'])} | {f(i[k]['p'])} | {f(i[k].get('p_holm'))} |")
            L += [f"", f"Baseline retention of marked nodes: topology {_fmt(i['base_topology'])}, "
                  f"control {_fmt(i['base_control'])} (lift is capped by baseline)."]
        L += ["", "### Multi-hop QA accuracy", ""]
        for c, v in a["qa"]["accuracy"].items():
            L.append(f"- {c}: {_fmt(v)}")
        for k in ("default_vs_topology", "default_vs_control", "topology_vs_control"):
            if k in a["qa"]:
                q = a["qa"][k]
                L.append(f"- {k}: {_fmt(q['acc_a'])} vs {_fmt(q['acc_b'])}, sign-test p = "
                         f"{f(q['p'])} (n = {q['n']})")
        L.append("")
    Path(path).write_text("\n".join(L), encoding="utf-8")


def env_info():
    pk = {}
    for name in ("networkx", "python-igraph", "scipy", "numpy", "pyyaml", "ollama"):
        try:
            pk[name] = md.version(name)
        except md.PackageNotFoundError:
            pk[name] = None
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": pk}


def run(cfg, out_dir, resume=True, log=print, oracle_qa=False):
    out = Path(out_dir)
    (out / "graphs").mkdir(parents=True, exist_ok=True)
    cfg_json = json.dumps(cfg, sort_keys=True)
    (out / "config_used.yaml").write_text(yaml.safe_dump(cfg))
    (out / "env.json").write_text(json.dumps(
        dict(env_info(), config_sha256=hashlib.sha256(cfg_json.encode()).hexdigest(),
             started=time.strftime("%Y-%m-%d %H:%M:%S")), indent=2))
    ex = cfg["experiment"]
    by_p, skipped = {}, []
    for p in ex["p_out_values"]:
        records = []
        for seed in range(ex["seed_start"], ex["seed_start"] + ex["n_graphs"]):
            f = out / "graphs" / f"pout{p}_seed{seed}.json"
            if resume and f.exists():
                records.append(json.loads(f.read_text()))
                continue
            rec = run_graph(cfg, p, seed, oracle_qa=oracle_qa)
            if rec is None:
                skipped.append({"p_out": p, "seed": seed})
                log(f"p_out={p} seed={seed}: disconnected, skipped")
                continue
            f.write_text(json.dumps(rec))
            records.append(rec)
            log(f"p_out={p} seed={seed}: done")
        if records:
            by_p[str(p)] = aggregate(records)
    (out / "results.json").write_text(json.dumps(by_p, indent=2))
    write_summary(out / "summary.md", cfg, by_p, skipped)
    return by_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", help="override llm.backend")
    ap.add_argument("--n-graphs", type=int, help="override experiment.n_graphs")
    ap.add_argument("--oracle-qa", action="store_true", help="use fast deterministic Oracle QA (7x speedup on Kaggle)")
    ap.add_argument("--no-resume", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.backend:
        cfg["llm"]["backend"] = a.backend
    if a.n_graphs:
        cfg["experiment"]["n_graphs"] = a.n_graphs
    run(cfg, a.out, resume=not a.no_resume, oracle_qa=a.oracle_qa)
    print(f"done: see {a.out}/summary.md")


if __name__ == "__main__":
    main()
