"""M8a: default vs topology-aware vs random-control prompts (R8).

Hypothesis pair: marking top-betweenness nodes (topology) raises retention of those nodes more
than marking random non-bridge nodes (control) raises retention of THEIR nodes.

For each condition X in {topology, control}, take the triple occurrences (in the community
inputs) that touch a node marked under X. Then
    lift_X = retention under X  -  retention of the SAME occurrences under the default prompt
and  did = lift_topology - lift_control   (difference in differences).
Comparing each set with itself under the default prompt removes the baseline gap between
bridges and ordinary nodes; the control separates a topology-specific effect from a generic
'the prompt told me to mention these names' effect. Caveat: lift is capped by baseline
retention (a node already at 100% cannot improve), so baselines are reported too.
"""
import numpy as np
from scipy import stats

from analysis import holm
from metrics import retained_l1, retained_l2
from summarize import summarize_level0

CONDITIONS = ("default", "topology", "control")


def run_conditions(llm, inputs, node_meta, k=3, seed=0, context_limit=None):
    """Same inputs, same model; only the prompt condition changes."""
    return {cond: summarize_level0(llm, inputs, node_meta, cond, k=k, seed=seed,
                                   context_limit=context_limit)
            for cond in CONDITIONS}


def _fraction(occurrences, summaries, fn):
    if not occurrences:
        return float("nan")
    return float(np.mean([fn(t, summaries[c]["summary"]) for c, t in occurrences]))


def intervention_stats(inputs, runs, level="l2"):
    fn = {"l1": retained_l1, "l2": retained_l2}[level]
    out = {}
    for cond in ("topology", "control"):
        marked = {c: set(r["marked"]) for c, r in runs[cond].items()}
        occ = [(c, t) for c, v in inputs.items() for t, _s in v["input"]
               if marked[c] & {t[0], t[2]}]
        base = _fraction(occ, runs["default"], fn)
        treated = _fraction(occ, runs[cond], fn)
        out[cond] = {"base": base, "treated": treated, "lift": treated - base, "n": len(occ)}
    out["did"] = out["topology"]["lift"] - out["control"]["lift"]
    return out


def analyse_interventions(per_graph):
    """per_graph: list of intervention_stats dicts. Aggregates with t-tests and Holm."""
    series = {
        "lift_topology": [g["topology"]["lift"] for g in per_graph],
        "lift_control": [g["control"]["lift"] for g in per_graph],
        "did": [g["did"] for g in per_graph],
    }
    res, raw = {}, {}
    for name, vals in series.items():
        v = np.array(vals, float)
        v = v[~np.isnan(v)]
        p = float(stats.ttest_1samp(v, 0.0).pvalue) if len(v) > 1 else float("nan")
        res[name] = {"mean": float(v.mean()), "p": p, "n_graphs": int(len(v))}
        raw[name] = p
    for name, padj in holm({k: p for k, p in raw.items() if not np.isnan(p)}).items():
        res[name]["p_holm"] = padj
    for cond in ("topology", "control"):
        res[f"base_{cond}"] = float(np.nanmean([g[cond]["base"] for g in per_graph]))
    return res
