"""M7: per-node retention tables, bridge-vs-hub contrast and centrality correlations.

Unit of analysis: the node. A node's retention is the fraction of the triple occurrences in
its community inputs (it is head or tail) that survive in that community's summary. A cut
triple occurs in two inputs, so it counts once per input.

Primary analysis: bridge vs hub group contrast at matched degree (planted roles; synthetic
data only). Secondary: Spearman and partial Spearman correlation of retention with
betweenness (global and within-community), controlling for degree, community input size and
occurrence count. Computed per graph, then aggregated across graphs (one-sample t-test on
the per-graph values) with Holm correction across the whole family of tests.
"""
import numpy as np
from scipy import stats

from communities import build_inputs, detect_hierarchy
from kg_gen import build_graph
from metrics import retained_l1, retained_l2


def node_table(inputs, summaries, node_meta, level0):
    t = {n: {"role": m["role"], "degree": m["degree"], "btw_g": m["betweenness_global"],
             "btw_l": m["betweenness_local"], "total": 0, "r_l1": 0, "r_l2": 0,
             "within_t": 0, "within_r": 0, "cut_t": 0, "cut_r": 0,
             "input_size": len(inputs[level0[n]]["input"]) if level0[n] in inputs else 0}
         for n, m in node_meta.items()}
    for c, v in inputs.items():
        s = summaries[c]["summary"]
        for triple, stratum in v["input"]:
            r1, r2 = retained_l1(triple, s), retained_l2(triple, s)
            for node in (triple[0], triple[2]):
                row = t[node]
                row["total"] += 1
                row["r_l1"] += r1
                row["r_l2"] += r2
                row[f"{stratum}_t"] += 1
                row[f"{stratum}_r"] += r2
    return t


def partial_spearman(x, y, controls):
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    C = np.column_stack([stats.rankdata(c) for c in controls] + [np.ones(len(x))])
    ex = rx - C @ np.linalg.lstsq(C, rx, rcond=None)[0]
    ey = ry - C @ np.linalg.lstsq(C, ry, rcond=None)[0]
    if ex.std() == 0 or ey.std() == 0:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def graph_stats(table, level="l2"):
    rows = [r for r in table.values() if r["total"] > 0]
    ret = np.array([r[f"r_{level}"] / r["total"] for r in rows])
    role = np.array([r["role"] for r in rows])
    controls = [np.array([r["degree"] for r in rows], float),
                np.array([r["input_size"] for r in rows], float),
                np.array([r["total"] for r in rows], float)]
    out = {}
    b, h = ret[role == "bridge"], ret[role == "hub"]
    out["bridge_mean"], out["hub_mean"] = float(b.mean()), float(h.mean())
    out["diff"] = out["bridge_mean"] - out["hub_mean"]
    out["p_diff"] = float(stats.mannwhitneyu(b, h).pvalue)
    for key, name in (("btw_g", "g"), ("btw_l", "l")):
        x = np.array([r[key] for r in rows])
        if ret.std() == 0 or x.std() == 0:
            out[f"rho_{name}"] = out[f"prho_{name}"] = float("nan")
            continue
        out[f"rho_{name}"] = float(stats.spearmanr(x, ret).statistic)
        out[f"prho_{name}"] = partial_spearman(x, ret, controls)
    for stratum in ("within", "cut"):          # H1 mechanism: where is the loss?
        for r_, name in (("bridge", "b"), ("hub", "h")):
            tot = sum(x[f"{stratum}_t"] for x in rows if x["role"] == r_)
            kept = sum(x[f"{stratum}_r"] for x in rows if x["role"] == r_)
            out[f"{stratum}_{name}"] = kept / tot if tot else float("nan")
    return out


def holm(pvals):
    """Holm step-down adjustment. pvals: dict name -> p. Returns dict name -> adjusted p."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, adj, running = len(items), {}, 0.0
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[name] = running
    return adj


def analyse(tables, levels=("l1", "l2")):
    """tables: list of node tables, one per graph. Returns aggregated results per test."""
    results, raw_p = {}, {}
    for lv in levels:
        per_graph = [graph_stats(t, lv) for t in tables]
        for metric in ("diff", "rho_g", "prho_g", "rho_l", "prho_l"):
            vals = np.array([g[metric] for g in per_graph], float)
            vals = vals[~np.isnan(vals)]
            p = float(stats.ttest_1samp(vals, 0.0).pvalue) if len(vals) > 1 else float("nan")
            results[f"{lv}:{metric}"] = {"mean": float(vals.mean()) if len(vals) else float("nan"),
                                         "n_graphs": int(len(vals)), "p": p}
            if not np.isnan(p):
                raw_p[f"{lv}:{metric}"] = p
        results[f"{lv}:graph_stats"] = per_graph
    # Stratum retention aggregated (L2 only — stratum tracking uses L2 by construction).
    # Adds keys like "l2:within_b", "l2:cut_h", etc. for the mechanism table.
    if "l2:graph_stats" in results:
        for stratum in ("within", "cut"):
            for role in ("b", "h"):
                key = f"{stratum}_{role}"
                vals = np.array(
                    [g[key] for g in results["l2:graph_stats"] if not np.isnan(g[key])],
                    dtype=float,
                )
                results[f"l2:{key}"] = {
                    "mean": float(vals.mean()) if len(vals) else float("nan"),
                    "n_graphs": int(len(vals)),
                }
    for name, padj in holm(raw_p).items():
        results[name]["p_holm"] = padj
    return results


def generate_tables(cfg, seeds, llm_factory, condition="default"):
    """Run the pipeline for each seed and return one node table per usable graph."""
    tables = []
    for seed in seeds:
        c = dict(cfg)
        c["seed"] = seed
        try:
            data = build_graph(c)
        except ValueError:                      # disconnected graph: skip this seed
            continue
        triples = [tuple(x) for x in data["triples"]]
        meta = data["nodes"]
        l0, _ = detect_hierarchy(triples, seed)
        inputs = build_inputs(triples, l0, seed)
        from summarize import summarize_level0
        summaries = summarize_level0(llm_factory(meta, seed), inputs, meta, condition, seed=seed)
        tables.append(node_table(inputs, summaries, meta, l0))
    return tables
