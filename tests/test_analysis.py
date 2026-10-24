import numpy as np
import pytest
import yaml
from analysis import (analyse, generate_tables, graph_stats, holm, node_table,
                      partial_spearman)
from communities import build_inputs, detect_hierarchy
from kg_gen import build_graph
from planted_mock import PlantedEffectMock
from summarize import summarize_level0

CFG = yaml.safe_load(open("config.yaml"))["graph"]


def test_holm_known_example():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})


def test_partial_spearman_removes_common_cause():
    rng = np.random.default_rng(0)
    z = rng.normal(size=500)
    x, y = z + 0.3 * rng.normal(size=500), z + 0.3 * rng.normal(size=500)
    raw = np.corrcoef(x, y)[0, 1]
    assert raw > 0.8 and abs(partial_spearman(x, y, [z])) < 0.15


def _table(use_template=False, empty=False):
    data = build_graph(CFG)
    triples = [tuple(t) for t in data["triples"]]
    meta = data["nodes"]
    l0, _ = detect_hierarchy(triples, 1)
    inputs = build_inputs(triples, l0, 1)
    if empty:
        summaries = {c: {"summary": ""} for c in inputs}
    else:
        summaries = summarize_level0(None, inputs, meta, use_template=use_template)
    return node_table(inputs, summaries, meta, l0)


def test_node_table_full_and_empty_summaries():
    full = _table(use_template=True)
    assert all(r["r_l2"] == r["total"] for r in full.values())
    empty = _table(empty=True)
    assert all(r["r_l2"] == 0 and r["r_l1"] == 0 for r in empty.values())


def test_group_sizes_and_matched_degree_in_table():
    t = _table(use_template=True)
    roles = [r["role"] for r in t.values()]
    assert roles.count("bridge") == CFG["n_bridges"] and roles.count("hub") == CFG["n_hubs"]
    assert {r["degree"] for r in t.values() if r["role"] in ("bridge", "hub")} == {12}


@pytest.fixture(scope="module")
def planted():
    f = lambda meta, seed: PlantedEffectMock(meta, 0.8, -0.4, seed)
    return analyse(generate_tables(CFG, range(1, 13), f))


@pytest.fixture(scope="module")
def null():
    f = lambda meta, seed: PlantedEffectMock(meta, 0.8, 0.0, seed)
    return analyse(generate_tables(CFG, range(100, 140), f))


def test_planted_effect_is_recovered_in_direction(planted):
    assert planted["l2:diff"]["mean"] < -0.08
    assert planted["l2:diff"]["p_holm"] < 0.05
    assert planted["l2:rho_g"]["mean"] < 0 and planted["l2:rho_g"]["p_holm"] < 0.05


def test_effect_survives_controls_for_degree_and_size(planted):
    assert planted["l2:prho_g"]["mean"] < 0 and planted["l2:prho_g"]["p_holm"] < 0.05


def test_null_dataset_has_no_l2_effect(null):
    assert abs(null["l2:rho_g"]["mean"]) < 0.1
    assert abs(null["l2:diff"]["mean"]) < 0.06


def test_null_false_positive_rate_per_graph(null):
    p = [g["p_diff"] for g in null["l2:graph_stats"]]
    assert sum(x < 0.05 for x in p) / len(p) < 0.15


def test_l1_is_biased_against_bridges_even_under_null(null):
    """Documents why L1 is a diagnostic only: it shows a bridge penalty with no planted effect."""
    assert null["l1:diff"]["mean"] < -0.03
