import pytest
import yaml
from communities import build_inputs, detect_hierarchy
from kg_gen import build_graph
from interventions import analyse_interventions, intervention_stats, run_conditions
from planted_mock import MarkerResponsiveMock
from summarize import build_prompt, select_marked

CFG = yaml.safe_load(open("config.yaml"))["graph"]


def _experiment(boost_flat, boost_scaled, seeds=range(1, 13)):
    per = []
    for s in seeds:
        c = dict(CFG)
        c["seed"] = s
        try:
            d = build_graph(c)
        except ValueError:
            continue
        triples = [tuple(t) for t in d["triples"]]
        meta = d["nodes"]
        l0, _ = detect_hierarchy(triples, s)
        inputs = build_inputs(triples, l0, s)
        llm = MarkerResponsiveMock(meta, 0.8, -0.4, boost_flat, boost_scaled, s)
        per.append(intervention_stats(inputs, run_conditions(llm, inputs, meta, k=3, seed=s)))
    return per


@pytest.fixture(scope="module")
def flat():
    return _experiment(0.15, 0.0)


@pytest.fixture(scope="module")
def scaled():
    return _experiment(0.0, 0.3)


@pytest.fixture(scope="module")
def none():
    return _experiment(0.0, 0.0)


def test_marker_names_are_parsed_from_the_real_prompt():
    d = build_graph(CFG)
    triples = [tuple(t) for t in d["triples"]]
    l0, _ = detect_hierarchy(triples, 1)
    inputs = build_inputs(triples, l0, 1)
    c = sorted(inputs)[0]
    marked = select_marked(inputs[c]["input"], d["nodes"], "topology", 3, 0)
    prompt = build_prompt(inputs[c]["input"], marked)
    assert MarkerResponsiveMock.marked_names(prompt) == set(marked)
    assert MarkerResponsiveMock.marked_names(build_prompt(inputs[c]["input"])) == set()


def test_stats_structure_and_did_definition(flat):
    g = flat[0]
    assert g["topology"]["n"] > 0 and g["control"]["n"] > 0
    assert g["did"] == pytest.approx(g["topology"]["lift"] - g["control"]["lift"])


def test_generic_prompt_effect_is_not_called_topology_specific(flat):
    r = analyse_interventions(flat)
    assert r["lift_topology"]["mean"] > 0.1 and r["lift_topology"]["p_holm"] < 0.05
    assert r["lift_control"]["mean"] > 0.08          # the placebo also works
    assert abs(r["did"]["mean"]) < 0.06 and r["did"]["p_holm"] > 0.05


def test_topology_specific_effect_is_detected(scaled):
    r = analyse_interventions(scaled)
    assert r["did"]["mean"] > 0.1 and r["did"]["p_holm"] < 0.05


def test_no_intervention_effect_means_no_signal(none):
    r = analyse_interventions(none)
    assert all(abs(r[k]["mean"]) < 0.05 for k in ("lift_topology", "lift_control", "did"))
    assert all(r[k]["p_holm"] > 0.05 for k in ("lift_topology", "lift_control", "did"))


def test_baselines_are_reported_and_differ(none):
    r = analyse_interventions(none)
    assert r["base_topology"] < r["base_control"]    # bridges start lower: ceiling caveat
