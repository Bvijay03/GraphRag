import re
import yaml
from kg_gen import build_graph
from communities import detect_hierarchy, build_inputs
from llm import MockLLM
from metrics import overall_retention
from summarize import (build_prompt, select_marked, template_summary, summarize_level0,
                       summarize_level1, triples_entering_level1, MARKER)

CFG = yaml.safe_load(open("config.yaml"))["graph"]
DATA = build_graph(CFG)
TRIPLES = [tuple(t) for t in DATA["triples"]]
META = DATA["nodes"]
L0, L1 = detect_hierarchy(TRIPLES, seed=1)
INPUTS = build_inputs(TRIPLES, L0, seed=1)
C0 = sorted(INPUTS)[0]
ITEMS = INPUTS[C0]["input"]


def test_default_prompt_has_all_facts_and_no_marker():
    p = build_prompt(ITEMS)
    assert p.count("FACT: ") == len(ITEMS)
    assert "connect this community" not in p


def test_topology_and_control_differ_only_in_names():
    top = select_marked(ITEMS, META, "topology", 3, seed=5)
    ctl = select_marked(ITEMS, META, "control", 3, seed=5)
    assert len(top) == len(ctl) == 3 and set(top) != set(ctl)
    strip = lambda p: re.sub(r"summary: .*", "summary: X", p)
    assert strip(build_prompt(ITEMS, top)) == strip(build_prompt(ITEMS, ctl))


def test_control_has_no_planted_bridges_and_no_topology_nodes():
    for seed in range(20):
        top = select_marked(ITEMS, META, "topology", 3, seed)
        ctl = select_marked(ITEMS, META, "control", 3, seed)
        assert all(META[n]["role"] != "bridge" for n in ctl)
        assert not set(ctl) & set(top)


def test_topology_picks_highest_betweenness():
    top = select_marked(ITEMS, META, "topology", 3, seed=0)
    nodes = {e for (h, _, t), _s in ITEMS for e in (h, t)}
    best = sorted(nodes, key=lambda n: -META[n]["betweenness_global"])[:3]
    assert top == best


def test_template_summary_retains_everything_that_entered():
    res = summarize_level0(None, INPUTS, META, use_template=True)
    for c, v in INPUTS.items():
        triples = [t for t, _s in v["input"]]
        assert overall_retention(triples, res[c]["summary"], "l2") == 1.0


def test_mock_run_retention_about_half_and_tokens_logged():
    res = summarize_level0(MockLLM(0.5, 1), INPUTS, META, "default", context_limit=100000)
    triples = [t for v in INPUTS.values() for t, _s in v["input"]]
    kept = sum(overall_retention([t for t, _s in v["input"]], res[c]["summary"], "l2")
               * len(v["input"]) for c, v in INPUTS.items()) / len(triples)
    assert 0.4 < kept < 0.6
    assert all(r["prompt_tokens"] > 0 and not r["near_limit"] for r in res.values())


def test_near_limit_flag():
    res = summarize_level0(MockLLM(0.5, 1), INPUTS, META, "default", context_limit=50)
    assert all(r["near_limit"] for r in res.values())


def test_same_seed_same_outputs():
    a = summarize_level0(MockLLM(0.5, 1), INPUTS, META, "topology", seed=4)
    b = summarize_level0(MockLLM(0.5, 1), INPUTS, META, "topology", seed=4)
    assert a == b


def test_level1_runs_and_entered_facts_defined():
    res = summarize_level0(MockLLM(0.5, 1), INPUTS, META, "default")
    kids = {c: r["summary"] for c, r in res.items()}
    out = summarize_level1(MockLLM(0.5, 1), kids)
    assert out["prompt_tokens"] > 0
    entered = triples_entering_level1({c: v["input"] for c, v in INPUTS.items()}, kids)
    assert 0 < len(entered) < sum(len(v["input"]) for v in INPUTS.values())
