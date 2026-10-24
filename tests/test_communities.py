from collections import Counter, defaultdict
import yaml
from kg_gen import build_graph
from communities import detect_hierarchy, cut_triples, build_inputs

CFG = yaml.safe_load(open("config.yaml"))["graph"]
DATA = build_graph(CFG)
TRIPLES = [tuple(t) for t in DATA["triples"]]
NODES = {e for h, _, t in TRIPLES for e in (h, t)}


def test_every_node_assigned_and_nested():
    l0, l1 = detect_hierarchy(TRIPLES, seed=1)
    assert set(l0) == NODES and set(l1) == NODES
    parent = defaultdict(set)
    for n in NODES:
        parent[l0[n]].add(l1[n])
    assert all(len(v) == 1 for v in parent.values())      # level 0 nests in level 1


def test_recovers_planted_partition_roughly():
    l0, _ = detect_hierarchy(TRIPLES, seed=1)
    planted = {n: m["planted_community"] for n, m in DATA["nodes"].items()}
    groups = defaultdict(list)
    for n, c in l0.items():
        groups[c].append(planted[n])
    pure = sum(Counter(v).most_common(1)[0][1] for v in groups.values())
    assert pure / len(NODES) > 0.9


def test_cut_plus_within_equals_total():
    l0, _ = detect_hierarchy(TRIPLES, seed=1)
    inputs = build_inputs(TRIPLES, l0, seed=1)
    within_total = sum(len(v["within"]) for v in inputs.values())
    assert within_total + len(cut_triples(TRIPLES, l0)) == len(TRIPLES)


def test_cut_triple_enters_both_adjacent_communities():
    l0, _ = detect_hierarchy(TRIPLES, seed=1)
    inputs = build_inputs(TRIPLES, l0, seed=1)
    for h, r, t in cut_triples(TRIPLES, l0)[:20]:
        for node in (h, t):
            ins = [x for x, s in inputs[l0[node]]["input"] if s == "cut"]
            assert (h, r, t) in ins


def test_input_is_subset_and_strata_labelled():
    l0, _ = detect_hierarchy(TRIPLES, seed=1)
    inputs = build_inputs(TRIPLES, l0, seed=1)
    for c, v in inputs.items():
        allowed = set(v["within"]) | set(v["cut"])
        assert {t for t, _ in v["input"]} <= allowed
        assert {s for _, s in v["input"]} <= {"within", "cut"}


def test_budget_drop_is_random_and_complete():
    l0, _ = detect_hierarchy(TRIPLES, seed=1)
    a = build_inputs(TRIPLES, l0, seed=1, max_triples=30)
    b = build_inputs(TRIPLES, l0, seed=2, max_triples=30)
    c0 = next(iter(a))
    assert len(a[c0]["input"]) == 30
    assert len(a[c0]["input"]) + len(a[c0]["dropped"]) == len(a[c0]["within"]) + len(a[c0]["cut"])
    assert [t for t, _ in a[c0]["input"]] != [t for t, _ in b[c0]["input"]]   # seed changes it


def test_deterministic_for_same_seed():
    l0a, _ = detect_hierarchy(TRIPLES, seed=3)
    l0b, _ = detect_hierarchy(TRIPLES, seed=3)
    assert l0a == l0b
