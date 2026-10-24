import copy
import statistics as st
import pytest
import yaml
from kg_gen import build_graph

CFG = yaml.safe_load(open("config.yaml"))["graph"]


@pytest.fixture(scope="module")
def data():
    return build_graph(CFG)


def test_same_seed_same_graph():
    assert build_graph(CFG)["triples"] == build_graph(CFG)["triples"]


def test_different_seed_different_graph():
    other = copy.deepcopy(CFG)
    other["seed"] += 1
    assert build_graph(CFG)["triples"] != build_graph(other)["triples"]


def test_planted_nodes_have_exact_degree(data):
    for n in data["nodes"].values():
        if n["role"] in ("bridge", "hub"):
            assert n["degree"] == CFG["planted_degree"]


def test_degree_matched_betweenness_separated(data):
    nodes = data["nodes"].values()
    b = [n["betweenness_global"] for n in nodes if n["role"] == "bridge"]
    h = [n["betweenness_global"] for n in nodes if n["role"] == "hub"]
    assert st.mean(b) > 2 * st.mean(h)


def test_names_unique(data):
    names = list(data["nodes"])
    assert len(names) == len(set(names))


def test_role_counts(data):
    roles = [n["role"] for n in data["nodes"].values()]
    assert roles.count("bridge") == CFG["n_bridges"]
    assert roles.count("hub") == CFG["n_hubs"]


def test_no_duplicate_triples(data):
    pairs = {frozenset((h, t)) for h, _, t in data["triples"]}
    assert len(pairs) == len(data["triples"])
