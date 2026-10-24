import yaml
from kg_gen import build_graph
from llm import MockLLM, FACT_PREFIX
from metrics import (retained_l1, retained_l2, is_distorted, node_retention,
                     overall_retention)
from relations import render

T = ("Kavortri12", "works_at", "Zunmir3")


def test_empty_summary_scores_zero():
    assert not retained_l1(T, "")
    assert not retained_l2(T, "")
    assert not is_distorted(T, "")
    assert overall_retention([T], "") == 0.0


def test_full_oracle_summary_scores_one():
    cfg = yaml.safe_load(open("config.yaml"))["graph"]
    triples = [tuple(t) for t in build_graph(cfg)["triples"]]
    summary = " ".join(render(t) for t in triples)
    assert overall_retention(triples, summary, "l1") == 1.0
    assert overall_retention(triples, summary, "l2") == 1.0
    assert not any(is_distorted(t, summary) for t in triples)


def test_entities_without_relation_is_l1_only():
    s = "Kavortri12 and Zunmir3 were mentioned together."
    assert retained_l1(T, s) and not retained_l2(T, s) and not is_distorted(T, s)


def test_paraphrase_cue_counts():
    assert retained_l2(T, "Kavortri12 is employed by Zunmir3.")


def test_wrong_relation_is_distorted():
    s = "Kavortri12 acquired Zunmir3."
    assert is_distorted(T, s) and not retained_l2(T, s)


def test_reversed_direction_is_distorted():
    s = "Zunmir3 works at Kavortri12."
    assert is_distorted(T, s) and not retained_l2(T, s)


def test_entities_in_different_sentences_not_l2():
    s = "Kavortri12 works at a firm. Zunmir3 is elsewhere."
    assert retained_l1(T, s) and not retained_l2(T, s)


def test_word_boundary_no_prefix_match():
    assert not retained_l1(("Kavortri12", "works_at", "Zunmir3"),
                           "Kavortri123 works at Zunmir3.")


def test_node_retention_counts_both_endpoints():
    t2 = ("Zunmir3", "located_in", "Belvox9")
    s = render(T)                      # only T is in the summary
    r = node_retention([T, t2], s)
    assert r["Kavortri12"] == (1, 1)
    assert r["Zunmir3"] == (1, 2)
    assert r["Belvox9"] == (0, 1)


def test_mock_50_percent_acceptance():
    cfg = yaml.safe_load(open("config.yaml"))["graph"]
    triples = [tuple(t) for t in build_graph(cfg)["triples"]]
    prompt = "\n".join(FACT_PREFIX + render(t) for t in triples)
    summary = MockLLM(0.5, seed=1).generate(prompt).text
    l1 = overall_retention(triples, summary, "l1")
    l2 = overall_retention(triples, summary, "l2")
    assert 0.45 < l2 < 0.55
    assert l1 >= l2          # L1 is inflated: a node is covered through its other triples
