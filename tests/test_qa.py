from collections import defaultdict
import pytest
import yaml
from communities import build_inputs, cut_triples, detect_hierarchy
from kg_gen import build_graph
from llm import LLMResult, MockLLM
from qa import (INSTRUCTION, LLMAnswerer, OracleReaderAnswerer, accuracy, compare_accuracy,
                evaluate_questions, make_questions, score_answer, visible_summary)
from summarize import summarize_level0

CFG = yaml.safe_load(open("config.yaml"))["graph"]
DATA = build_graph(CFG)
TRIPLES = [tuple(t) for t in DATA["triples"]]
META = DATA["nodes"]
L0, _ = detect_hierarchy(TRIPLES, 1)
INPUTS = build_inputs(TRIPLES, L0, 1)
QS = make_questions(TRIPLES, L0, 1, n=40, node_meta=META)


def test_questions_are_valid_chains_with_unique_hops():
    out = defaultdict(lambda: defaultdict(list))
    for h, r, t in TRIPLES:
        out[h][r].append(t)
    assert len(QS) == 40
    assert {len(q["hops"]) for q in QS} == {2, 3}
    for q in QS:
        assert q["hops"][0][0] == q["start"] and q["gold"] == q["hops"][-1][2]
        for (h, r, t), nxt in zip(q["hops"], q["hops"][1:] + [None]):
            assert (h, r, t) in set(TRIPLES)
            assert len(out[h][r]) == 1                      # hop is unambiguous
            assert nxt is None or nxt[0] == t               # chained
        assert INSTRUCTION in q["text"]


def test_every_question_crosses_a_community_boundary():
    cut = set(cut_triples(TRIPLES, L0))
    assert all(any(h in cut for h in q["hops"]) for q in QS)


def test_same_seed_same_questions():
    assert QS == make_questions(TRIPLES, L0, 1, n=40, node_meta=META)


def test_full_summaries_answer_everything_empty_answers_nothing():
    full = summarize_level0(None, INPUTS, META, use_template=True)
    empty = {c: {"summary": ""} for c in INPUTS}
    a = OracleReaderAnswerer()
    assert accuracy(evaluate_questions(QS, L0, full, INPUTS, a))["acc"] == 1.0
    assert accuracy(evaluate_questions(QS, L0, empty, INPUTS, a))["acc"] == 0.0


def test_lossy_summaries_lower_multi_hop_accuracy_below_single_fact_retention():
    summ = summarize_level0(MockLLM(0.5, 1), INPUTS, META)
    acc = accuracy(evaluate_questions(QS, L0, summ, INPUTS, OracleReaderAnswerer()))["acc"]
    assert acc < 0.45                                       # 2-3 hops compound the loss


def test_questions_blocked_by_the_input_budget_are_flagged_and_excluded():
    small = build_inputs(TRIPLES, L0, 1, max_triples=40)
    summ = summarize_level0(None, small, META, use_template=True)
    acc = accuracy(evaluate_questions(QS, L0, summ, small, OracleReaderAnswerer()))
    assert acc["n_blocked"] > 0 and acc["n"] + acc["n_blocked"] == len(QS)
    assert acc["acc"] == 1.0                                # answerable ones are all answered
    assert accuracy(evaluate_questions(
        QS, L0, summarize_level0(None, INPUTS, META, use_template=True), INPUTS,
        OracleReaderAnswerer()))["n_blocked"] == 0


def test_scoring_is_normalised_and_strict():
    assert score_answer("  Kavortri12. ", "Kavortri12")
    assert score_answer("kavortri12", "Kavortri12")
    assert not score_answer("The answer is Kavortri12", "Kavortri12")
    assert not score_answer("Kavortri123", "Kavortri12")


def test_visible_summary_covers_all_path_communities():
    q = QS[0]
    comms, text = visible_summary(q, L0, summarize_level0(None, INPUTS, META, use_template=True))
    assert comms == sorted({L0[e] for e in [q["start"]] + [t for _h, _r, t in q["hops"]]})
    assert all(f"COMMUNITY {c}:" in text for c in comms)


def test_llm_answerer_builds_constrained_prompt_and_scores():
    class Stub:
        def generate(self, prompt):
            self.prompt = prompt
            return LLMResult(QS[0]["gold"] + ".", 1)

    stub = Stub()
    summ = summarize_level0(None, INPUTS, META, use_template=True)
    rec = evaluate_questions(QS[:1], L0, summ, INPUTS, LLMAnswerer(stub))[0]
    assert rec["correct"]
    assert QS[0]["text"] in stub.prompt and "COMMUNITY" in stub.prompt


def test_compare_accuracy_sign_test():
    a = [{"id": i, "blocked": False, "correct": c} for i, c in enumerate([1, 1, 0, 0, 1])]
    b = [{"id": i, "blocked": False, "correct": c} for i, c in enumerate([1, 0, 0, 1, 1])]
    r = compare_accuracy(a, b)
    assert (r["a_only"], r["b_only"], r["n"]) == (1, 1, 5) and r["p"] == 1.0
    b2 = [{"id": i, "blocked": False, "correct": True} for i in range(5)]
    assert compare_accuracy(a, b2)["b_only"] == 2
