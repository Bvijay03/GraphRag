import copy
import yaml
from kg_gen import build_graph
from llm import FACT_PREFIX
from planted_mock import PlantedEffectMock
from relations import render

DATA = build_graph(yaml.safe_load(open("config.yaml"))["graph"])
META = DATA["nodes"]
TRIPLES = [tuple(t) for t in DATA["triples"]]
PROMPT = "\n".join(FACT_PREFIX + render(t) for t in TRIPLES)


def test_keep_prob_decreases_with_betweenness_and_stays_in_range():
    m = PlantedEffectMock(META, base=0.8, effect=-0.4)
    ranked = sorted(META, key=lambda n: META[n]["betweenness_global"])
    low, high = ranked[0], ranked[-1]
    assert m.keep_prob(low, low) > m.keep_prob(high, high)
    for h, _, t in TRIPLES:
        assert 0.4 <= m.keep_prob(h, t) <= 0.8


def test_ignores_role_label_and_degree():
    scrambled = copy.deepcopy(META)
    for i, n in enumerate(scrambled):
        scrambled[n]["role"] = ["bridge", "hub", "regular"][i % 3]
        scrambled[n]["degree"] = 999 - i
    a = PlantedEffectMock(META, seed=3).generate(PROMPT).text
    b = PlantedEffectMock(scrambled, seed=3).generate(PROMPT).text
    assert a == b


def test_kept_sentences_are_canonical_and_deterministic():
    out = PlantedEffectMock(META, seed=3).generate(PROMPT).text
    canonical = {render(t) for t in TRIPLES}
    kept = [s for s in out.split(". ") if s]
    assert all((s if s.endswith(".") else s + ".") in canonical for s in kept)
    assert out == PlantedEffectMock(META, seed=3).generate(PROMPT).text


def test_zero_effect_keeps_about_base():
    out = PlantedEffectMock(META, base=0.8, effect=0.0, seed=1).generate(PROMPT).text
    n_kept = sum(1 for t in TRIPLES if render(t) in out)
    assert 0.72 < n_kept / len(TRIPLES) < 0.88
