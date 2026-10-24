import re
import yaml
from kg_gen import build_graph
from relations import render

DATA = build_graph(yaml.safe_load(open("config.yaml"))["graph"])
TRIPLES = [tuple(t) for t in DATA["triples"]]


def test_one_sentence_per_triple_with_head_and_tail():
    for t in TRIPLES:
        s = render(t)
        assert s.startswith(t[0]) and s.endswith(t[2] + ".")


def test_mention_count_equals_degree_so_matched_degree_controls_mentions():
    text = " ".join(render(t) for t in TRIPLES)
    for name, m in DATA["nodes"].items():
        assert len(re.findall(rf"(?<!\w){re.escape(name)}(?!\w)", text)) == m["degree"]
