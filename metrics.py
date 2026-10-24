"""M3: retention metrics. Tiered check of whether a triple survives in a summary.

L1  entity coverage: head and tail both appear somewhere in the summary.
L2  head, a cue of the right relation and tail in ONE sentence, head before tail.
distorted: head and tail share a sentence, but the relation is a different one or the
           direction is reversed (and the fact is not also stated correctly).
L3  (judge-verified) is added later on the GPU machine; not in this file yet.

Entity matching is exact with word boundaries. Do NOT use fuzzy matching: names such as
Kavortri12 and Kavortri13 differ by one character and would be confused.
"""
import re

from relations import RELATION_CUES


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _find(sentence, phrase):
    """Position of phrase as a whole word/phrase (case-insensitive), or -1."""
    m = re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", sentence, flags=re.IGNORECASE)
    return m.start() if m else -1


def has_entity(text, name):
    return _find(text, name) >= 0


def retained_l1(triple, summary):
    head, _, tail = triple
    return has_entity(summary, head) and has_entity(summary, tail)


def _cue_positions(sentence, relation):
    return [p for cue in RELATION_CUES[relation] if (p := _find(sentence, cue)) >= 0]


def retained_l2(triple, summary):
    head, rel, tail = triple
    for s in split_sentences(summary):
        ph, pt = _find(s, head), _find(s, tail)
        if ph >= 0 and pt >= 0 and ph < pt and _cue_positions(s, rel):
            return True
    return False


def is_distorted(triple, summary):
    head, rel, tail = triple
    if retained_l2(triple, summary):
        return False
    for s in split_sentences(summary):
        ph, pt = _find(s, head), _find(s, tail)
        if ph < 0 or pt < 0:
            continue
        if _cue_positions(s, rel) and ph > pt:            # right relation, reversed
            return True
        if any(_cue_positions(s, other) for other in RELATION_CUES if other != rel):
            return True                                   # a different relation
    return False


def score_triple(triple, summary):
    return {"l1": retained_l1(triple, summary),
            "l2": retained_l2(triple, summary),
            "distorted": is_distorted(triple, summary)}


def node_retention(input_triples, summary, level="l2"):
    """Per-node retention over the triples that ENTERED the summary input.

    Returns {node: (n_retained, n_total)}. A triple counts for both its endpoints.
    """
    fn = {"l1": retained_l1, "l2": retained_l2}[level]
    out = {}
    for t in input_triples:
        ok = fn(t, summary)
        for node in (t[0], t[2]):
            r, n = out.get(node, (0, 0))
            out[node] = (r + ok, n + 1)
    return out


def overall_retention(input_triples, summary, level="l2"):
    fn = {"l1": retained_l1, "l2": retained_l2}[level]
    if not input_triples:
        return 0.0
    return sum(fn(t, summary) for t in input_triples) / len(input_triples)
