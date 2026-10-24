"""Planted-effect mock summarizer, used to prove that analysis.py can detect a real effect.

Keeps each input triple with probability
    p = base + effect * sigmoid(max(btw_z[head], btw_z[tail]))
using a seeded RNG, where btw_z is the z-scored global betweenness. With effect < 0, triples
touching high-betweenness nodes are lost more often. It is keyed ONLY on continuous
betweenness, never on the planted role label or on degree. Kept triples are written back as
the canonical sentences, so they go through the real metrics and analysis code unchanged.
"""
import math
import random
import zlib

from llm import FACT_PREFIX, LLMResult


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def btw_zscores(node_meta):
    vals = [m["betweenness_global"] for m in node_meta.values()]
    mu = sum(vals) / len(vals)
    sd = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5
    return {n: (m["betweenness_global"] - mu) / sd for n, m in node_meta.items()}


class PlantedEffectMock:
    def __init__(self, node_meta, base=0.8, effect=-0.4, seed=0):
        self.z = btw_zscores(node_meta)
        self.base, self.effect, self.seed = base, effect, seed

    def keep_prob(self, head, tail):
        p = self.base + self.effect * sigmoid(max(self.z[head], self.z[tail]))
        return min(1.0, max(0.0, p))

    def generate(self, prompt):
        lines = [l[len(FACT_PREFIX):] for l in prompt.splitlines()
                 if l.startswith(FACT_PREFIX)]
        rng = random.Random(self.seed + zlib.crc32(prompt.encode()))
        kept = []
        for s in lines:                       # canonical sentence: "<head> <cue> <tail>."
            words = s.split()
            head, tail = words[0], words[-1].rstrip(".")
            if rng.random() < self.keep_prob(head, tail):
                kept.append(s)
        return LLMResult(" ".join(kept), prompt_tokens=len(prompt.split()))


MARKER_PREFIX = "These entities connect this community to others"


class MarkerResponsiveMock(PlantedEffectMock):
    """Planted-effect mock that also reacts to the marker sentence of the topology/control prompts.

    A triple touching a marked node gets  boost_flat + boost_scaled * sigmoid(z of that node)
    added to its keep probability. boost_flat is a generic 'you asked me to mention it' effect
    that helps any marked node equally; boost_scaled helps high-betweenness nodes more. Used to
    check that the intervention analysis separates a generic prompt effect (placebo) from a
    topology-specific one.
    """

    def __init__(self, node_meta, base=0.8, effect=-0.4, boost_flat=0.0, boost_scaled=0.0,
                 seed=0):
        super().__init__(node_meta, base, effect, seed)
        self.boost_flat, self.boost_scaled = boost_flat, boost_scaled

    @staticmethod
    def marked_names(prompt):
        for line in prompt.splitlines():
            if line.startswith(MARKER_PREFIX):
                return {n.strip() for n in line.split("summary:", 1)[1].split(",")}
        return set()

    def generate(self, prompt):
        marked = self.marked_names(prompt)
        lines = [l[len(FACT_PREFIX):] for l in prompt.splitlines()
                 if l.startswith(FACT_PREFIX)]
        rng = random.Random(self.seed + zlib.crc32(prompt.encode()))
        kept = []
        for s in lines:
            words = s.split()
            head, tail = words[0], words[-1].rstrip(".")
            p = self.keep_prob(head, tail)
            touched = [n for n in (head, tail) if n in marked]
            if touched:
                p += self.boost_flat + self.boost_scaled * max(sigmoid(self.z[n])
                                                               for n in touched)
            if rng.random() < min(1.0, max(0.0, p)):
                kept.append(s)
        return LLMResult(" ".join(kept), prompt_tokens=len(prompt.split()))
