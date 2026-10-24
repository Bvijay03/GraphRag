"""M5: hierarchical communities, cut triples and summary-input tracking.

Level 0: Leiden on the entity graph. Level 1: Leiden on the graph of level-0 communities
(nodes = communities, edge weight = number of triples between them). Level-0 communities
therefore nest inside level-1 communities by construction.

A triple is WITHIN a community if both endpoints are in it, and CUT if its endpoints are in
two different communities. Cut triples go into the input of BOTH adjacent communities as a
separate stratum, so a bridge's cross-community facts are measured, not discarded.
"""
import random
from collections import defaultdict

import igraph as ig


def _entity_graph(triples):
    names = sorted({e for h, _, t in triples for e in (h, t)})
    idx = {n: i for i, n in enumerate(names)}
    edges = sorted({tuple(sorted((idx[h], idx[t]))) for h, _, t in triples})
    g = ig.Graph(n=len(names), edges=edges)
    return g, names


def detect_hierarchy(triples, seed, resolution=1.0):
    """Return two dicts (level 0, level 1): entity name -> community id."""
    ig.set_random_number_generator(random.Random(seed))
    g, names = _entity_graph(triples)
    part0 = g.community_leiden(objective_function="modularity", resolution=resolution,
                               n_iterations=-1)
    mem0 = part0.membership
    k0 = max(mem0) + 1

    weight = defaultdict(int)
    for e in g.es:
        a, b = mem0[e.source], mem0[e.target]
        if a != b:
            weight[tuple(sorted((a, b)))] += 1
    agg = ig.Graph(n=k0, edges=list(weight), edge_attrs={"weight": list(weight.values())})
    part1 = agg.community_leiden(objective_function="modularity", weights="weight",
                                 resolution=resolution, n_iterations=-1)
    mem1_of_comm = part1.membership

    level0 = {n: mem0[i] for i, n in enumerate(names)}
    level1 = {n: mem1_of_comm[mem0[i]] for i, n in enumerate(names)}
    return level0, level1


def cut_triples(triples, membership):
    return [t for t in triples if membership[t[0]] != membership[t[2]]]


def build_inputs(triples, membership, seed, max_triples=None):
    """Per community: which triples enter its summary input.

    Returns {community: {"within": [...], "cut": [...], "input": [(triple, stratum)...],
                         "dropped": [...]}}
    Order is shuffled. If max_triples is set, the drop is RANDOM (not degree-prioritised).
    max_triples is a stand-in for the token budget; real token counts are logged in M6.
    """
    rng = random.Random(seed)
    within, cut = defaultdict(list), defaultdict(list)
    for t in triples:
        a, b = membership[t[0]], membership[t[2]]
        if a == b:
            within[a].append(t)
        else:
            cut[a].append(t)
            cut[b].append(t)
    out = {}
    for c in sorted(set(within) | set(cut)):
        items = [(t, "within") for t in within[c]] + [(t, "cut") for t in cut[c]]
        rng.shuffle(items)
        kept, dropped = items, []
        if max_triples is not None and len(items) > max_triples:
            kept, dropped = items[:max_triples], items[max_triples:]
        out[c] = {"within": within[c], "cut": cut[c], "input": kept,
                  "dropped": [t for t, _ in dropped]}
    return out
