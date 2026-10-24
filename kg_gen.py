"""M2: synthetic knowledge graph with planted communities, bridge nodes and hub nodes.

Bridges and hubs have the SAME degree. Bridges spread their edges over several other
communities (high betweenness); hubs keep all edges inside their own community.
"""
import json
import random

import networkx as nx

RELATIONS = ["works_at", "located_in", "founded_by", "funded_by",
             "partners_with", "member_of", "supplies", "acquired"]

SYLLABLES = ["ka", "vor", "tri", "lex", "an", "ora", "pel", "zun", "mir", "dax",
             "sol", "ven", "qui", "bar", "nel", "tov", "rin", "gal", "fen", "yos"]


def make_names(n, rng):
    """Unique, unambiguous entity names. The index guarantees uniqueness."""
    names = []
    for i in range(n):
        base = "".join(rng.choice(SYLLABLES) for _ in range(3)).capitalize()
        names.append(f"{base}{i}")
    return names


def plant_node(G, node, role, comm, comm_nodes, ordinary, cfg, rng):
    """Remove node's random edges, then wire exactly planted_degree edges.

    hub    -> neighbours inside its OWN community
    bridge -> neighbours spread equally over n_cross OTHER communities
    Neighbours are always ordinary (non-planted) nodes, so degrees never interfere.
    """
    G.remove_edges_from(list(G.edges(node)))
    d = cfg["planted_degree"]
    c = comm[node]
    if role == "hub":
        pool = [m for m in comm_nodes[c] if m in ordinary]
        neighbours = rng.sample(pool, d)
    else:
        n_cross = cfg["n_cross"]
        if d % n_cross != 0:
            raise ValueError("planted_degree must be divisible by n_cross")
        per = d // n_cross
        others = [x for x in range(cfg["n_communities"]) if x != c]
        neighbours = []
        for other in rng.sample(others, n_cross):
            pool = [m for m in comm_nodes[other] if m in ordinary]
            neighbours += rng.sample(pool, per)
    G.add_edges_from((node, m) for m in neighbours)


def build_graph(cfg):
    rng = random.Random(cfg["seed"])
    k, size = cfg["n_communities"], cfg["community_size"]
    p = [[cfg["p_in"] if i == j else cfg["p_out"] for j in range(k)] for i in range(k)]
    G = nx.stochastic_block_model([size] * k, p, seed=cfg["seed"])

    comm = {}
    for ci, nodes in enumerate(G.graph["partition"]):
        for n in nodes:
            comm[n] = ci
    comm_nodes = {ci: sorted(nodes) for ci, nodes in enumerate(G.graph["partition"])}

    chosen = rng.sample(sorted(G.nodes()), cfg["n_bridges"] + cfg["n_hubs"])
    bridges, hubs = chosen[:cfg["n_bridges"]], chosen[cfg["n_bridges"]:]
    ordinary = set(G.nodes()) - set(chosen)
    role = {n: "regular" for n in G.nodes()}
    # remove all planted nodes' random edges first so planted nodes never link to each other
    for n in bridges:
        role[n] = "bridge"
    for n in hubs:
        role[n] = "hub"
    for n in bridges:
        plant_node(G, n, "bridge", comm, comm_nodes, ordinary, cfg, rng)
    for n in hubs:
        plant_node(G, n, "hub", comm, comm_nodes, ordinary, cfg, rng)

    if not nx.is_connected(G):
        raise ValueError("graph is not connected: change the seed or raise p_in/p_out")

    names = make_names(G.number_of_nodes(), rng)
    name = {n: names[i] for i, n in enumerate(sorted(G.nodes()))}

    bet_global = nx.betweenness_centrality(G)
    bet_local = {}
    for ci, nodes in comm_nodes.items():
        bet_local.update(nx.betweenness_centrality(G.subgraph(nodes)))

    triples = []
    for u, v in sorted(tuple(sorted(e)) for e in G.edges()):
        h, t = (u, v) if rng.random() < 0.5 else (v, u)
        triples.append([name[h], rng.choice(RELATIONS), name[t]])

    nodes_meta = {
        name[n]: {
            "planted_community": comm[n],
            "role": role[n],
            "degree": G.degree(n),
            "betweenness_global": bet_global[n],
            "betweenness_local": bet_local[n],
        }
        for n in G.nodes()
    }
    return {"triples": triples, "nodes": nodes_meta}


def save_graph(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    import sys
    import yaml
    cfg = yaml.safe_load(open("config.yaml"))["graph"]
    out = sys.argv[1] if len(sys.argv) > 1 else "graph.json"
    data = build_graph(cfg)
    save_graph(data, out)
    nodes = data["nodes"].values()
    for r in ("bridge", "hub", "regular"):
        sel = [n for n in nodes if n["role"] == r]
        mb = sum(n["betweenness_global"] for n in sel) / len(sel)
        md = sum(n["degree"] for n in sel) / len(sel)
        print(f"{r:8s} n={len(sel):3d} mean degree={md:5.2f} mean betweenness={mb:.4f}")
