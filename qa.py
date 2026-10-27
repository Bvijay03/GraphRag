"""M8b: multi-hop question answering (R9).

Questions are 2- or 3-hop chains over directed triples, e.g. "Start from A. Step 1: find the
entity that A works at. Step 2: ...". Every hop is unique (exactly one matching triple), and at
least one hop is a cut triple, so the question needs facts that cross a community boundary.

Visibility rule (no LLM retrieval): the answerer sees the summaries of every community that
contains an entity on the gold path. Questions whose hop triple is in NO community input (for
example dropped by the input budget) cannot be answered by any summary, so they are flagged
'blocked' and left out of accuracy.

Scoring is constrained: the answer must be one entity name; normalised exact match.
"""
import random
import re
from collections import defaultdict

from scipy import stats

from communities import cut_triples
from metrics import retained_l2
from relations import RELATION_CUES

INSTRUCTION = "Answer with only the entity name."


def make_questions(triples, membership, seed, n=40, hop_choices=(2, 3), node_meta=None,
                   max_tries=50000):
    rng = random.Random(seed)
    out_adj = defaultdict(lambda: defaultdict(list))
    for h, r, t in triples:
        out_adj[h][r].append(t)
    heads = sorted(out_adj)
    cut = set(cut_triples(triples, membership))
    questions, seen = [], set()
    for _ in range(max_tries):
        if len(questions) >= n:
            break
        n_hops = rng.choice(hop_choices)
        start = rng.choice(heads)
        cur, visited, path, ok = start, {start}, [], True
        for _ in range(n_hops):
            rels = sorted(r for r, ts in out_adj[cur].items() if len(ts) == 1)
            if not rels:
                ok = False
                break
            r = rng.choice(rels)
            nxt = out_adj[cur][r][0]
            if nxt in visited:
                ok = False
                break
            path.append((cur, r, nxt))
            visited.add(nxt)
            cur = nxt
        if not ok or not any(t in cut for t in path) or tuple(path) in seen:
            continue
        seen.add(tuple(path))
        steps = [f"Step {i + 1}: find the entity that "
                 f"{start if i == 0 else 'this entity'} {RELATION_CUES[r][0]}."
                 for i, (_h, r, _t) in enumerate(path)]
        questions.append({
            "id": len(questions), "start": start, "hops": path, "gold": path[-1][2],
            "text": f"Start from {start}. " + " ".join(steps) +
                    f" Which entity do you reach? {INSTRUCTION}",
            "n_cut_hops": sum(t in cut for t in path),
            "n_bridge_nodes": (sum(node_meta[e]["role"] == "bridge" for e in visited)
                               if node_meta else None),
        })
    return questions


def path_entities(q):
    return [q["start"]] + [t for _h, _r, t in q["hops"]]


def visible_summary(q, membership, summaries):
    comms = sorted({membership[e] for e in path_entities(q)})
    return comms, "\n".join(f"COMMUNITY {c}: {summaries[c]['summary']}" for c in comms)


def is_blocked(q, inputs):
    entered = {t for v in inputs.values() for t, _s in v["input"]}
    return any(hop not in entered for hop in q["hops"])


def normalize(answer):
    return answer.strip().strip(".,;:!\"'").lower()


def score_answer(response, gold):
    norm_gold = normalize(gold)
    # 1. Exact match on normalized full response (original behavior)
    if normalize(response) == norm_gold:
        return True
    # 2. Explicit 'Answer: <entity>' line
    m = re.findall(r"(?i)^\s*Answer:\s*([A-Za-z0-9_]+)\s*$", response, flags=re.MULTILINE)
    if m and normalize(m[-1]) == norm_gold:
        return True
    # 3. Last non-empty line is exactly the entity
    lines = [l.strip() for l in response.strip().splitlines() if l.strip()]
    if lines and normalize(lines[-1]) == norm_gold:
        return True
    return False


class OracleReaderAnswerer:
    """Mock answerer: answers correctly iff every hop is stated (L2) in the visible text."""

    def answer(self, q, text):
        return q["gold"] if all(retained_l2(h, text) for h in q["hops"]) else "UNKNOWN"


class LLMAnswerer:
    """Real answerer: pinned model, constrained answer. Used on the GPU machine."""

    def __init__(self, llm):
        self.llm = llm

    def build_prompt(self, q, text):
        return ("Use only the community summaries below to answer the question.\n\n"
                f"{text}\n\nQuestion: {q['text']}\n"
                "IMPORTANT: Respond with ONLY the final entity name (no explanation).")

    def answer(self, q, text):
        return self.llm.generate(self.build_prompt(q, text)).text


def evaluate_questions(questions, membership, summaries, inputs, answerer):
    records = []
    for q in questions:
        blocked = is_blocked(q, inputs)
        _comms, text = visible_summary(q, membership, summaries)
        resp = answerer.answer(q, text)
        records.append({"id": q["id"], "blocked": blocked,
                        "correct": bool(score_answer(resp, q["gold"])),
                        "response": resp, "gold": q["gold"],
                        "n_hops": len(q["hops"]), "n_bridge_nodes": q["n_bridge_nodes"]})
    return records


def accuracy(records):
    live = [r for r in records if not r["blocked"]]
    return {"acc": (sum(r["correct"] for r in live) / len(live)) if live else float("nan"),
            "n": len(live), "n_blocked": len(records) - len(live)}


def compare_accuracy(records_a, records_b):
    """Paired comparison on the same questions (exact sign test on discordant pairs)."""
    b_by_id = {r["id"]: r for r in records_b}
    pairs = [(r["correct"], b_by_id[r["id"]]["correct"]) for r in records_a
             if not r["blocked"] and not b_by_id[r["id"]]["blocked"]]
    a_only = sum(a and not b for a, b in pairs)
    b_only = sum(b and not a for a, b in pairs)
    p = stats.binomtest(b_only, a_only + b_only, 0.5).pvalue if a_only + b_only else 1.0
    return {"acc_a": sum(a for a, _ in pairs) / len(pairs),
            "acc_b": sum(b for _, b in pairs) / len(pairs),
            "a_only": a_only, "b_only": b_only, "p": float(p), "n": len(pairs)}
