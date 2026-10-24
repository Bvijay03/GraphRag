"""M6: prompts, summarizers (LLM and template) and per-community summary runs.

Three prompt conditions (R8). The marker sentence is IDENTICAL in 'topology' and 'control';
only the marked names differ:
  default   plain summary prompt
  topology  marks the top-k betweenness nodes of the community
  control   marks k random nodes of the community that are NOT planted bridges and NOT in
            the topology set
"""
import random

from llm import FACT_PREFIX
from metrics import retained_l2
from relations import render

HEADER = "Write a summary of the following facts about one community of entities."
MARKER = ("These entities connect this community to others, so make sure they appear in "
          "your summary: {names}")
LEVEL1_HEADER = "Write a summary of the following community summaries."


def community_nodes(input_items):
    return sorted({e for (h, _, t), _s in input_items for e in (h, t)})


def select_marked(input_items, node_meta, condition, k, seed):
    """Return the list of names to mark for this community and condition."""
    if condition == "default":
        return []
    nodes = community_nodes(input_items)
    ranked = sorted(nodes, key=lambda n: (-node_meta[n]["betweenness_global"], n))
    top = ranked[:k]
    if condition == "topology":
        return top
    if condition == "control":
        pool = [n for n in nodes if node_meta[n]["role"] != "bridge" and n not in top]
        return sorted(random.Random(seed).sample(pool, min(k, len(pool))))
    raise ValueError(f"unknown condition: {condition}")


def build_prompt(input_items, marked=None):
    parts = [HEADER]
    if marked:
        parts.append(MARKER.format(names=", ".join(marked)))
    parts.append("")
    parts += [FACT_PREFIX + render(t) for t, _stratum in input_items]
    return "\n".join(parts)


def template_summary(input_items):
    """No LLM: restates every input triple. Compression loss is zero by construction,
    so any loss left over comes from partitioning or the input budget."""
    return " ".join(render(t) for t, _s in input_items)


def summarize_community(llm, input_items, marked=None, context_limit=None):
    prompt = build_prompt(input_items, marked)
    res = llm.generate(prompt)
    near = bool(context_limit) and res.prompt_tokens >= 0.9 * context_limit
    return {"summary": res.text, "prompt_tokens": res.prompt_tokens, "near_limit": near,
            "n_input": len(input_items), "marked": list(marked or [])}


def summarize_level0(llm, inputs, node_meta, condition="default", k=3, seed=0,
                     context_limit=None, use_template=False):
    out = {}
    for c, v in inputs.items():
        items = v["input"]
        if use_template:
            out[c] = {"summary": template_summary(items), "prompt_tokens": 0,
                      "near_limit": False, "n_input": len(items), "marked": []}
            continue
        marked = select_marked(items, node_meta, condition, k, seed + int(c))
        out[c] = summarize_community(llm, items, marked, context_limit)
    return out


def build_level1_prompt(child_summaries):
    """child_summaries: {level0_community: summary text}. Sentences go on FACT lines so the
    mock backend can act on them."""
    from metrics import split_sentences
    parts = [LEVEL1_HEADER, ""]
    for c in sorted(child_summaries):
        parts.append(f"COMMUNITY {c}:")
        parts += [FACT_PREFIX + s for s in split_sentences(child_summaries[c])]
    return "\n".join(parts)


def summarize_level1(llm, child_summaries, context_limit=None):
    prompt = build_level1_prompt(child_summaries)
    res = llm.generate(prompt)
    near = bool(context_limit) and res.prompt_tokens >= 0.9 * context_limit
    return {"summary": res.text, "prompt_tokens": res.prompt_tokens, "near_limit": near}


def triples_entering_level1(input_items_by_child, child_summaries):
    """Facts that entered a level-1 input = facts retained (L2) in their level-0 summary."""
    entered = []
    for c, items in input_items_by_child.items():
        s = child_summaries[c]
        entered += [t for t, _stratum in items if retained_l2(t, s)]
    return entered
