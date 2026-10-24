"""Relation vocabulary shared by corpus_gen (writes sentences) and metrics (reads summaries).

Every cue is written so that the sentence reads "<head> <cue> <tail>."
The first cue of each relation is the canonical one used to render the oracle sentence.
"""

RELATION_CUES = {
    "works_at":      ["works at", "is employed by", "is employed at"],
    "located_in":    ["is located in", "is based in", "is situated in"],
    "founded_by":    ["was founded by", "was started by", "was established by"],
    "funded_by":     ["is funded by", "is financed by", "receives funding from"],
    "partners_with": ["partners with", "collaborates with", "is a partner of"],
    "member_of":     ["is a member of", "belongs to"],
    "supplies":      ["supplies", "provides goods to"],
    "acquired":      ["acquired", "bought", "took over"],
}


def render(triple):
    """One oracle sentence per triple (no extraction step)."""
    head, rel, tail = triple
    return f"{head} {RELATION_CUES[rel][0]} {tail}."
