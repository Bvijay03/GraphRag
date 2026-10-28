"""Inspector tool: inspect graphs, communities, prompts, summaries, and QA errors.

Allows examining every stage of the pipeline independently:
  1. Raw graphs: node counts, bridge/hub degree and betweenness
  2. Communities: Leiden partition, within vs cut triples, multi-hop questions
  3. Prompts & Summaries: community prompts and LLM summaries for any condition
  4. QA Errors: detailed failure breakdown with questions, gold answers, and responses

Usage:
    # General overview of a run directory:
    python inspect_run.py --dir runs/exp1

    # Inspect raw graph (Step 1):
    python inspect_run.py --dir runs/exp1 --seed 1 --raw

    # Inspect community partition & cut edges (Step 2):
    python inspect_run.py --dir runs/exp1 --seed 1 --community 0

    # Inspect prompt & summary for Condition 1 (default) or Condition 2 (topology):
    python inspect_run.py --dir runs/exp1 --seed 1 --community 0 --condition default
    python inspect_run.py --dir runs/exp1 --seed 1 --community 0 --condition topology

    # Inspect QA errors (Step 4):
    python inspect_run.py --dir runs/exp1 --seed 1 --qa-errors
"""
import argparse
import json
from pathlib import Path


def show_overview(base_dir):
    p = Path(base_dir)
    print(f"\n{'='*75}\nRun Overview: {p.resolve()}\n{'='*75}")

    raw_files = sorted((p / "graphs_raw").glob("*.json")) if (p / "graphs_raw").exists() else []
    comm_files = sorted((p / "communities").glob("*.json")) if (p / "communities").exists() else []
    
    sum_default = sorted((p / "summaries" / "default").glob("*.json")) if (p / "summaries" / "default").exists() else []
    sum_topo = sorted((p / "summaries" / "topology").glob("*.json")) if (p / "summaries" / "topology").exists() else []
    sum_ctrl = sorted((p / "summaries" / "control").glob("*.json")) if (p / "summaries" / "control").exists() else []
    
    eval_files = sorted((p / "graphs").glob("*.json")) if (p / "graphs").exists() else []

    print(f"Step 1 (Raw Graphs):         {len(raw_files)} generated")
    print(f"Step 2 (Communities):        {len(comm_files)} partitioned")
    print(f"Step 3 (Prompt 1 - Default): {len(sum_default)} summarized")
    print(f"Step 3 (Prompt 2 - Topology):{len(sum_topo)} summarized")
    print(f"Step 3 (Prompt 3 - Control): {len(sum_ctrl)} summarized")
    print(f"Step 4 (Evaluated Graphs):   {len(eval_files)} evaluated")

    if eval_files:
        print("\nEvaluated Graphs:")
        print(f"{'File':<25} | {'Triples':<8} | {'Cut':<6} | {'QA Def':<8} | {'QA Topo':<8} | {'QA Ctrl':<8}")
        print("-" * 75)
        for f in eval_files:
            try:
                data = json.loads(f.read_text())
                qa = data.get("qa", {})
                n_t = data.get("n_triples", "?")
                n_c = data.get("n_cut_triples", "?")
                def _get_acc(c):
                    v = qa.get(c)
                    if isinstance(v, dict):
                        return f"{v.get('acc', 0)*100:.1f}%"
                    elif isinstance(v, (int, float)):
                        return f"{v*100:.1f}%"
                    return "n/a"
                q_def = _get_acc("default")
                q_top = _get_acc("topology")
                q_ctrl = _get_acc("control")
                print(f"{f.name:<25} | {str(n_t):<8} | {str(n_c):<6} | {q_def:<8} | {q_top:<8} | {q_ctrl:<8}")
            except Exception as e:
                print(f"{f.name:<25} | Error reading: {e}")


def inspect_raw_graph(base_dir, p_out, seed):
    p = Path(base_dir)
    f = p / "graphs_raw" / f"pout{p_out}_seed{seed}.json"
    if not f.exists():
        print(f"Error: Raw graph file {f} not found.")
        return

    data = json.loads(f.read_text())
    nodes = data["nodes"]
    triples = data["triples"]

    print(f"\n{'='*75}")
    print(f"Step 1 Raw Graph: p_out={p_out}, seed={seed}")
    print(f"Total Nodes: {len(nodes)} | Total Triples: {len(triples)}")
    print(f"{'='*75}")

    roles = {"bridge": [], "hub": [], "regular": []}
    for name, meta in nodes.items():
        roles[meta["role"]].append(meta)

    print(f"{'Role':<10} | {'Count':<6} | {'Mean Degree':<12} | {'Mean Global Betweenness':<24}")
    print("-" * 65)
    for r in ("bridge", "hub", "regular"):
        sel = roles[r]
        if sel:
            md = sum(x["degree"] for x in sel) / len(sel)
            mb = sum(x["betweenness_global"] for x in sel) / len(sel)
            print(f"{r:<10} | {len(sel):<6} | {md:<12.2f} | {mb:<24.4f}")

    print(f"\nSample Triples (first 5 of {len(triples)}):")
    for t in triples[:5]:
        print(f"  {t[0]} -[{t[1]}]-> {t[2]}")


def inspect_community(base_dir, p_out, seed, comm_id, condition=None):
    p = Path(base_dir)
    comm_f = p / "communities" / f"pout{p_out}_seed{seed}.json"
    if not comm_f.exists():
        print(f"Error: Community file {comm_f} not found.")
        return

    comm_data = json.loads(comm_f.read_text())
    comm_str = str(comm_id)
    inputs = comm_data.get("inputs", {}).get(comm_str)

    if not inputs:
        print(f"Community {comm_id} not found. Available: {list(comm_data.get('inputs', {}).keys())}")
        return

    print(f"\n{'='*75}")
    print(f"Community {comm_id} (p_out={p_out}, seed={seed})")
    print(f"Within-community triples: {len(inputs['within'])} | Cut triples: {len(inputs['cut'])}")
    print(f"{'='*75}")

    print("Sample Within Triples:")
    for t in inputs["within"][:6]:
        print(f"  {t[0]} -[{t[1]}]-> {t[2]}")
    if len(inputs["within"]) > 6:
        print(f"  ... and {len(inputs['within']) - 6} more")

    print("\nSample Cut Triples:")
    for t in inputs["cut"][:6]:
        print(f"  {t[0]} -[{t[1]}]-> {t[2]}")
    if len(inputs["cut"]) > 6:
        print(f"  ... and {len(inputs['cut']) - 6} more")

    # Check summaries
    conds = [condition] if condition else ["default", "topology", "control"]
    for c in conds:
        sum_f = p / "summaries" / c / f"pout{p_out}_seed{seed}.json"
        if sum_f.exists():
            s_data = json.loads(sum_f.read_text()).get("summaries", {}).get(comm_str)
            if s_data:
                print(f"\n{'-'*75}")
                print(f"Condition [{c.upper()}] Prompt & Summary:")
                print(f"Marked nodes: {s_data.get('marked', [])}")
                print(f"Prompt tokens: {s_data.get('prompt_tokens', '?')}")
                print(f"Summary Text:\n{s_data.get('summary', '').strip()}")


def inspect_qa_errors(base_dir, p_out, seed, condition=None):
    p = Path(base_dir)
    graph_file = p / "graphs" / f"pout{p_out}_seed{seed}.json"
    if not graph_file.exists():
        print(f"Error: Evaluated graph file {graph_file} not found. Run step4_evaluate.py first.")
        return

    data = json.loads(graph_file.read_text())
    qa_records = data.get("qa_records", {})
    conditions = [condition] if condition else list(qa_records.keys())

    print(f"\n{'='*75}")
    print(f"QA Error Diagnosis: p_out={p_out}, seed={seed}")
    print(f"{'='*75}")

    for cond in conditions:
        records = qa_records.get(cond, [])
        wrong = [r for r in records if not r.get("correct", False)]
        raw_acc = data.get("qa", {}).get(cond, 0)
        acc = raw_acc.get("acc", 0) if isinstance(raw_acc, dict) else raw_acc

        print(f"\nCondition: [{cond.upper()}] — Accuracy: {acc*100:.1f}% ({len(records)-len(wrong)}/{len(records)})")
        print(f"Blocked by input budget: {len(blocked)} / {len(records)}")
        print(f"Incorrect answers to inspect ({len(wrong)}):\n")

        for idx, r in enumerate(wrong[:15]):
            b_flag = " [BLOCKED BY BUDGET]" if r.get("blocked") else ""
            print(f"  {idx+1}. Q-ID: {r.get('id')}{b_flag}")
            print(f"     Hops: {r.get('n_hops')} | Bridge nodes on path: {r.get('n_bridge_nodes')}")
            print(f"     Gold entity:     {r.get('gold')}")
            print(f"     Model response:  {r.get('response', 'n/a')}")
        if len(wrong) > 15:
            print(f"  ... and {len(wrong) - 15} more errors.")


def main():
    ap = argparse.ArgumentParser(description="Inspect run outputs, graphs, communities, prompts, and QA")
    ap.add_argument("--dir", required=True, help="run output directory (e.g. runs/exp1)")
    ap.add_argument("--seed", type=int, default=1, help="seed number")
    ap.add_argument("--p-out", type=float, default=0.004, help="p_out value")
    ap.add_argument("--raw", action="store_true", help="inspect raw graph (step 1)")
    ap.add_argument("--community", type=int, help="community id to inspect (step 2 & 3)")
    ap.add_argument("--condition", choices=["default", "topology", "control"], help="filter condition (step 3)")
    ap.add_argument("--qa-errors", action="store_true", help="show QA failure breakdown (step 4)")
    a = ap.parse_args()

    if a.raw:
        inspect_raw_graph(a.dir, a.p_out, a.seed)
    elif a.community is not None:
        inspect_community(a.dir, a.p_out, a.seed, a.community, a.condition)
    elif a.qa_errors:
        inspect_qa_errors(a.dir, a.p_out, a.seed, a.condition)
    else:
        show_overview(a.dir)


if __name__ == "__main__":
    main()
