"""Inspector tool: diagnose prompts, summaries, retention, and QA failures.

Allows examining exactly what went into any prompt, what the LLM produced,
which facts were retained vs dropped, and why specific QA questions failed.

Usage:
    # 1. Summary overview of a run directory:
    python inspect_run.py --dir runs/exp1

    # 2. Inspect prompts and summaries for a specific graph and community:
    python inspect_run.py --dir runs/exp1 --seed 1 --community 0

    # 3. Inspect QA failures (wrong answers and reasons):
    python inspect_run.py --dir runs/exp1 --seed 1 --qa-errors

    # 4. Filter QA errors by condition:
    python inspect_run.py --dir runs/exp1 --seed 1 --qa-errors --condition default
"""
import argparse
import json
from pathlib import Path


def show_overview(base_dir):
    p = Path(base_dir)
    print(f"\n{'='*70}\nRun Overview: {p.resolve()}\n{'='*70}")

    prep_files = sorted((p / "prepared").glob("*.json")) if (p / "prepared").exists() else []
    sum_files = sorted((p / "summaries").glob("*.json")) if (p / "summaries").exists() else []
    graph_files = sorted((p / "graphs").glob("*.json")) if (p / "graphs").exists() else []

    print(f"Prepared graphs:  {len(prep_files)}")
    print(f"Summarized graphs: {len(sum_files)}")
    print(f"Evaluated graphs:  {len(graph_files)}")

    if not graph_files:
        print("\nNo evaluated graph files found in graphs/.")
        return

    print("\nEvaluated Graphs:")
    print(f"{'File':<25} | {'Triples':<8} | {'Cut':<6} | {'QA Def':<8} | {'QA Topo':<8} | {'QA Ctrl':<8}")
    print("-" * 75)
    for f in graph_files:
        try:
            data = json.loads(f.read_text())
            qa = data.get("qa", {})
            n_t = data.get("n_triples", "?")
            n_c = data.get("n_cut_triples", "?")
            q_def = f"{qa.get('default', 0):.2f}" if "default" in qa else "n/a"
            q_top = f"{qa.get('topology', 0):.2f}" if "topology" in qa else "n/a"
            q_ctrl = f"{qa.get('control', 0):.2f}" if "control" in qa else "n/a"
            print(f"{f.name:<25} | {str(n_t):<8} | {str(n_c):<6} | {q_def:<8} | {q_top:<8} | {q_ctrl:<8}")
        except Exception as e:
            print(f"{f.name:<25} | Error reading file: {e}")


def inspect_community(base_dir, p_out, seed, comm_id):
    p = Path(base_dir)
    prep_file = p / "prepared" / f"pout{p_out}_seed{seed}.json"
    sum_file = p / "summaries" / f"pout{p_out}_seed{seed}.json"

    if not prep_file.exists():
        print(f"Error: {prep_file} not found.")
        return

    prep = json.loads(prep_file.read_text())
    comm_str = str(comm_id)

    inputs = prep.get("inputs", {}).get(comm_str)
    if not inputs:
        print(f"Community {comm_id} not found in prepared graph. Available: {list(prep.get('inputs', {}).keys())}")
        return

    print(f"\n{'='*75}")
    print(f"Graph p_out={p_out}, seed={seed} — Community {comm_id}")
    print(f"{'='*75}")
    print(f"Within-community triples ({len(inputs['within'])}):")
    for t in inputs["within"][:10]:
        print(f"  {t[0]} -[{t[1]}]-> {t[2]}")
    if len(inputs["within"]) > 10:
        print(f"  ... and {len(inputs['within']) - 10} more")

    print(f"\nCut triples ({len(inputs['cut'])}):")
    for t in inputs["cut"][:10]:
        print(f"  {t[0]} -[{t[1]}]-> {t[2]}")
    if len(inputs["cut"]) > 10:
        print(f"  ... and {len(inputs['cut']) - 10} more")

    if not sum_file.exists():
        print(f"\nNo summaries generated yet for this graph ({sum_file.name} missing).")
        return

    sums = json.loads(sum_file.read_text()).get("summaries", {})
    for cond in ("default", "topology", "control"):
        c_data = sums.get(cond, {}).get(comm_str)
        if not c_data:
            continue
        print(f"\n{'-'*75}")
        print(f"Condition: {cond.upper()}")
        print(f"Marked nodes: {c_data.get('marked', [])}")
        print(f"Tokens: {c_data.get('prompt_tokens', '?')}")
        print(f"Summary Output:\n{c_data.get('summary', '').strip()}")


def inspect_qa_errors(base_dir, p_out, seed, condition=None):
    p = Path(base_dir)
    graph_file = p / "graphs" / f"pout{p_out}_seed{seed}.json"
    if not graph_file.exists():
        print(f"Error: {graph_file} not found.")
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
        blocked = [r for r in records if r.get("blocked", False)]
        acc = data.get("qa", {}).get(cond, 0)

        print(f"\nCondition: [{cond}] — Accuracy: {acc*100:.1f}% ({len(records)-len(wrong)}/{len(records)})")
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
    ap = argparse.ArgumentParser(description="Inspect run outputs, prompts, and errors")
    ap.add_argument("--dir", required=True, help="run output directory (e.g. runs/exp1)")
    ap.add_argument("--seed", type=int, default=1, help="seed number")
    ap.add_argument("--p-out", type=float, default=0.004, help="p_out value")
    ap.add_argument("--community", type=int, help="community id to inspect")
    ap.add_argument("--qa-errors", action="store_true", help="show QA failure breakdown")
    ap.add_argument("--condition", choices=["default", "topology", "control"], help="filter condition")
    a = ap.parse_args()

    if a.community is not None:
        inspect_community(a.dir, a.p_out, a.seed, a.community)
    elif a.qa_errors:
        inspect_qa_errors(a.dir, a.p_out, a.seed, a.condition)
    else:
        show_overview(a.dir)


if __name__ == "__main__":
    main()
