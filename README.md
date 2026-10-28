# MP2: Node centrality and fact retention in hierarchical GraphRAG community summaries

**Core Research Question:** When a knowledge graph is split into hierarchical communities and each community is compressed into an LLM summary, do high-betweenness **bridge** nodes lose more facts than **hub** nodes of the same degree? Is the loss caused by the partition (cut edges) or by the LLM's compression? Does a topology-aware prompt help?

---

## ⚡ Fine-Grained Step-by-Step Architecture

Every phase of the experiment runs independently, saves its outputs, and can be inspected or tested separately. You can even run **one prompt condition per day**:

```
[Step 1: Graph Gen] ──► [Step 2: Communities] ──► [Step 3: Summaries] ──► [Step 4: Evaluate] ──► [Step 5: Report]
 (Local / 1 sec)         (Local / 2 sec)          (GPU: per condition)     (Local / 2 sec)        (Local / 1 sec)
                                                        │
                                     ┌──────────────────┼──────────────────┐
                                     ▼                  ▼                  ▼
                               [3a: Default]      [3b: Topology]      [3c: Control]
                                 (~35 min)          (~35 min)           (~35 min)
```

| Step | Script | Hardware | Time (40 Graphs) | What It Does & What It Saves |
| :--- | :--- | :---: | :---: | :--- |
| **1. Graphs** | [`step1_graph.py`](file:///d:/MP_2/Code/mp2/step1_graph.py) | Laptop | **~1 second** | Builds SBM graph, planted bridges & hubs with matched degree ($k=12$).<br>Saves to `<out>/graphs_raw/poutX_seedY.json` |
| **2. Communities** | [`step2_communities.py`](file:///d:/MP_2/Code/mp2/step2_communities.py) | Laptop | **~2 seconds** | Runs Leiden hierarchy (L0/L1), splits within vs cut strata, generates QA questions.<br>Saves to `<out>/communities/poutX_seedY.json` |
| **3. Summarize** | [`step3_summarize.py`](file:///d:/MP_2/Code/mp2/step3_summarize.py) | **Kaggle GPU** | **~35 min / prompt** | **Runs one prompt condition at a time!**<br>`--condition default`: `<out>/summaries/default/`<br>`--condition topology`: `<out>/summaries/topology/`<br>`--condition control`: `<out>/summaries/control/` |
| **4. Evaluate** | [`step4_evaluate.py`](file:///d:/MP_2/Code/mp2/step4_evaluate.py) | Laptop | **~2 seconds** | Evaluates fact retention (L1/L2, within vs cut strata) & deterministic Oracle QA.<br>Works with whatever prompt conditions have been completed! Saves to `<out>/graphs/` |
| **5. Report** | [`step5_report.py`](file:///d:/MP_2/Code/mp2/step5_report.py) | Laptop | **~1 second** | Aggregates all graph records, runs t-tests, Holm correction, sign-tests, and writes `summary.md` and `results.json`. |

---

## 🔍 Diagnostic Inspector (`inspect_run.py`)

Inspect every stage of the pipeline independently:

```bash
# 1. Overview of run status, completed stages, and QA accuracy:
python inspect_run.py --dir runs/exp1

# 2. Inspect raw graph (degree matching, bridge vs hub count, betweenness):
python inspect_run.py --dir runs/exp1 --seed 1 --raw

# 3. Inspect community partition & cut edges:
python inspect_run.py --dir runs/exp1 --seed 1 --community 0

# 4. Inspect prompt input and LLM summary for a condition:
python inspect_run.py --dir runs/exp1 --seed 1 --community 0 --condition default
python inspect_run.py --dir runs/exp1 --seed 1 --community 0 --condition topology

# 5. Diagnose QA errors (shows question, hops, bridge count, gold entity, and model response):
python inspect_run.py --dir runs/exp1 --seed 1 --qa-errors
```

---

## 🚀 Running on Kaggle

1. Open [`kaggle_mp2_modular.ipynb`](file:///d:/MP_2/Code/mp2/kaggle_mp2_modular.ipynb) in Kaggle.
2. Select **Accelerator: GPU T4**, **Internet: On**, and **Persistence: Files only**.
3. Run the cells:
   ```bash
   # Cell 1-2: Environment setup
   !apt-get update -qq && apt-get install -y -qq zstd
   !curl -fsSL https://ollama.com/install.sh | sh
   !pip install -q ollama
   !ollama pull qwen3:8b

   # Cell 3: Clone repo & install dependencies
   !git clone https://github.com/Bvijay03/GraphRag.git /kaggle/working/mp2
   %cd /kaggle/working/mp2
   !pip install -q -r requirements.txt

   # Cell 5: Step 1 - Generate Graphs (1 second)
   !python step1_graph.py --config config_kaggle.yaml --out runs/kaggle_exp

   # Cell 6: Step 2 - Leiden Communities (2 seconds)
   !python step2_communities.py --config config_kaggle.yaml --out runs/kaggle_exp

   # Cell 7: Step 3a - Prompt 1 (Default GraphRAG summary, ~35 min for 40 graphs)
   !python step3_summarize.py --config config_kaggle.yaml --out runs/kaggle_exp --condition default --model qwen3:8b

   # Cell 8: Step 3b - Prompt 2 (Topology prompt, run now or next session)
   !python step3_summarize.py --config config_kaggle.yaml --out runs/kaggle_exp --condition topology --model qwen3:8b

   # Cell 9: Step 3c - Prompt 3 (Control prompt, run now or next session)
   !python step3_summarize.py --config config_kaggle.yaml --out runs/kaggle_exp --condition control --model qwen3:8b

   # Cell 10: Step 4 - Evaluate Retention & QA (2 seconds)
   !python step4_evaluate.py --config config_kaggle.yaml --out runs/kaggle_exp

   # Cell 11: Step 5 - Generate Report & Render summary.md (1 second)
   !python step5_report.py --config config_kaggle.yaml --out runs/kaggle_exp
   ```

4. **Resumable & Incremental:** Every single graph and condition is saved to disk immediately. If interrupted, simply restart the cell — already processed files are automatically skipped!

---

## 💻 Local Testing & Unified Runner

Run all 77 automated unit and regression tests on your laptop:
```bash
python -m pytest -q
```

Run a fast mock experiment locally across all conditions:
```bash
python step1_graph.py --config config.yaml --n-graphs 2 --out runs/test
python step2_communities.py --config config.yaml --out runs/test
python step3_summarize.py --config config.yaml --out runs/test --condition all --backend mock
python step4_evaluate.py --config config.yaml --out runs/test
python step5_report.py --config config.yaml --out runs/test
```
