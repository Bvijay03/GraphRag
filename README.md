# MP2: Node centrality and fact retention in hierarchical GraphRAG community summaries

**Core Research Question:** When a knowledge graph is split into hierarchical communities and each community is compressed into an LLM summary, do high-betweenness **bridge** nodes lose more facts than **hub** nodes of the same degree? Is the loss caused by the partition (cut edges) or by the LLM's compression? Does a topology-aware prompt help?

---

## ⚡ Fast Modular Pipeline (Recommended for Kaggle & GPU Runs)

Instead of running an monolithic script that does graph generation, Leiden clustering, LLM summarization, QA answering, and statistics in one long 20+ hour loop, the experiment is broken into **4 clean, independent, resumable steps**:

```
[Step 1: Prepare]  ──►  [Step 2: Summarize]  ──►  [Step 3: Evaluate]  ──►  [Step 4: Report]
 (Local / 5 sec)          (GPU / ~2 hours)          (Local / 3 sec)         (Local / 1 sec)
```

| Script | What it does | Hardware | Time (40 Graphs) | Output Saved |
| :--- | :--- | :---: | :---: | :--- |
| **`step1_prepare.py`** | SBM graph generation, Leiden community hierarchy, within/cut triple stratification, question generation | CPU (laptop) | **~5 seconds** | `<out>/prepared/poutX_seedY.json` |
| **`step2_summarize.py`** | Generates community summaries under 3 conditions (`default`, `topology`, `control`). **The only step needing GPU.** | **GPU (Kaggle T4)** | **~2 hours** | `<out>/summaries/poutX_seedY.json` |
| **`step3_evaluate.py`** | Evaluates fact retention (L1, L2, within/cut strata), DiD lift, and exact Oracle QA accuracy deterministically | CPU (laptop) | **~3 seconds** | `<out>/graphs/poutX_seedY.json` |
| **`step4_report.py`** | Aggregates all graph records, runs t-tests, Holm correction, sign-tests, and writes reports | CPU (laptop) | **~1 second** | `<out>/results.json`<br>`<out>/summary.md` |

### Why is this 7x Faster?
In the monolithic runner, Qwen is asked to answer 120 QA questions per graph on top of generating 18 summaries. On Kaggle T4, this blew up each graph to ~28 minutes (18+ hours total). In the modular pipeline:
- **Summaries (the actual experiment):** 18 LLM calls per graph (~2.5 minutes per graph).
- **QA & Retention (the evaluation):** Deterministically scored in Step 3 in milliseconds using `OracleReaderAnswerer`. Fact retention is measured directly against what the model generated, removing answering noise entirely.

---

## 🚀 Running on Kaggle

1. Open the included notebook: [`kaggle_mp2_modular.ipynb`](file:///d:/MP_2/Code/mp2/kaggle_mp2_modular.ipynb)
2. In Kaggle, create a new notebook with:
   - **Accelerator:** GPU T4 x2 or T4 x1
   - **Internet:** On
   - **Persistence:** Files only
3. Execute the notebook cells:
   ```bash
   # Cell 1-2: Install zstd, Ollama & pull model
   !apt-get update -qq && apt-get install -y -qq zstd
   !curl -fsSL https://ollama.com/install.sh | sh
   !ollama pull qwen3:8b

   # Cell 3: Clone repo
   !git clone https://github.com/Bvijay03/GraphRag.git /kaggle/working/mp2
   %cd /kaggle/working/mp2

   # Cell 4: Step 1 - Prepare (5 seconds)
   !python step1_prepare.py --config config_kaggle.yaml --out runs/exp1

   # Cell 5: Step 2 - Summarize (GPU accelerated, ~2 hours)
   !python step2_summarize.py --config config_kaggle.yaml --out runs/exp1 --backend ollama

   # Cell 6: Step 3 - Evaluate (3 seconds)
   !python step3_evaluate.py --config config_kaggle.yaml --out runs/exp1

   # Cell 7: Step 4 - Generate Report (1 second)
   !python step4_report.py --config config_kaggle.yaml --out runs/exp1
   ```
4. **Resumable by default:** Every graph's summary is saved to disk immediately. If Kaggle times out or disconnects, simply restart and re-run — already completed graphs are automatically skipped!

---

## 🔍 Diagnostic Inspector (`inspect_run.py`)

Want to inspect each prompt, see what the LLM generated, check which facts were lost, or analyze why specific QA questions failed?

```bash
# 1. Overview of completed graphs and QA scores
python inspect_run.py --dir runs/exp1

# 2. Inspect the prompt and LLM summary for Community 0 of Graph seed 1
python inspect_run.py --dir runs/exp1 --seed 1 --community 0

# 3. Inspect QA failures and see why answers were marked wrong
python inspect_run.py --dir runs/exp1 --seed 1 --qa-errors

# 4. Filter errors by condition
python inspect_run.py --dir runs/exp1 --seed 1 --qa-errors --condition default
```

---

## 💻 Local Testing & Unified Runner

Run all 77 automated tests on your laptop:
```bash
pip install -r requirements.txt
python -m pytest -q
```

Run a fast mock experiment locally:
```bash
python run_experiment.py --backend mock --n-graphs 2 --out runs/test
```

Or run the unified runner with fast Oracle QA:
```bash
python run_experiment.py --config config.yaml --oracle-qa --out runs/fast_run
```

---

## 📂 Codebase Architecture

| Stage | File | Purpose |
| :--- | :--- | :--- |
| **Modular Pipeline** | `step1_prepare.py`<br>`step2_summarize.py`<br>`step3_evaluate.py`<br>`step4_report.py` | 4-step decoupled experiment pipeline (instant prep, GPU summarization, deterministic evaluation, statistical reporting) |
| **Diagnostic Tool** | `inspect_run.py` | CLI tool to inspect individual community prompts, LLM summaries, retained facts, and QA failures |
| **Graph Generation** | `kg_gen.py` | Stochastic block model with matched-degree planted bridges and hubs |
| **Sentences & Corpus**| `relations.py` | Deterministic sentence generator for triples (removes extraction errors) |
| **Communities** | `communities.py` | Two-level Leiden hierarchical clustering; within/cut strata splitting |
| **Summarization** | `summarize.py` | Community prompts under `default`, `topology`, and `control` conditions |
| **Metrics** | `metrics.py` | Retention L1 / L2 and fact distortion detection |
| **Analysis** | `analysis.py` | Bridge vs hub contrast, Spearman & partial Spearman, Holm correction |
| **Interventions** | `interventions.py` | DiD (Difference-in-Differences) lift computation |
| **QA System** | `qa.py` | Multi-hop cross-community QA with `LLMAnswerer` and `OracleReaderAnswerer` |
| **Unified Runner** | `run_experiment.py` | End-to-end monolithic runner (with `--oracle-qa` option) |
| **LLM Backends** | `llm.py`<br>`planted_mock.py` | `MockLLM`, `OllamaLLM`, and `MarkerResponsiveMock` |

---

## 🔬 Design Decisions to Defend in Thesis / Paper

- **Matched degree:** With one sentence per triple, an entity's mention count equals its degree. Bridges and hubs share the exact same degree ($k=12$). Betweenness centrality is the **only** structural difference.
- **Stratified input tracking:** Cut triples are provided to both adjacent communities as a distinct stratum. This strictly isolates **partition loss** (information cut by clustering) from **compression loss** (information omitted by LLM summary).
- **L2 is the primary metric:** L1 (names co-occurring anywhere in summary) is biased against bridges even under the null hypothesis. L2 requires the semantic relation to be preserved.
- **Node-level unit of analysis:** Retention statistics are calculated per node, aggregated per graph, and tested with Holm family-wise error correction.
- **Random control prompt:** The control prompt marks the same number of nodes as the topology prompt ($k=3$), but selects random non-bridge nodes. Difference-in-Differences separates genuine topology-awareness from generic attention priming.
