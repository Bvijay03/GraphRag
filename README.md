# MP2: Node centrality and fact retention in hierarchical GraphRAG community summaries

Question: when a knowledge graph is split into hierarchical communities and each community is
compressed into an LLM summary, do high-betweenness **bridge** nodes lose more facts than
**hub** nodes of the same degree? Is the loss caused by the partition (cut edges) or by the
LLM's compression? Does a topology-aware prompt help?

Everything runs on a laptop with the mock backends. Real summaries need the GPU machine.

## Install and test (laptop)

```
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                                              # about 80 s
python run_experiment.py --backend planted --n-graphs 6 --out runs/demo
```

Open `runs/demo/summary.md`. The `planted` backend contains a known effect (bridges lose
more), so the demo shows the analysis detecting it. Do not report demo numbers as results.

## Pipeline and modules

| Stage | File | What it does |
| --- | --- | --- |
| 1 Graph | `kg_gen.py` | Stochastic block model; planted bridges and hubs with **matched degree** and different betweenness; 8 relation types |
| 2 Sentences | `relations.py` | One oracle sentence per triple (no extraction step), relation vocabulary |
| 3 Communities | `communities.py` | Two-level Leiden (level 1 clusters the level-0 communities); cut triples; summary-input tracking |
| 4 Summaries | `summarize.py` | Three prompt conditions, template baseline, token logging, level-1 summaries |
| 5 Metrics | `metrics.py` | Retention L1 / L2 and distorted facts, per node |
| 6 Analysis | `analysis.py` | Bridge vs hub contrast, Spearman and partial Spearman, Holm correction |
| 7 Interventions | `interventions.py` | Default vs topology vs random control (lift and difference in differences) |
| 8 QA | `qa.py` | 2-3 hop questions crossing communities, constrained scoring |
| Driver | `run_experiment.py` | Loops over graphs, resumable, writes `results.json` and `summary.md` |
| LLM | `llm.py` | `MockLLM`, `OllamaLLM` (Qwen via Ollama) |
| Test mocks | `planted_mock.py` | Known planted effects for validating the analysis |

## Design decisions to be able to defend

- **Matched degree.** With one sentence per triple, an entity's mention count equals its degree,
  so bridges and hubs share the same degree. Betweenness is then the only structural difference.
- **Retention is measured only over facts that entered the summary input.** Cut triples are
  put into the input of both adjacent communities as a separate stratum, so partition loss and
  compression loss are separated. The template summarizer (`--backend template`) has zero
  compression loss by construction.
- **L2 is the primary metric.** L1 (both names appear anywhere) is inflated by degree and is
  biased against bridges even with no effect (see `test_l1_is_biased_against_bridges_even_under_null`).
- **Unit of analysis is the node.** Statistics are computed per graph, then aggregated over
  graphs, with Holm correction across all tests.
- **Control prompt.** Same wording and same number of marked nodes as the topology prompt, but
  random non-bridge nodes. The difference in differences separates a topology-specific effect
  from a generic "you told me to mention these names" effect.
- **Exact entity matching, no fuzzy matching.** Names such as Kavortri12 and Kavortri13 differ
  by one character.

## Running the real experiment (GPU machine)

1. Follow the setup guide: driver, Python, Ollama, `ollama pull qwen3:14b`, `pip install -r requirements.txt`.
2. Run its checks first: GPU in use, prompt token count against `num_ctx`, thinking off, same prompt twice gives the same output.
3. In `config.yaml` set `llm.backend: ollama` and pin `llm.model`. Keep `num_ctx` explicit.
4. `python run_experiment.py --out runs/real1`. Interrupted runs resume from the per-graph files in `runs/real1/graphs/`.
5. Read `summary.md`, and check "prompts near the context limit" is 0.

## Outputs

`config_used.yaml`, `env.json` (versions, config hash), `graphs/*.json` (one file per graph with
node tables, summaries, marked nodes, QA records), `results.json`, `summary.md`.

## Reading the results

- Primary result: `l2 diff` (bridge minus hub retention at matched degree). The secondary
  correlations use betweenness; the partial versions control for degree, input size and count.
- Report the cut-triple stratum separately: it is the test of the mechanism.
- Report baselines next to every lift: lift is capped by baseline retention.
- Interventions and QA are only meaningful with the real model.

## Requirements to code to tests

| Req | Code | Tests |
| --- | --- | --- |
| R1 graph | `kg_gen.py` | `test_kg_gen.py` |
| R2 sentences, mentions | `relations.py` | `test_corpus.py` |
| R3 communities | `communities.py` | `test_communities.py` |
| R4 input tracking | `communities.py`, `summarize.py` | `test_communities.py`, `test_summarize.py` |
| R5 summaries, template, token logging | `summarize.py`, `llm.py` | `test_summarize.py`, `test_llm.py` |
| R6 retention | `metrics.py` | `test_metrics.py` (L3 judge: not yet) |
| R7 analysis | `analysis.py` | `test_analysis.py` (planted and null data) |
| R8 interventions | `interventions.py` | `test_interventions.py` |
| R9 QA | `qa.py` | `test_qa.py` |
| R10 reproducibility | `run_experiment.py`, `config.yaml` | `test_run_experiment.py` |

## Known gaps and limitations

- **No L3 judge and no hand-labelled validation yet** (R6). L2 misses correct paraphrases it
  does not know, so it under-counts retention. Add phrasings from real outputs, then validate
  on about 50 hand-labelled facts.
- `OllamaLLM` is tested only against a stub. Real behaviour (token counts, thinking mode) must
  be verified on the GPU machine.
- The input budget is counted in triples; the real token count is only logged.
- The bridge-vs-hub contrast uses planted role labels, so it works on synthetic graphs only.
- The summarizer receives oracle sentences, so extraction errors are not studied.
- Level-1 summaries are implemented but the driver does not run or score them yet.
- Results come from one pinned model and synthetic data; they may not transfer to real corpora.
