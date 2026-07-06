# Sentiment & Sarcasm Classification across English Varieties (BESSTIE-CW-26)

Sentiment and sarcasm classification on the **BESSTIE-CW-26** dataset across three
varieties of English — Australian (`en-AU`), Indian (`en-IN`), and British (`en-UK`) —
from classical baselines through encoder fine-tuning to Qwen2.5-3B Q-LoRA adapters, with
a cross-variety transfer study, error analysis, and a deployable routing service.

The project is delivered as a clean, importable package ([`src/`](src/)) with one entry
script per section ([`scripts/`](scripts/)), so every table and figure can be regenerated
from the cached results **without re-training**.

---

## Headline results (sarcasm unless noted; mean over seeds {42, 123})

| Stage | Model | Train → Test | Macro-F1 |
|---|---|---|---|
| Classical baseline | TF-IDF + LogReg | pooled → pooled | 0.626 |
| Encoder, sentiment | **DeBERTa-v3-large** | pooled → pooled | **0.911** (best sentiment) |
| Encoder, sarcasm | RoBERTa-large | pooled → pooled | 0.717 |
| Cross-variety | DeBERTa-v3-base | en-AU → en-AU | 0.748 |
| Q-LoRA, per-variety | **Qwen2.5-3B + QLoRA** | en-AU → en-AU | **0.771** (best sarcasm) |
| Q-LoRA, pooled | Qwen2.5-3B + QLoRA | pooled → en-UK | 0.747 |
| Threshold-tuned | Qwen2.5-3B + QLoRA | en-AU → en-AU | 0.768 |

Full per-experiment numbers: [`outputs/tables/08_FINAL_COMPARISON.csv`](outputs/tables/08_FINAL_COMPARISON.csv)
and the master registry [`outputs/tables/07_ALL_EXPERIMENTS_SUMMARY.csv`](outputs/tables/07_ALL_EXPERIMENTS_SUMMARY.csv).

---

## What's in the study

- **Q1 — EDA & vocabulary.** Label distributions, class imbalance (sarcasm is ~7% positive
  in en-IN/en-UK), text-length stats, pairwise vocabulary Jaccard similarity, variety-specific
  wordclouds, and a pooled-vs-Reddit-only domain ablation that separates the *variety* gap
  from the *domain* gap.
- **Q2.1 — Classical vs PTLM.** TF-IDF+LogReg and GloVe+SVM baselines vs fine-tuned
  RoBERTa-large / DeBERTa-v3-large (bf16, gradient checkpointing, weighted CE for sarcasm).
- **Q2.2 — Cross-variety transfer.** DeBERTa-v3-base trained on each variety and tested on all
  three — a full 3×3 matrix (×2 seeds) of macro-F1 and minority-class F1.
- **Q2.3 — Parameter-efficient adaptation.** One Qwen2.5-3B Q-LoRA adapter per variety
  (4-bit NF4 base + rank-16 LoRA on attention projections), plus pooled / Reddit-only controls,
  a target-module/rank ablation, and validation-set threshold tuning.
- **Q3 — Evaluation.** A single registry-driven comparison across every model and regime.
- **Q4 — Error analysis & few-shot.** Ten errors from the best adapter, four annotated with
  linguistic explanations, an explanation-rich 4-shot prompt, tested on the held-out six.
- **Q5 — Deployment.** A task- and variety-aware Gradio service with a 13-route registry,
  PeftModel adapter swapping, a CPU classical fallback, and latency benchmarks.

---

## Repository layout

```
deployment_app.py       Gradio deployment service (registry, routing, lazy loaders)
src/                    Importable pipeline package
  config.py             constants, paths, seeds, encoder registry + naming helpers
  data.py               HuggingFace dataset loading & reporting
  metrics.py            metrics, confusion-matrix plotting, VRAM cleanup
  classical.py          TF-IDF / GloVe baselines
  transformers_ft.py    encoder fine-tuning + threshold tuning
  qlora.py              Qwen2.5-3B Q-LoRA training / loading
  registry.py           master experiment registry (resume-from-cache)
scripts/                One entry point per section (run_eda, run_classical, ...)
  run_all.py            run every stage, in order
  staged/               optional extra experiments (run deliberately)
outputs/tables/         All result CSVs incl. the master registry
figures/                Figures + confusion_matrices/<group>/
predictions/            Cached validation probabilities (threshold-tuning resume)
```

`checkpoints/` (encoder checkpoints + Qwen adapters, ~37 GB) is **not** in the repo; it is
git-ignored. Every result is preserved in the registry CSV, so the analysis reproduces
without it.

---

## Setup

```bash
python3 -m venv nlpenv --prompt "nlp-besstie"
source nlpenv/bin/activate
pip install -r requirements.txt
```

Python 3.12. A CUDA GPU is required only for *training* and for the live Qwen/encoder
deployment backends; everything else (EDA, classical baselines, all summary tables and
figures, the CPU deployment fallback) runs on CPU.

## Running

Each script reproduces one section of the analysis and **resumes from cache by default** —
heavy stages skip automatically when their rows are already in the master registry:

```bash
python scripts/run_all.py            # every stage, in order
python scripts/run_evaluation.py     # just rebuild the comparison tables (CPU, no data download)
python scripts/run_eda.py            # regenerate the EDA figures
```

By default `RUN_TRAINING=0`: the scripts rebuild tables and figures from the cached registry
and never touch a GPU or re-train. To re-enable the (still resumable) train/evaluate loops
for any missing rows:

```bash
RUN_TRAINING=1 python scripts/run_qlora.py
```

## Optional staged experiments

[`scripts/staged/`](scripts/staged/) holds extra experiments to run deliberately: a
second-seed replication of the cross-variety result (a training job — `RUN_TRAINING=1`),
and revised Q4 few-shot prompts (inference only — `python scripts/staged/q4_prompt_v2.py`).

---

## Dataset

[`surrey-nlp/BESSTIE-CW-26`](https://huggingface.co/datasets/surrey-nlp/BESSTIE-CW-26) —
loaded directly from the Hugging Face Hub; no data files are stored in the repo.
