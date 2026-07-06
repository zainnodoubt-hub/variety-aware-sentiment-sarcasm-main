"""Section 7 - Evaluation. Aggregates the master registry into the grouped
registry summary and the final headline comparison table (08_FINAL_COMPARISON).
Pure CSV/registry work - no models loaded."""
from __future__ import annotations

import numpy as np
import pandas as pd

import _bootstrap
from _bootstrap import show
from src.config import (
    VARIETIES, is_plain_transformer_model_key, is_transformer_crossvariety_key,
)
from src.registry import (
    backfill_per_class_f1, load_master_results, _ensure_master_schema,
    EXPERIMENT_KEY_COLS, MASTER_RESULTS_PATH, FINAL_COMPARISON_PATH, QWEN_REGISTRY_BASE_KEY,
)

QWEN_MODEL_KEY = QWEN_REGISTRY_BASE_KEY
HEADLINE_MODEL_KEYS = {
    "tfidf_logreg", "glove_svm", "roberta-large", "deberta-v3-large",
    "deberta-v3-base_crossvariety", "qwen2.5-3b-qlora",
}
HEADLINE_ABLATION_KEYS = {
    "qwen2.5-3b-qlora_attention_mlp_rank16_bias_none",
    "qwen2.5-3b-qlora_threshold_tuned",
    "deberta-v3-base_threshold_tuned",
}
DOMAIN_ABLATION_KEY = "roberta-large_reddit"


def _restrict(df, allowed):
    if df is None or len(df) == 0:
        return df
    return df[df["model_key"].astype(str).isin(allowed)].copy()


def summarise_for_comparison(df, label):
    if df is None or len(df) == 0:
        return pd.DataFrame()
    work = df.copy()
    work["comparison_group"] = label
    for col in ["test_macro_f1", "test_macro_precision", "test_macro_recall"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work["minority_class_f1"] = np.where(
        work["task"].eq("sarcasm"),
        pd.to_numeric(work["test_f1_Sarcastic"], errors="coerce"),
        pd.to_numeric(work["test_f1_Positive"], errors="coerce"))
    return (work.groupby(["comparison_group", "model_type", "model_key", "model_name", "task", "trained_on", "tested_on"], dropna=False)
            .agg(seeds=("seed", "nunique"), test_macro_f1_mean=("test_macro_f1", "mean"),
                 test_macro_f1_std=("test_macro_f1", "std"), test_macro_precision_mean=("test_macro_precision", "mean"),
                 test_macro_recall_mean=("test_macro_recall", "mean"), minority_class_f1_mean=("minority_class_f1", "mean"))
            .reset_index())


def main() -> None:
    all_results = backfill_per_class_f1(load_master_results())
    all_results = _ensure_master_schema(all_results).drop_duplicates(EXPERIMENT_KEY_COLS, keep="last")
    all_results.to_csv(MASTER_RESULTS_PATH, index=False)
    print(f"Master registry loaded: {MASTER_RESULTS_PATH}\nUnique experiments: {len(all_results)}")
    if len(all_results) == 0:
        print("No experiments in the master registry yet.")
        return

    for col in ["test_macro_f1", "test_macro_precision", "test_macro_recall"]:
        all_results[col] = pd.to_numeric(all_results[col], errors="coerce")
    all_results["minority_class_f1"] = np.where(
        all_results["task"].eq("sarcasm"),
        pd.to_numeric(all_results["test_f1_Sarcastic"], errors="coerce"),
        pd.to_numeric(all_results["test_f1_Positive"], errors="coerce"))
    registry_summary = (all_results.groupby(["model_type", "task", "trained_on", "tested_on"], dropna=False)
                        .agg(seeds=("seed", "nunique"), test_macro_f1_mean=("test_macro_f1", "mean"),
                             test_macro_f1_std=("test_macro_f1", "std"),
                             test_macro_precision_mean=("test_macro_precision", "mean"),
                             test_macro_recall_mean=("test_macro_recall", "mean"),
                             minority_class_f1_mean=("minority_class_f1", "mean")).reset_index().round(4))
    print("\nRegistry summary (head 30):")
    show(registry_summary.head(30))

    model_keys = all_results["model_key"].astype(str)
    parts = []
    classical = all_results[all_results["model_type"].eq("Classical ML")
                            & all_results["trained_on"].eq("pooled_all_varieties")
                            & all_results["tested_on"].eq("pooled_all_varieties")].copy()
    if len(classical):
        parts.append(summarise_for_comparison(_restrict(classical, HEADLINE_MODEL_KEYS), "Classical ML (headline)"))
    plain_mask = model_keys.apply(is_plain_transformer_model_key)
    parts.append(summarise_for_comparison(_restrict(all_results[plain_mask
        & all_results["trained_on"].eq("pooled_all_varieties") & all_results["tested_on"].eq("pooled_all_varieties")],
        HEADLINE_MODEL_KEYS), "Encoder transformers pooled"))
    cross_mask = model_keys.apply(is_transformer_crossvariety_key)
    parts.append(summarise_for_comparison(_restrict(all_results[cross_mask
        & all_results["trained_on"].eq(all_results["tested_on"])], HEADLINE_MODEL_KEYS),
        "Encoder transformers per-variety diagonal"))
    parts.append(summarise_for_comparison(all_results[all_results["model_key"].eq(QWEN_MODEL_KEY)
        & all_results["trained_on"].isin(VARIETIES) & all_results["trained_on"].eq(all_results["tested_on"])],
        "Qwen2.5-3B QLoRA per-variety"))
    parts.append(summarise_for_comparison(all_results[all_results["model_key"].eq(QWEN_MODEL_KEY)
        & all_results["trained_on"].eq("pooled_all_varieties")], "Qwen2.5-3B QLoRA pooled"))
    lora_sweep_mask = model_keys.str.startswith(f"{QWEN_MODEL_KEY}_") & model_keys.str.contains("rank", na=False)
    parts.append(summarise_for_comparison(_restrict(all_results[lora_sweep_mask], HEADLINE_ABLATION_KEYS),
        "Qwen2.5-3B QLoRA target/rank sweep (kept ablation)"))
    parts.append(summarise_for_comparison(all_results[model_keys.str.contains("threshold_tuned", na=False)],
        "Threshold-tuned variants"))
    parts.append(summarise_for_comparison(all_results[model_keys.eq(DOMAIN_ABLATION_KEY)],
        "Domain ablation (Reddit-only training)"))

    final_df = pd.concat([x for x in parts if x is not None and len(x)], ignore_index=True) if parts else pd.DataFrame()
    if not len(final_df):
        print("No rows available for final comparison yet.")
        return
    order = [
        "Classical ML (headline)", "Encoder transformers pooled",
        "Encoder transformers per-variety diagonal", "Qwen2.5-3B QLoRA per-variety",
        "Qwen2.5-3B QLoRA pooled", "Qwen2.5-3B QLoRA target/rank sweep (kept ablation)",
        "Threshold-tuned variants", "Domain ablation (Reddit-only training)",
    ]
    final_df["comparison_group"] = pd.Categorical(final_df["comparison_group"], categories=order, ordered=True)
    final_df = final_df.sort_values(["comparison_group", "task", "trained_on", "tested_on", "model_key"]).reset_index(drop=True)
    final_df["best_marker"] = ""
    for _, task_df in final_df.groupby("task", dropna=False):
        if len(task_df) and task_df["test_macro_f1_mean"].notna().any():
            final_df.loc[task_df["test_macro_f1_mean"].idxmax(), "best_marker"] = "BEST"
    final_df = final_df.round(4)
    final_df.to_csv(FINAL_COMPARISON_PATH, index=False)
    print(f"\nFull final comparison saved to: {FINAL_COMPARISON_PATH}")
    compact = [c for c in ["best_marker", "comparison_group", "task", "model_name", "trained_on",
                           "tested_on", "seeds", "test_macro_f1_mean", "test_macro_f1_std", "minority_class_f1_mean"]
               if c in final_df.columns]
    print("Compact report view:")
    show(final_df[compact])


if __name__ == "__main__":
    main()
