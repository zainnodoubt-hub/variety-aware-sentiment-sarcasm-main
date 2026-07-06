"""Section 6 - Qwen2.5-3B Q-LoRA variety adapters (Q2.3).

Variety-specific, pooled, and Reddit-only adapters, plus validation-set threshold
tuning and an optional LoRA target-module/rank ablation. Resumable: every adapter
and evaluation already in the master registry is skipped. GPU train/load loops only
run when RUN_TRAINING=1; by default this rebuilds the Q-LoRA summary, the Q2.3
headline table, and (when enabled) threshold-tuned rows from cached results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader

import _bootstrap
from _bootstrap import show, training_enabled
from src.config import (
    TEXT_COL, VARIETY_COL, SARCASM_COL, SEEDS, VARIETIES, FIGURES_DIR, ADAPTERS_DIR, device, env_flag,
)
from src.metrics import evaluate_predictions, plot_confusion_matrix, clear_vram
from src.transformers_ft import (
    TextClassificationDataset, evaluate_transformer_model, predict_transformer_probabilities,
    evaluate_threshold, get_validation_subset_for_scope, get_test_subset_for_scope, tune_threshold_for_saved_model,
)
from src.qlora import (
    LORA_MODEL, QWEN_MODEL_KEY, QWEN_MODEL_NAME, train_qlora_model, load_qlora_adapter,
)
from src.registry import load_master_results, is_experiment_done, append_result_row, TABLES_DIR


def _eval_adapter_on_subsets(model, tokenizer, eval_sets, missing, seed, trained_on, cm_fmt):
    """Evaluate a loaded adapter on the missing test subsets and append rows."""
    for tested_on, subset in eval_sets:
        if tested_on not in missing or len(subset) == 0:
            continue
        ds = TextClassificationDataset(subset[TEXT_COL], subset[SARCASM_COL], tokenizer, 128)
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        y_true, y_pred, _ = evaluate_transformer_model(model, loader, device)
        metrics_full, _, cm = evaluate_predictions(y_true, y_pred, ["Not Sarcastic", "Sarcastic"])
        plot_confusion_matrix(cm, ["Not Sarcastic", "Sarcastic"],
                              f"Qwen Q-LoRA {trained_on} -> {tested_on} (seed={seed})",
                              filename=cm_fmt(tested_on), show=False)
        append_result_row({
            "model_type": "Qwen Q-LoRA", "model_key": QWEN_MODEL_KEY, "model_name": QWEN_MODEL_NAME,
            "task": "sarcasm", "seed": seed, "trained_on": trained_on, "validated_on": trained_on,
            "tested_on": tested_on, "val_macro_f1": np.nan,
            "test_macro_f1": metrics_full["macro_f1"], "test_macro_precision": metrics_full["macro_precision"],
            "test_macro_recall": metrics_full["macro_recall"],
            "test_f1_Not Sarcastic": metrics_full.get("f1_Not Sarcastic", 0),
            "test_f1_Sarcastic": metrics_full.get("f1_Sarcastic", 0),
            "adapter_path": str(model_adapter_path(trained_on, seed)),
            "is_same_variety": tested_on == trained_on})


def model_adapter_path(trained_on, seed):
    if trained_on in VARIETIES:
        return ADAPTERS_DIR / f"qwen_{trained_on}_seed{seed}"
    if trained_on == "pooled_all_varieties":
        return ADAPTERS_DIR / f"qwen_pooled_seed{seed}"
    if trained_on == "reddit_only":
        return ADAPTERS_DIR / f"qwen_reddit_seed{seed}"
    return ADAPTERS_DIR / f"qwen_{trained_on}_seed{seed}"


def _run_adapter_group(train_df, val_df, eval_sets_fn, scopes, train_data_fn):
    """Generic resumable adapter loop for a set of (scope, seed) training jobs."""
    for scope in scopes:
        for seed in SEEDS:
            eval_sets = eval_sets_fn(scope, seed)
            missing = [name for name, subset in eval_sets if len(subset)
                       and not is_experiment_done(QWEN_MODEL_KEY, "sarcasm", scope, scope, name, seed)]
            if not missing:
                print(f"  [SKIP] {QWEN_MODEL_KEY} {scope} | seed={seed} (all evaluations done)")
                continue
            if not training_enabled():
                print(f"  [SKIP-NO-TRAIN] {QWEN_MODEL_KEY} {scope} | seed={seed} (RUN_TRAINING=0)")
                continue
            adapter_path = model_adapter_path(scope, seed)
            model = tokenizer = None
            try:
                if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
                    print(f"  [LOAD] Existing adapter: {adapter_path}")
                    model, tokenizer, _ = load_qlora_adapter(LORA_MODEL, adapter_path, num_labels=2)
                else:
                    print(f"  [RUN] Train Qwen adapter {scope}, seed={seed}")
                    t_df, v_df = train_data_fn(scope)
                    model, tokenizer, _, _ = train_qlora_model(
                        model_name=LORA_MODEL, train_df=t_df, val_df=v_df, text_col=TEXT_COL,
                        label_col=SARCASM_COL, variety=scope, seed=seed, max_length=128, batch_size=2,
                        grad_accum_steps=4, learning_rate=1e-4, epochs=7, patience=2,
                        use_class_weights=True, device=device, adapter_output_dir=adapter_path)
                cm_fmt = (lambda tested_on, s=scope, sd=seed:
                          f"06_cm_qwen_{'pooled' if s == 'pooled_all_varieties' else ('reddit' if s == 'reddit_only' else s)}_to_{tested_on}_seed{sd}.png")
                _eval_adapter_on_subsets(model, tokenizer, eval_sets, missing, seed, scope, cm_fmt)
            finally:
                del model, tokenizer
                clear_vram(verbose=True)


def main() -> None:
    _, train_df, val_df, test_df = _bootstrap.load_all_splits()

    # 6.1 Variety-specific adapters
    print(f"Variety-specific Qwen Q-LoRA adapters with {LORA_MODEL}...")
    _run_adapter_group(
        train_df, val_df,
        eval_sets_fn=lambda scope, seed: [(tv, test_df[test_df[VARIETY_COL] == tv].reset_index(drop=True)) for tv in VARIETIES],
        scopes=VARIETIES,
        train_data_fn=lambda scope: (train_df[train_df[VARIETY_COL] == scope].reset_index(drop=True),
                                     val_df[val_df[VARIETY_COL] == scope].reset_index(drop=True)))

    # 6.2 Pooled adapter (control)
    print("\nPooled Qwen Q-LoRA adapter (control)...")
    _run_adapter_group(
        train_df, val_df,
        eval_sets_fn=lambda scope, seed: [("pooled_all_varieties", test_df)]
        + [(v, test_df[test_df[VARIETY_COL] == v].reset_index(drop=True)) for v in VARIETIES],
        scopes=["pooled_all_varieties"],
        train_data_fn=lambda scope: (train_df, val_df))

    # 6.3 Reddit-only adapter (domain control)
    from src.config import SOURCE_COL
    reddit_val = val_df[val_df[SOURCE_COL].astype(str).str.lower() == "reddit"].reset_index(drop=True)
    reddit_train = train_df[train_df[SOURCE_COL].astype(str).str.lower() == "reddit"].reset_index(drop=True)
    reddit_test = test_df[test_df[SOURCE_COL].astype(str).str.lower() == "reddit"].reset_index(drop=True)
    if len(reddit_train) > 50 and len(reddit_val) > 10 and len(reddit_test) > 10:
        print("\nReddit-only Qwen Q-LoRA adapter (domain control)...")
        _run_adapter_group(
            reddit_train, reddit_val,
            eval_sets_fn=lambda scope, seed: [("reddit_only", reddit_test)]
            + [(f"reddit_{v}", reddit_test[reddit_test[VARIETY_COL] == v].reset_index(drop=True)) for v in VARIETIES],
            scopes=["reddit_only"],
            train_data_fn=lambda scope: (reddit_train, reddit_val))

    # 6.4.1 Threshold tuning (resumable; loads a saved adapter only when needed)
    _threshold_tuning(val_df, test_df)

    # Rebuild Q-LoRA summary + Q2.3 headline from the registry
    master = load_master_results()
    qlora_df = master[master["model_key"].astype(str).str.startswith(QWEN_MODEL_KEY)].copy()
    qlora_df.to_csv(TABLES_DIR / "06_qlora_results.csv", index=False)
    print(f"\nQwen Q-LoRA rows available: {len(qlora_df)}")
    if len(qlora_df):
        print("\nQwen Q-LoRA Results Summary:")
        show(qlora_df.groupby(["model_key", "trained_on", "tested_on"]).agg({
            "val_macro_f1": ["mean"], "test_macro_f1": ["mean", "std"],
            "test_macro_precision": ["mean"], "test_macro_recall": ["mean"],
            "trainable_params_pct": ["mean"]}).round(4))

    _q23_headline()


def _threshold_tuning(val_df, test_df) -> None:
    master_df = load_master_results()
    qwen_default = master_df[(master_df["model_key"].eq(QWEN_MODEL_KEY)) & (master_df["task"].eq("sarcasm"))
                             & (master_df["trained_on"].isin(VARIETIES)) & (master_df["adapter_path"].notna())].copy()
    qwen_default["val_macro_f1"] = pd.to_numeric(qwen_default["val_macro_f1"], errors="coerce")
    qwen_default = qwen_default.dropna(subset=["val_macro_f1"])
    if len(qwen_default) == 0:
        print("  [SKIP] Qwen threshold tuning: no completed adapter with validation Macro-F1 yet")
        return
    best = qwen_default.sort_values("val_macro_f1", ascending=False).iloc[0]
    tuned_key = f"{QWEN_MODEL_KEY}_threshold_tuned"
    trained_on, tested_on, seed = best["trained_on"], best["tested_on"], int(best["seed"])
    if is_experiment_done(tuned_key, "sarcasm", trained_on, trained_on, tested_on, seed):
        print(f"  [SKIP] {tuned_key} | {trained_on}->{tested_on} | seed={seed} already done")
        return
    if not training_enabled():
        print(f"  [SKIP-NO-TRAIN] Qwen threshold tuning (RUN_TRAINING=0)")
        return
    adapter_path = Path(best["adapter_path"])
    if not (adapter_path / "adapter_config.json").exists():
        print(f"  [SKIP] Qwen threshold tuning: adapter missing at {adapter_path}")
        return
    model, tokenizer, param_info = load_qlora_adapter(LORA_MODEL, adapter_path, num_labels=2)
    val_subset = get_validation_subset_for_scope(val_df, trained_on)
    test_subset = get_test_subset_for_scope(test_df, tested_on)
    best_threshold, sweep_df = tune_threshold_for_saved_model(model, tokenizer, val_subset, f"qwen_threshold_{trained_on}_seed{seed}", device=device)
    test_ds = TextClassificationDataset(test_subset[TEXT_COL], test_subset[SARCASM_COL], tokenizer, 128)
    y_test, test_probs = predict_transformer_probabilities(model, DataLoader(test_ds, batch_size=8, shuffle=False), device)
    _, metrics_full, cm = evaluate_threshold(y_test, test_probs, best_threshold)
    plot_confusion_matrix(cm, ["Not Sarcastic", "Sarcastic"],
                          f"Qwen threshold tuned {trained_on}->{tested_on} (thr={best_threshold:.2f})",
                          filename=f"06_cm_qwen_threshold_{trained_on}_to_{tested_on}_seed{seed}.png", show=False)
    append_result_row({
        "model_type": "Threshold Tuned", "model_key": tuned_key, "model_name": QWEN_MODEL_NAME, "task": "sarcasm",
        "seed": seed, "trained_on": trained_on, "validated_on": trained_on, "tested_on": tested_on,
        "val_macro_f1": sweep_df["val_macro_f1"].max(), "test_macro_f1": metrics_full["macro_f1"],
        "test_macro_precision": metrics_full["macro_precision"], "test_macro_recall": metrics_full["macro_recall"],
        "test_f1_Not Sarcastic": metrics_full.get("f1_Not Sarcastic", 0),
        "test_f1_Sarcastic": metrics_full.get("f1_Sarcastic", 0),
        "trainable_params_pct": param_info.get("trainable_percentage", np.nan),
        "adapter_path": str(adapter_path), "threshold": best_threshold})
    del model, tokenizer
    clear_vram(verbose=True)
    print(f"  Added threshold-tuned row: {tuned_key} thr={best_threshold:.2f}")


def _q23_headline() -> None:
    from src.registry import MASTER_RESULTS_PATH
    m = pd.read_csv(MASTER_RESULTS_PATH)
    same = m[m["model_key"].eq(QWEN_MODEL_KEY) & m["task"].eq("sarcasm")
            & m["trained_on"].isin(VARIETIES) & (m["trained_on"] == m["tested_on"])].copy()
    pool = m[m["model_key"].eq(QWEN_MODEL_KEY) & m["task"].eq("sarcasm")
            & m["trained_on"].eq("pooled_all_varieties") & m["tested_on"].isin(VARIETIES)].copy()
    if not len(same) and not len(pool):
        return
    same.insert(0, "regime", "Same-variety adapter")
    pool.insert(0, "regime", "Pooled adapter")
    allrows = pd.concat([same, pool], ignore_index=True)
    summary = (allrows.groupby(["regime", "tested_on"], as_index=False)
               .agg(seeds=("seed", "nunique"), macro_f1_mean=("test_macro_f1", "mean"),
                    macro_f1_std=("test_macro_f1", "std"), sarcastic_f1_mean=("test_f1_Sarcastic", "mean")).round(4))
    print("\nQ2.3 summary - Qwen Q-LoRA: same-variety vs pooled")
    show(summary)


if __name__ == "__main__":
    main()
