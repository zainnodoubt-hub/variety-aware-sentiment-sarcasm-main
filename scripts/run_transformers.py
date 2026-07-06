"""Section 4 - Pre-trained encoder transformers (Q2.1, part 2).

Pooled fine-tuning + Reddit-only domain ablation. Resumable: completed rows are
skipped, so on a populated project this rebuilds the encoder summary table, the
training-curve figure, and the Q2.1 headline from cached results - no GPU needed.
Set RUN_EXTENDED_ENCODER_MODELS=1 to include DeBERTa-v3-large / RoBERTa-large.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import _bootstrap
from _bootstrap import show
from src import config
from src.config import (
    TEXT_COL, SOURCE_COL, SARCASM_COL, SEEDS, FIGURES_DIR, device,
    TRANSFORMER_MODEL_CONFIGS, selected_transformer_model_keys,
    is_transformer_model_key, is_transformer_crossvariety_key,
    transformer_checkpoint_dir, transformer_cm_filename, transformer_history_path,
)
from src.metrics import get_task_info, evaluate_predictions, plot_confusion_matrix, clear_vram
from src.transformers_ft import TextClassificationDataset, train_transformer_model, evaluate_transformer_model
from src.registry import load_master_results, is_experiment_done, append_result_row, TABLES_DIR


def _train_and_log(model_key, cfg, train_df, val_df, eval_df, task_name, label_col, label_names,
                   seed, scope, trained_on, validated_on, tested_on, notes, extra_row=None):
    model = tokenizer = None
    try:
        model, tokenizer, history_df = train_transformer_model(
            model_name=cfg["hf_id"], train_df=train_df, val_df=val_df,
            text_col=TEXT_COL, label_col=label_col, task_name=task_name, seed=seed,
            max_length=cfg["max_length"], batch_size=cfg["batch_size"], eval_batch_size=cfg["eval_batch_size"],
            grad_accum_steps=cfg["grad_accum_steps"], learning_rate=cfg["learning_rate"],
            epochs=cfg["epochs"], patience=cfg["patience"],
            use_class_weights=(task_name == "sarcasm"), device=device,
        )
        checkpoint_dir = transformer_checkpoint_dir(model_key, task_name, seed, scope=scope)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint_dir); tokenizer.save_pretrained(checkpoint_dir)
        test_ds = TextClassificationDataset(eval_df[TEXT_COL], eval_df[label_col], tokenizer, cfg["max_length"])
        loader = DataLoader(test_ds, batch_size=cfg["eval_batch_size"], shuffle=False)
        y_true, y_pred, _ = evaluate_transformer_model(model, loader, device)
        metrics_full, _, cm = evaluate_predictions(y_true, y_pred, label_names)
        plot_confusion_matrix(cm, label_names, f'{cfg["display_name"]} - {task_name} (seed={seed})',
                              filename=transformer_cm_filename(model_key, task_name, seed, scope=scope), show=False)
        row = {
            "model_type": cfg["family"], "model_key": model_key, "model_name": cfg["display_name"],
            "task": task_name, "seed": seed, "trained_on": trained_on, "validated_on": validated_on,
            "tested_on": tested_on, "val_macro_f1": history_df["val_macro_f1"].max(),
            "test_macro_f1": metrics_full["macro_f1"], "test_macro_precision": metrics_full["macro_precision"],
            "test_macro_recall": metrics_full["macro_recall"],
            f"test_f1_{label_names[0]}": metrics_full.get(f"f1_{label_names[0]}", 0),
            f"test_f1_{label_names[1]}": metrics_full.get(f"f1_{label_names[1]}", 0),
            "epochs_trained": len(history_df), "trainable_params_pct": 100.0,
            "adapter_path": str(checkpoint_dir), "notes": notes,
        }
        if extra_row:
            row.update(extra_row)
        append_result_row(row)
        history_df.to_csv(transformer_history_path(model_key, task_name, seed, scope=scope), index=False)
    finally:
        del model, tokenizer
        clear_vram(verbose=True)


def main() -> None:
    df, train_df, val_df, test_df = _bootstrap.load_all_splits()
    encoder_keys = selected_transformer_model_keys()
    print(f"Pooled encoder transformers (idempotent): {encoder_keys}")

    # 4.1 Pooled fine-tuning
    for model_key in encoder_keys:
        cfg = TRANSFORMER_MODEL_CONFIGS[model_key]
        for task_name in ["sentiment", "sarcasm"]:
            label_col, label_names = get_task_info(task_name)
            for seed in SEEDS:
                if is_experiment_done(model_key, task_name, "pooled_all_varieties", "pooled_all_varieties", "pooled_all_varieties", seed):
                    print(f"  [SKIP] {model_key} | {task_name} | seed={seed} (already done)")
                    continue
                if not _bootstrap.training_enabled():
                    print(f"  [SKIP-NO-TRAIN] {model_key} | {task_name} | seed={seed} (RUN_TRAINING=0)")
                    continue
                print(f'\n  [RUN] {cfg["display_name"]} | task={task_name}, seed={seed}')
                _train_and_log(model_key, cfg, train_df, val_df, test_df, task_name, label_col, label_names,
                               seed, "pooled_all_varieties", "pooled_all_varieties", "pooled_all_varieties",
                               "pooled_all_varieties", f'hf_id={cfg["hf_id"]}')

    # 4.2 Reddit-only domain ablation
    reddit_df = df[df[SOURCE_COL].astype(str).str.lower() == "reddit"].copy()
    reddit_train = reddit_df[reddit_df["split"] == "train"].reset_index(drop=True)
    reddit_val = reddit_df[reddit_df["split"] == "validation"].reset_index(drop=True)
    reddit_test = reddit_df[reddit_df["split"] == "test"].reset_index(drop=True)
    print(f"\nReddit-only: train={len(reddit_train)}, val={len(reddit_val)}, test={len(reddit_test)}")
    if len(reddit_train) > 50 and len(reddit_val) > 10 and len(reddit_test) > 10:
        for base_key in encoder_keys:
            cfg = TRANSFORMER_MODEL_CONFIGS[base_key]
            model_key = f"{base_key}_reddit"
            for seed in SEEDS:
                if is_experiment_done(model_key, "sarcasm", "reddit_only", "reddit_only", "reddit_only", seed):
                    print(f"  [SKIP] {model_key} | sarcasm | seed={seed} (already done)")
                    continue
                if not _bootstrap.training_enabled():
                    print(f"  [SKIP-NO-TRAIN] {model_key} | sarcasm | seed={seed} (RUN_TRAINING=0)")
                    continue
                print(f'\n  [RUN] Reddit {cfg["display_name"]}, seed={seed}')
                _train_and_log(model_key, cfg, reddit_train, reddit_val, reddit_test, "sarcasm",
                               SARCASM_COL, ["Not Sarcastic", "Sarcastic"], seed, "reddit_only",
                               "reddit_only", "reddit_only", "reddit_only", f'hf_id={cfg["hf_id"]}; reddit_only')

    # Rebuild summary table from registry
    master = load_master_results()
    mask = master["model_key"].astype(str).apply(is_transformer_model_key)
    non_cross = ~master["model_key"].astype(str).apply(is_transformer_crossvariety_key)
    transformer_df = master[mask & non_cross].copy()
    transformer_df.to_csv(TABLES_DIR / "04_transformer_results.csv", index=False)
    print(f"\nEncoder pooled/reddit rows available: {len(transformer_df)}")

    if len(transformer_df):
        print("\nEncoder Transformer Results Summary:")
        show(transformer_df.groupby(["model_name", "model_key", "task", "trained_on", "tested_on"], dropna=False)
             .agg({"val_macro_f1": ["mean", "std"], "test_macro_f1": ["mean", "std"],
                   "test_macro_precision": ["mean"], "test_macro_recall": ["mean"], "seed": ["nunique"]}).round(4))

    # Training-curve figure from cached history CSVs
    history_files = sorted(glob.glob(str(TABLES_DIR / "04_*history*.csv")))
    if history_files:
        n_cols = min(3, len(history_files)); n_rows = int(np.ceil(len(history_files) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
        axes_flat = axes.ravel()
        for ax, file in zip(axes_flat, history_files):
            h = pd.read_csv(file)
            ax.plot(h["epoch"], h["val_macro_f1"], "o-", label="Val Macro-F1")
            ax.set_xlabel("Epoch"); ax.set_ylabel("Macro-F1")
            ax.set_title(Path(file).stem.replace("04_", "").replace("_history_", "\n"), fontsize=9)
            ax.legend(); ax.grid(alpha=0.3)
        for ax in axes_flat[len(history_files):]:
            ax.axis("off")
        plt.tight_layout(); plt.savefig(FIGURES_DIR / "04_encoder_training_curves.png", dpi=300); plt.close()
        print("Saved figure: 04_encoder_training_curves.png")

    # Q2.1 headline (classical vs transformer gap on sarcasm)
    from src.registry import MASTER_RESULTS_PATH
    m = pd.read_csv(MASTER_RESULTS_PATH)
    q21 = m[m["model_key"].astype(str).isin(["tfidf_logreg", "roberta-large"]) & m["task"].eq("sarcasm")
            & m["trained_on"].eq("pooled_all_varieties") & m["tested_on"].eq("pooled_all_varieties")].copy()
    if len(q21):
        q21_summary = (q21.groupby(["model_key", "model_name"], as_index=False)
                       .agg(seeds=("seed", "nunique"), macro_f1_mean=("test_macro_f1", "mean"),
                            macro_f1_std=("test_macro_f1", "std"), sarcastic_f1_mean=("test_f1_Sarcastic", "mean")).round(4))
        q21_summary["family"] = q21_summary["model_key"].map({"tfidf_logreg": "Classical", "roberta-large": "Transformer"})
        print("\nQ2.1 summary - Sarcasm Macro-F1 gap (pooled train -> pooled test)")
        show(q21_summary[["family", "model_name", "seeds", "macro_f1_mean", "macro_f1_std", "sarcastic_f1_mean"]])


if __name__ == "__main__":
    main()
