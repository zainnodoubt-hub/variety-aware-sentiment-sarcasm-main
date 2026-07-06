"""Section 3 - Classical baselines (Q2.1, part 1): TF-IDF+LogReg and GloVe+SVM.

Resumable: every (model, task, seed) already in the master registry is skipped,
so on a populated project this only rebuilds the summary table. Classical fits
are CPU-only and fast; no transformer training happens here.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

import _bootstrap
from _bootstrap import show
from src.config import TEXT_COL, SEEDS, FIGURES_DIR
from src.metrics import get_task_info, evaluate_predictions, plot_confusion_matrix, save_confusion_matrix_csv
from src.classical import build_classical_model
from src.registry import (
    load_master_results, is_experiment_done, append_result_row, CONFUSION_DIR, TABLES_DIR,
)
import pandas as pd

CLASSICAL_MODELS = {
    "tfidf_logreg": "TF-IDF + LogReg",
    "glove_svm": "GloVe 100d + Linear SVM",
}
HEADLINE_CLASSICAL_KEYS = ["tfidf_logreg", "glove_svm"]


def main(render_headline_panels: bool = False) -> None:
    _, train_df, val_df, test_df = _bootstrap.load_all_splits()
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    print(f"Combined train+val: {len(train_val_df)} | test: {len(test_df)}")

    print("\nTraining classical models with idempotency checks...")
    for task_name in ["sentiment", "sarcasm"]:
        label_col, label_names = get_task_info(task_name)
        class_weight = "balanced" if task_name == "sarcasm" else None
        for model_key in CLASSICAL_MODELS:
            for seed in SEEDS:
                if is_experiment_done(model_key, task_name, "pooled_all_varieties", "pooled_all_varieties", "pooled_all_varieties", seed):
                    print(f"  [SKIP] {model_key} | {task_name} | seed={seed} (already done)")
                    continue
                print(f"  [RUN] {model_key} on {task_name} (seed={seed})")
                model = build_classical_model(model_key, class_weight=class_weight, seed=seed)
                model.fit(train_val_df[TEXT_COL], train_val_df[label_col])
                y_pred = model.predict(test_df[TEXT_COL])
                metrics_dict, _, cm = evaluate_predictions(test_df[label_col], y_pred, label_names)
                cm_stem = f"{task_name}_{model_key}_seed{seed}"
                plot_confusion_matrix(cm, label_names, f"{CLASSICAL_MODELS[model_key]} - {task_name} (seed={seed})",
                                      filename=Path("confusion_matrices") / "classical" / f"{cm_stem}.png", show=False)
                save_confusion_matrix_csv(cm, label_names, Path("classical") / f"{cm_stem}.csv")
                val_model = build_classical_model(model_key, class_weight=class_weight, seed=seed)
                val_model.fit(train_df[TEXT_COL], train_df[label_col])
                val_metrics, _, _ = evaluate_predictions(val_df[label_col], val_model.predict(val_df[TEXT_COL]), label_names)
                append_result_row({
                    "model_type": "Classical ML", "model_key": model_key, "model_name": CLASSICAL_MODELS[model_key],
                    "task": task_name, "seed": seed, "trained_on": "pooled_all_varieties",
                    "validated_on": "pooled_all_varieties", "tested_on": "pooled_all_varieties",
                    "val_macro_f1": val_metrics["macro_f1"], "test_macro_f1": metrics_dict["macro_f1"],
                    "test_macro_precision": metrics_dict["macro_precision"], "test_macro_recall": metrics_dict["macro_recall"],
                    f"test_f1_{label_names[0]}": metrics_dict.get(f"f1_{label_names[0]}", 0),
                    f"test_f1_{label_names[1]}": metrics_dict.get(f"f1_{label_names[1]}", 0),
                })

    classical_df = load_master_results()
    classical_df = classical_df[classical_df["model_type"].eq("Classical ML")].copy()
    classical_df.to_csv(TABLES_DIR / "03_classical_results.csv", index=False)
    print(f"\nClassical rows available: {len(classical_df)}")

    print("\nCLASSICAL HEADLINE SUMMARY (mean +/- std over seeds):")
    summary = (classical_df[classical_df["model_key"].isin(HEADLINE_CLASSICAL_KEYS)]
               .groupby(["task", "model_name"])
               .agg({"val_macro_f1": ["mean", "std"], "test_macro_f1": ["mean", "std"],
                     "test_macro_precision": ["mean"], "test_macro_recall": ["mean"]}).round(4))
    show(summary)

    if render_headline_panels:
        _render_headline_panels(train_val_df, test_df)
    else:
        print("\n(Headline CM panels skipped; pass --panels to refit classical models and re-render.)")


def _render_headline_panels(train_val_df, test_df) -> None:
    headline = {k: CLASSICAL_MODELS[k] for k in HEADLINE_CLASSICAL_KEYS}
    for task_name in ["sentiment", "sarcasm"]:
        label_col, label_names = get_task_info(task_name)
        class_weight = "balanced" if task_name == "sarcasm" else None
        fig, axes = plt.subplots(1, len(headline), figsize=(5 * len(headline), 4))
        for idx, (model_key, model_label) in enumerate(headline.items()):
            model = build_classical_model(model_key, class_weight=class_weight, seed=42)
            model.fit(train_val_df[TEXT_COL], train_val_df[label_col])
            _, _, cm = evaluate_predictions(test_df[label_col], model.predict(test_df[TEXT_COL]), label_names)
            plot_confusion_matrix(cm, label_names, f"{model_label} - {task_name}",
                                  filename=f"03_cm_{model_key}_{task_name}.png", show=False)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                        xticklabels=label_names, yticklabels=label_names, ax=axes[idx])
            axes[idx].set_title(f"{model_key}\n{task_name}", fontsize=10)
            axes[idx].set_xlabel("Predicted"); axes[idx].set_ylabel("True" if idx == 0 else "")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"03_cm_classical_{task_name}_headline.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  saved combined panel: 03_cm_classical_{task_name}_headline.png")


if __name__ == "__main__":
    import sys
    main(render_headline_panels="--panels" in sys.argv)
