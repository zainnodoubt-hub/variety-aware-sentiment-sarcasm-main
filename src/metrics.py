"""Evaluation metrics, confusion-matrix plotting/saving, and VRAM cleanup."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

from .config import FIGURES_DIR, CONFUSION_DIR, SENTIMENT_COL, SARCASM_COL


def evaluate_predictions(y_true, y_pred, label_names):
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    report = classification_report(y_true, y_pred, target_names=label_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    metrics_dict = {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }

    for i, label in enumerate(label_names):
        # classification_report keys by target_names when provided; keep integer fallback for safety.
        report_key = label if label in report else str(i)
        metrics_dict[f"f1_{label}"] = report.get(report_key, {}).get("f1-score", 0)

    return metrics_dict, pd.DataFrame(report).T, cm


def get_task_info(task_name):
    if task_name == "sentiment":
        return SENTIMENT_COL, ["Negative", "Positive"]
    elif task_name == "sarcasm":
        return SARCASM_COL, ["Not Sarcastic", "Sarcastic"]
    else:
        raise ValueError(f"Unknown task: {task_name}")


def plot_confusion_matrix(cm, label_names, title, filename=None, show=True):
    """Plot a confusion matrix and, when ``filename`` is an image, also save the
    matching CSV. Relative bare filenames are routed to the right
    ``confusion_matrices/<group>/`` subdirectory based on their prefix."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    save_path = None
    if filename:
        filename = Path(filename)
        if not filename.is_absolute() and filename.parent == Path("."):
            group = None
            if filename.name.startswith("03_cm_"):
                group = "classical"
            elif filename.name.startswith("04_cm_deberta"):
                group = "deberta"
            elif filename.name.startswith("05_cm_crossvariety"):
                group = "cross_variety"
            elif filename.name.startswith("06_cm_qlora"):
                group = "qlora"
            if group:
                filename = Path("confusion_matrices") / group / filename
        save_path = filename if filename.is_absolute() else FIGURES_DIR / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        if save_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf"}:
            pd.DataFrame(cm, index=label_names, columns=label_names).to_csv(save_path.with_suffix(".csv"))
    if show:
        plt.show()
    else:
        plt.close()
    return save_path


def save_confusion_matrix_csv(cm, label_names, filename):
    filename = Path(filename)
    save_path = filename if filename.is_absolute() else CONFUSION_DIR / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cm, index=label_names, columns=label_names).to_csv(save_path)
    return save_path


def clear_vram(verbose: bool = False) -> None:
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        if verbose:
            free_mb = torch.cuda.mem_get_info()[0] / 1024 ** 2
            total_mb = torch.cuda.mem_get_info()[1] / 1024 ** 2
            used_mb = total_mb - free_mb
            print(f"    [VRAM] used: {used_mb:.0f} MB, free: {free_mb:.0f} MB")
