"""Project-wide configuration: seeds, dataset constants, paths, and the
encoder-transformer model registry plus its naming/checkpoint helpers.

Importing this module is side-effect light. Call ``ensure_directories()`` from a
script before writing any outputs, and ``configure_display()`` to widen pandas
output.
"""
from __future__ import annotations

import os
import re
import random
from pathlib import Path

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Reproducibility and run flags
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def env_flag(name: str, default: str = "0") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------- #
# Dataset constants
# --------------------------------------------------------------------------- #
DATASET_NAME = "surrey-nlp/BESSTIE-CW-26"
TEXT_COL = "text"
VARIETY_COL = "variety"
SOURCE_COL = "source"
SENTIMENT_COL = "Sentiment"
SARCASM_COL = "Sarcasm"

VARIETIES = ["en-AU", "en-IN", "en-UK"]
SOURCES = ["Google", "Reddit"]

SENTIMENT_LABELS = {0: "Negative", 1: "Positive"}
SARCASM_LABELS = {0: "Not Sarcastic", 1: "Sarcastic"}
SEEDS = [42, 123]


# --------------------------------------------------------------------------- #
# Directories (anchored at the repo root, not the current working directory)
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = PROJECT_ROOT / "figures"
CONFUSION_DIR = FIGURES_DIR / "confusion_matrices"
TABLES_DIR = OUTPUTS_DIR / "tables"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
ADAPTERS_DIR = CHECKPOINTS_DIR / "qwen_adapters"


# --------------------------------------------------------------------------- #
# Encoder-transformer fine-tuning candidates.
# The first key preserves the original DeBERTa-v3-base experiment.
# --------------------------------------------------------------------------- #
TRANSFORMER_MODEL_CONFIGS = {
    "deberta-v3-base": {
        "hf_id": "microsoft/deberta-v3-base",
        "display_name": "DeBERTa-v3-base",
        "family": "DeBERTa",
        "max_length": 128,
        "batch_size": 4,
        "eval_batch_size": 8,
        "grad_accum_steps": 4,
        "learning_rate": 2e-5,
        "epochs": 7,
        "patience": 2,
    },
    "deberta-v3-large": {
        "hf_id": "microsoft/deberta-v3-large",
        "display_name": "DeBERTa-v3-large",
        "family": "DeBERTa",
        "max_length": 128,
        "batch_size": 2,
        "eval_batch_size": 4,
        "grad_accum_steps": 8,
        "learning_rate": 1e-5,
        "epochs": 7,
        "patience": 2,
    },
    "roberta-large": {
        "hf_id": "FacebookAI/roberta-large",
        "display_name": "RoBERTa-large",
        "family": "RoBERTa",
        "max_length": 128,
        "batch_size": 2,
        "eval_batch_size": 4,
        "grad_accum_steps": 8,
        "learning_rate": 1e-5,
        "epochs": 7,
        "patience": 2,
    },
}
CORE_TRANSFORMER_MODEL_KEYS = ["deberta-v3-base"]
EXTENDED_TRANSFORMER_MODEL_KEYS = ["deberta-v3-large", "roberta-large"]
TRANSFORMER_KEY_SUFFIXES = ("_threshold_tuned", "_crossvariety", "_reddit")


def selected_transformer_model_keys() -> list[str]:
    keys = list(CORE_TRANSFORMER_MODEL_KEYS)
    if env_flag("RUN_EXTENDED_ENCODER_MODELS"):
        keys.extend(EXTENDED_TRANSFORMER_MODEL_KEYS)
    return keys


def selected_cross_variety_transformer_model_keys() -> list[str]:
    keys = list(CORE_TRANSFORMER_MODEL_KEYS)
    if env_flag("RUN_EXTENDED_ENCODER_CROSS_VARIETY"):
        keys.extend(EXTENDED_TRANSFORMER_MODEL_KEYS)
    return keys


def transformer_base_key(model_key: str) -> str:
    key = str(model_key)
    for suffix in TRANSFORMER_KEY_SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return key


def is_transformer_model_key(model_key: str) -> bool:
    return transformer_base_key(model_key) in TRANSFORMER_MODEL_CONFIGS


def is_plain_transformer_model_key(model_key: str) -> bool:
    key = str(model_key)
    return key in TRANSFORMER_MODEL_CONFIGS


def is_transformer_reddit_key(model_key: str) -> bool:
    key = str(model_key)
    return key.endswith("_reddit") and transformer_base_key(key) in TRANSFORMER_MODEL_CONFIGS


def is_transformer_crossvariety_key(model_key: str) -> bool:
    key = str(model_key)
    return key.endswith("_crossvariety") and transformer_base_key(key) in TRANSFORMER_MODEL_CONFIGS


def transformer_display_name(model_key: str) -> str:
    base_key = transformer_base_key(model_key)
    return TRANSFORMER_MODEL_CONFIGS.get(base_key, {}).get("display_name", base_key)


def transformer_family(model_key: str) -> str:
    base_key = transformer_base_key(model_key)
    return TRANSFORMER_MODEL_CONFIGS.get(base_key, {}).get("family", "Encoder Transformer")


def model_key_slug(model_key: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(model_key)).strip("_").lower()


def transformer_checkpoint_dir(model_key: str, task_name: str, seed: int, scope: str = "pooled_all_varieties", train_variety: str | None = None) -> Path:
    base_key = transformer_base_key(model_key)
    if base_key == "deberta-v3-base":
        if scope == "pooled_all_varieties":
            return CHECKPOINTS_DIR / f"deberta_{task_name}_seed{seed}"
        if scope == "reddit_only":
            return CHECKPOINTS_DIR / f"deberta_reddit_{task_name}_seed{seed}"
        if scope == "cross_variety" and train_variety is not None:
            return CHECKPOINTS_DIR / f"deberta_cross_{train_variety}_seed{seed}"
    scope_slug = model_key_slug(scope)
    if train_variety is not None:
        scope_slug = f"{scope_slug}_{model_key_slug(train_variety)}"
    return CHECKPOINTS_DIR / "encoder_transformers" / f"{model_key_slug(base_key)}_{task_name}_{scope_slug}_seed{seed}"


def transformer_confusion_subdir(model_key: str, scope: str) -> str:
    base_key = transformer_base_key(model_key)
    if scope == "cross_variety":
        return "cross_variety" if base_key == "deberta-v3-base" else "encoder_transformers"
    return "deberta" if base_key == "deberta-v3-base" else "encoder_transformers"


def transformer_cm_stem(model_key: str, task_name: str, seed: int, scope: str = "pooled_all_varieties", trained_on: str | None = None, tested_on: str | None = None) -> str:
    base_key = transformer_base_key(model_key)
    if base_key == "deberta-v3-base":
        if scope == "pooled_all_varieties":
            return f"04_cm_deberta_{task_name}_seed{seed}"
        if scope == "reddit_only":
            return f"04_cm_deberta_reddit_seed{seed}"
        if scope == "cross_variety":
            return f"05_cm_crossvariety_{trained_on}_to_{tested_on}_seed{seed}"
    slug = model_key_slug(base_key)
    if scope == "pooled_all_varieties":
        return f"04_cm_{slug}_{task_name}_seed{seed}"
    if scope == "reddit_only":
        return f"04_cm_{slug}_reddit_{task_name}_seed{seed}"
    if scope == "cross_variety":
        return f"05_cm_{slug}_crossvariety_{trained_on}_to_{tested_on}_seed{seed}"
    return f"cm_{slug}_{task_name}_{scope}_seed{seed}"


def transformer_cm_filename(model_key: str, task_name: str, seed: int, scope: str = "pooled_all_varieties", trained_on: str | None = None, tested_on: str | None = None) -> Path:
    subdir = transformer_confusion_subdir(model_key, scope)
    stem = transformer_cm_stem(model_key, task_name, seed, scope, trained_on, tested_on)
    return Path("confusion_matrices") / subdir / f"{stem}.png"


def transformer_cm_csv_path(model_key: str, task_name: str, seed: int, scope: str = "pooled_all_varieties", trained_on: str | None = None, tested_on: str | None = None) -> Path:
    subdir = transformer_confusion_subdir(model_key, scope)
    stem = transformer_cm_stem(model_key, task_name, seed, scope, trained_on, tested_on)
    return CONFUSION_DIR / subdir / f"{stem}.csv"


def transformer_history_path(model_key: str, task_name: str, seed: int, scope: str = "pooled_all_varieties") -> Path:
    base_key = transformer_base_key(model_key)
    if base_key == "deberta-v3-base" and scope == "pooled_all_varieties":
        return TABLES_DIR / f"04_deberta_history_{task_name}_seed{seed}.csv"
    return TABLES_DIR / f"04_{model_key_slug(base_key)}_{scope}_history_{task_name}_seed{seed}.csv"


# --------------------------------------------------------------------------- #
# Device + setup helpers
# --------------------------------------------------------------------------- #
device = "cuda" if torch.cuda.is_available() else "cpu"


def ensure_directories() -> None:
    """Create every output directory the pipeline writes to."""
    for directory in [OUTPUTS_DIR, FIGURES_DIR, CONFUSION_DIR, TABLES_DIR,
                      PREDICTIONS_DIR, CHECKPOINTS_DIR, ADAPTERS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    for subdir in ["classical", "deberta", "cross_variety", "qlora", "encoder_transformers"]:
        (CONFUSION_DIR / subdir).mkdir(parents=True, exist_ok=True)


def configure_display() -> None:
    import pandas as pd
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.max_colwidth", 120)


def print_environment() -> None:
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Device: {device}")
    print(f"Core encoder models: {CORE_TRANSFORMER_MODEL_KEYS}")
    print(f"Extended encoder models available: {EXTENDED_TRANSFORMER_MODEL_KEYS}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
