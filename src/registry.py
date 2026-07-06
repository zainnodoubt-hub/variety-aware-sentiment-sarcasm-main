"""Master experiment registry. A single CSV (``07_ALL_EXPERIMENTS_SUMMARY.csv``)
records every completed run so heavy training cells/scripts can skip work that is
already done and resume from cached results."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    TABLES_DIR, CONFUSION_DIR, FIGURES_DIR, CHECKPOINTS_DIR,
    VARIETIES, SEEDS, env_flag,
    selected_transformer_model_keys,
    is_transformer_model_key, is_transformer_crossvariety_key, is_transformer_reddit_key,
    transformer_cm_csv_path,
)

EXPERIMENT_KEY_COLS = ["model_key", "task", "trained_on", "validated_on", "tested_on", "seed"]
CLASSICAL_MODEL_KEYS = ["tfidf_logreg", "glove_svm"]
MASTER_RESULTS_PATH = TABLES_DIR / "07_ALL_EXPERIMENTS_SUMMARY.csv"
FINAL_COMPARISON_PATH = TABLES_DIR / "08_FINAL_COMPARISON.csv"
QWEN_REGISTRY_BASE_KEY = "qwen2.5-3b-qlora"

MASTER_BASE_COLUMNS = [
    "model_type", "model_key", "model_name", "task", "seed",
    "trained_on", "validated_on", "tested_on",
    "val_macro_f1", "test_macro_f1", "test_macro_precision", "test_macro_recall",
    "test_f1_Negative", "test_f1_Positive", "test_f1_Not Sarcastic", "test_f1_Sarcastic",
    "epochs_trained", "trainable_params_pct", "adapter_path", "threshold",
    "is_same_variety", "notes",
]


def make_experiment_id(row: dict) -> str:
    return "|".join(str(row.get(c, "")) for c in EXPERIMENT_KEY_COLS)


def empty_master_results() -> pd.DataFrame:
    return pd.DataFrame(columns=MASTER_BASE_COLUMNS + ["experiment_id"])


def _normalise_seed(value):
    if pd.isna(value):
        return value
    try:
        return int(value)
    except Exception:
        return value


def _ensure_master_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return empty_master_results()
    df = df.copy()
    for col in MASTER_BASE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    for col in EXPERIMENT_KEY_COLS:
        if col not in df.columns:
            df[col] = ""
    df["seed"] = df["seed"].apply(_normalise_seed)
    df["experiment_id"] = df.apply(lambda r: make_experiment_id(r.to_dict()), axis=1)
    return df


def _f1_from_confusion_matrix(cm: pd.DataFrame, label_names: list[str]) -> dict:
    values = cm.to_numpy(dtype=float)
    out = {}
    for idx, label in enumerate(label_names):
        tp = values[idx, idx]
        fp = values[:, idx].sum() - tp
        fn = values[idx, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        out[f"test_f1_{label}"] = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return out


def _cm_path_for_row(row: pd.Series) -> Path | None:
    model_key = str(row.get("model_key", ""))
    task = str(row.get("task", ""))
    seed = _normalise_seed(row.get("seed", ""))
    trained_on = str(row.get("trained_on", ""))
    tested_on = str(row.get("tested_on", ""))
    if model_key in CLASSICAL_MODEL_KEYS:
        return CONFUSION_DIR / "classical" / f"{task}_{model_key}_seed{seed}.csv"
    if model_key == "deberta-v3-base_threshold_tuned":
        return FIGURES_DIR / f"06_cm_deberta_threshold_seed{seed}.csv"
    if is_transformer_model_key(model_key):
        if is_transformer_crossvariety_key(model_key):
            return transformer_cm_csv_path(model_key, task, seed, scope="cross_variety", trained_on=trained_on, tested_on=tested_on)
        if is_transformer_reddit_key(model_key):
            return transformer_cm_csv_path(model_key, task, seed, scope="reddit_only")
        return transformer_cm_csv_path(model_key, task, seed, scope="pooled_all_varieties")
    if model_key == f"{QWEN_REGISTRY_BASE_KEY}_threshold_tuned":
        return CONFUSION_DIR / "qlora" / f"06_cm_qwen_threshold_{trained_on}_to_{tested_on}_seed{seed}.csv"
    if model_key.startswith(f"{QWEN_REGISTRY_BASE_KEY}_"):
        suffix = model_key.replace(f"{QWEN_REGISTRY_BASE_KEY}_", "")
        if "rank" in suffix and "bias_" in suffix:
            return CONFUSION_DIR / "qlora" / f"06_cm_qwen_{suffix}_{trained_on}_to_{tested_on}_seed{seed}.csv"
    if model_key.startswith(QWEN_REGISTRY_BASE_KEY):
        if trained_on == "pooled_all_varieties":
            return CONFUSION_DIR / "qlora" / f"06_cm_qwen_pooled_to_{tested_on}_seed{seed}.csv"
        if trained_on == "reddit_only":
            return CONFUSION_DIR / "qlora" / f"06_cm_qwen_reddit_to_{tested_on}_seed{seed}.csv"
        return CONFUSION_DIR / "qlora" / f"06_cm_qwen_{trained_on}_to_{tested_on}_seed{seed}.csv"
    return None


def backfill_per_class_f1(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_master_schema(df)
    for idx, row in df.iterrows():
        label_names = ["Negative", "Positive"] if row.get("task") == "sentiment" else ["Not Sarcastic", "Sarcastic"]
        cm_path = _cm_path_for_row(row)
        if cm_path is None or not cm_path.exists():
            continue
        cm_df = pd.read_csv(cm_path, index_col=0)
        f1_values = _f1_from_confusion_matrix(cm_df, label_names)
        for col, value in f1_values.items():
            df.at[idx, col] = value
    return _ensure_master_schema(df)


def _seed_from_legacy_tables() -> pd.DataFrame:
    return empty_master_results()


def load_master_results() -> pd.DataFrame:
    if MASTER_RESULTS_PATH.exists():
        df = pd.read_csv(MASTER_RESULTS_PATH)
        return _ensure_master_schema(df)
    df = _seed_from_legacy_tables()
    MASTER_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = _ensure_master_schema(df).drop_duplicates(EXPERIMENT_KEY_COLS, keep="last")
    df.to_csv(MASTER_RESULTS_PATH, index=False)
    return df


def is_experiment_done(model_key, task, trained_on, validated_on, tested_on, seed) -> bool:
    df = load_master_results()
    if len(df) == 0:
        return False
    query = {
        "model_key": model_key,
        "task": task,
        "trained_on": trained_on,
        "validated_on": validated_on,
        "tested_on": tested_on,
        "seed": _normalise_seed(seed),
    }
    exp_id = make_experiment_id(query)
    return exp_id in set(df["experiment_id"].astype(str))


def append_result_row(row: dict) -> None:
    row = dict(row)
    for col in EXPERIMENT_KEY_COLS:
        if col not in row:
            raise ValueError(f"Missing experiment key column: {col}")
    row["seed"] = _normalise_seed(row["seed"])
    df = load_master_results()
    row_df = _ensure_master_schema(pd.DataFrame([row]))
    exp_id = row_df.iloc[0]["experiment_id"]
    if len(df) and exp_id in set(df["experiment_id"].astype(str)):
        print(f"  [WARN] Replacing duplicate experiment row: {exp_id}")
    out = pd.concat([df, row_df], ignore_index=True, sort=False)
    out = _ensure_master_schema(out).drop_duplicates(EXPERIMENT_KEY_COLS, keep="last")
    out.to_csv(MASTER_RESULTS_PATH, index=False)


def print_resume_status() -> None:
    df = load_master_results()
    print(f"Master registry: {MASTER_RESULTS_PATH}")
    print(f"Unique experiments complete: {len(df)}")
    if len(df) and "model_type" in df.columns:
        status_df = df.groupby("model_type")["experiment_id"].nunique().reset_index(name="completed").sort_values("completed", ascending=False)
        print(status_df.to_string(index=False))

    print(f"Encoder models selected for pooled/reddit runs: {selected_transformer_model_keys()}")
    if not env_flag("RUN_EXTENDED_ENCODER_MODELS"):
        print("Extended encoder models are available but disabled for safe Run All. Set RUN_EXTENDED_ENCODER_MODELS=1 to add them.")

    expected_qwen = []
    for train_variety in VARIETIES:
        for seed in SEEDS:
            for test_variety in VARIETIES:
                expected_qwen.append((QWEN_REGISTRY_BASE_KEY, "sarcasm", train_variety, train_variety, test_variety, seed))
    missing = [x for x in expected_qwen if not is_experiment_done(*x)]
    print(f"Expected Qwen variety-specific rows missing: {len(missing)} / {len(expected_qwen)}")
    for item in missing[:12]:
        print(f"  missing: {item[2]} -> {item[4]} seed={item[5]}")
    if len(missing) > 12:
        print(f"  ... {len(missing) - 12} more")
