"""Shared bootstrap for the entry scripts.

Importing this puts the repo root on ``sys.path`` (so ``import src...`` works when
a script is run directly), forces a non-interactive matplotlib backend, ensures
output directories exist, and provides a small data-loading convenience.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")  # scripts run head-less; save figures, never block on a window

from src import config  # noqa: E402

config.ensure_directories()
config.configure_display()


def load_all_splits():
    """Load BESSTIE-CW-26 and return (df, train_df, val_df, test_df)."""
    from src.data import load_dataframe, get_split_dataframes
    df = load_dataframe(config.DATASET_NAME)
    train_df, val_df, test_df = get_split_dataframes(df)
    print(f"Loaded dataset: {len(df)} rows | train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    return df, train_df, val_df, test_df


def training_enabled() -> bool:
    """GPU training/loading loops only run when RUN_TRAINING is truthy.

    Default is OFF: the heavy scripts then resume purely from the cached master
    registry and saved checkpoints, rebuilding tables and figures without
    touching a GPU or re-training anything. Set RUN_TRAINING=1 to re-enable the
    faithful resumable train/evaluate loops (they still skip completed rows).
    """
    return config.env_flag("RUN_TRAINING")


def show(obj) -> None:
    """Print a dataframe/series the way the notebook's display() would render it."""
    try:
        import pandas as pd
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            print(obj.to_string())
            return
    except Exception:
        pass
    print(obj)
