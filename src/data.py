"""Dataset loading and basic reporting for BESSTIE-CW-26."""
from __future__ import annotations

import pandas as pd
from datasets import load_dataset

from .config import (
    TEXT_COL, VARIETY_COL, SOURCE_COL, SENTIMENT_COL, SARCASM_COL,
)


def load_besstie_dataset(dataset_name):
    return load_dataset(dataset_name)


def dataset_to_dataframe(dataset) -> pd.DataFrame:
    rows = []
    for split_name in dataset.keys():
        for example in dataset[split_name]:
            rows.append({
                TEXT_COL: example.get(TEXT_COL, ""),
                VARIETY_COL: example.get(VARIETY_COL, ""),
                SOURCE_COL: example.get(SOURCE_COL, ""),
                SENTIMENT_COL: example.get(SENTIMENT_COL, -1),
                SARCASM_COL: example.get(SARCASM_COL, -1),
                "split": split_name,
            })
    return pd.DataFrame(rows)


def get_split_dataframes(df: pd.DataFrame):
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "validation"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    return train_df, val_df, test_df


def basic_dataset_report(df: pd.DataFrame) -> dict:
    return {
        "num_rows": len(df),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": df.duplicated().sum(),
        "split_counts": df["split"].value_counts().to_dict(),
        "variety_counts": df[VARIETY_COL].value_counts().to_dict(),
        "source_counts": df[SOURCE_COL].value_counts().to_dict(),
    }


def check_expected_columns(df: pd.DataFrame) -> None:
    expected_cols = [TEXT_COL, VARIETY_COL, SOURCE_COL, SENTIMENT_COL, SARCASM_COL, "split"]
    missing = [col for col in expected_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    print(f"All expected columns present: {expected_cols}")


def load_dataframe(dataset_name) -> pd.DataFrame:
    """Convenience: load the HF dataset and return a single validated dataframe."""
    dataset = load_besstie_dataset(dataset_name)
    df = dataset_to_dataframe(dataset)
    check_expected_columns(df)
    return df
