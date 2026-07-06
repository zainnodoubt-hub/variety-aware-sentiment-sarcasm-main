"""Section 8 - Error analysis and explanation-rich few-shot prompt (Q4).

Default mode reads the cached Q4 artifacts (10 errors, 4 explained examples, the
4-shot prompt, and the 6-example prompt results) and prints the recovery summary.
The live Qwen extraction/prompting paths only run when RUN_TRAINING=1 (they load
the 4-bit Qwen base + best adapter on a GPU).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

import _bootstrap
from _bootstrap import show, training_enabled
from src.config import TEXT_COL, VARIETY_COL, SOURCE_COL, SARCASM_COL, VARIETIES, device
from src.metrics import clear_vram
from src.registry import load_master_results, TABLES_DIR

ERROR_ANALYSIS_DIR = TABLES_DIR / "08_sarcasm_error_analysis"
ERROR_CASES_PATH = ERROR_ANALYSIS_DIR / "qwen_best_adapter_10_errors.csv"
EXPLAINED_EXAMPLES_PATH = ERROR_ANALYSIS_DIR / "qwen_explained_4_examples.csv"
FEWSHOT_EVAL_SET_PATH = ERROR_ANALYSIS_DIR / "qwen_fewshot_remaining_6_examples.csv"
FEWSHOT_PROMPT_PATH = ERROR_ANALYSIS_DIR / "qwen_sarcasm_4shot_explanation_prompt.txt"
FEWSHOT_RESULTS_PATH = ERROR_ANALYSIS_DIR / "qwen_fewshot_remaining_6_results.csv"
SARCASM_ID_TO_LABEL = {0: "Not Sarcastic", 1: "Sarcastic"}


def main() -> None:
    ERROR_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # Q4.1 - best adapter row (informational) + cached 10 errors
    master = load_master_results()
    try:
        best = _select_best_qwen_adapter_row(master)
        print("Best Qwen/Q-LoRA row selected for Q4 error analysis:")
        show(best[["model_key", "trained_on", "tested_on", "seed", "val_macro_f1", "test_macro_f1", "adapter_path"]])
    except ValueError as e:
        print(f"  (Could not select best adapter row: {e})")

    if not ERROR_CASES_PATH.exists():
        if not training_enabled():
            print("No cached Q4 error sample and RUN_TRAINING=0; nothing to do.")
            return
        print("Live Qwen error extraction would run here (RUN_TRAINING=1); left to a GPU session.")
        return

    error_cases = pd.read_csv(ERROR_CASES_PATH)
    print(f"\nLoaded cached 10-error sample: {ERROR_CASES_PATH}")
    print("Error type counts in selected Q4 sample:")
    print(error_cases["error_type"].value_counts().to_string())

    # Q4.2 - 4 explained examples + the 4-shot prompt
    if EXPLAINED_EXAMPLES_PATH.exists():
        explained = pd.read_csv(EXPLAINED_EXAMPLES_PATH)
        print("\nFour explanation examples for the report:")
        show(explained[["case_id", TEXT_COL, VARIETY_COL, "gold_label_name", "predicted_label_name", "manual_explanation"]])
    if FEWSHOT_PROMPT_PATH.exists():
        prompt = FEWSHOT_PROMPT_PATH.read_text()
        print("\nPrompt preview:")
        print(prompt[:1200] + ("..." if len(prompt) > 1200 else ""))

    # Q4.3 - few-shot recovery summary
    if FEWSHOT_RESULTS_PATH.exists():
        results = pd.read_csv(FEWSHOT_RESULTS_PATH)
        before = int(results["adapter_was_correct"].sum())
        after = int(results["fewshot_was_correct"].sum())
        print(f"\nBefore few-shot prompt: adapter correct on {before}/{len(results)} selected errors")
        print(f"After  few-shot prompt: prompt correct on {after}/{len(results)} selected errors")
        print(f"Net improvement: {after - before:+d} examples")
        show(results[["case_id", TEXT_COL, VARIETY_COL, "gold_label_name",
                      "adapter_prediction_name", "fewshot_prediction_name", "fewshot_was_correct", "raw_qwen_response"]])
    else:
        print("\nNo cached few-shot result table; run scripts/staged/q4_prompt_v2.py on a GPU to generate it.")


def _select_best_qwen_adapter_row(master_df: pd.DataFrame) -> pd.Series:
    qwen_rows = master_df[(master_df["task"].astype(str) == "sarcasm")
                          & (master_df["model_key"].astype(str).str.startswith("qwen2.5-3b-qlora"))
                          & (master_df["tested_on"].astype(str).isin(VARIETIES))
                          & (master_df["adapter_path"].notna())].copy()
    if qwen_rows.empty:
        raise ValueError("No completed Qwen/Q-LoRA sarcasm adapter rows found in the master registry.")
    qwen_rows["test_macro_f1"] = pd.to_numeric(qwen_rows["test_macro_f1"], errors="coerce")
    qwen_rows["val_macro_f1"] = pd.to_numeric(qwen_rows["val_macro_f1"], errors="coerce")
    return qwen_rows.sort_values(["test_macro_f1", "val_macro_f1"], ascending=False).iloc[0]


if __name__ == "__main__":
    main()
