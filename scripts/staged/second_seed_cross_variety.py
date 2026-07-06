"""Staged experiment - second-seed replication for the cross-variety ancestry result.

The en-IN->en-UK cross-variety result (Q2.2) rests on seed 42 alone; seed 123 collapsed
to majority class. This retrains the en-IN and en-UK cross-variety DeBERTa-v3-base models
on two ADDITIONAL seeds (7 and 2024) and appends the rows to the master registry, so the
heatmaps can be rebuilt with more than one working seed and the hypothesis confirmed or
retired.

NOT run as part of the pipeline. Execute deliberately on a GPU:

    RUN_TRAINING=1 python scripts/staged/second_seed_cross_variety.py
    python scripts/run_cross_variety.py    # rebuild heatmaps with the extra seeds
    python scripts/run_evaluation.py

Edit REPLICATION_SEEDS / REPLICATION_VARIETIES below to widen the sweep.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPTS_DIR.parent
for p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import _bootstrap  # noqa: E402
from _bootstrap import training_enabled  # noqa: E402
from src.config import (  # noqa: E402
    TEXT_COL, VARIETY_COL, SARCASM_COL, device, TRANSFORMER_MODEL_CONFIGS,
    transformer_checkpoint_dir, transformer_cm_filename,
)
from src.metrics import evaluate_predictions, plot_confusion_matrix, clear_vram  # noqa: E402
from src.transformers_ft import (  # noqa: E402
    TextClassificationDataset, train_transformer_model, evaluate_transformer_model,
)
from src.registry import is_experiment_done, append_result_row  # noqa: E402

REPLICATION_SEEDS = [7, 2024]            # additional non-collapsed seeds to try
REPLICATION_VARIETIES = ["en-IN", "en-UK"]  # the two whose seed-123 runs collapsed
ALL_VARIETIES = ["en-AU", "en-IN", "en-UK"]
BASE_KEY = "deberta-v3-base"
MODEL_KEY = f"{BASE_KEY}_crossvariety"


def main() -> None:
    if not training_enabled():
        print("RUN_TRAINING=0: this is a GPU training job. Re-run with RUN_TRAINING=1 to execute.")
        print(f"Would retrain {MODEL_KEY} for {REPLICATION_VARIETIES} on seeds {REPLICATION_SEEDS}.")
        return

    _, train_df, val_df, test_df = _bootstrap.load_all_splits()
    cfg = TRANSFORMER_MODEL_CONFIGS[BASE_KEY]

    for train_variety in REPLICATION_VARIETIES:
        v_train = train_df[train_df[VARIETY_COL] == train_variety].reset_index(drop=True)
        v_val = val_df[val_df[VARIETY_COL] == train_variety].reset_index(drop=True)
        if len(v_train) < 50 or len(v_val) < 10:
            print(f"  [SKIP] {train_variety}: insufficient data")
            continue
        for seed in REPLICATION_SEEDS:
            missing = [tv for tv in ALL_VARIETIES
                       if not is_experiment_done(MODEL_KEY, "sarcasm", train_variety, train_variety, tv, seed)]
            if not missing:
                print(f"  [SKIP] {MODEL_KEY} | train={train_variety} | seed={seed} (already done)")
                continue
            print(f"\n  [RUN] {train_variety} cross-variety, seed={seed}; missing={missing}")
            model = tokenizer = None
            try:
                model, tokenizer, history_df = train_transformer_model(
                    model_name=cfg["hf_id"], train_df=v_train, val_df=v_val, text_col=TEXT_COL,
                    label_col=SARCASM_COL, task_name="sarcasm", seed=seed, max_length=cfg["max_length"],
                    batch_size=cfg["batch_size"], eval_batch_size=cfg["eval_batch_size"],
                    grad_accum_steps=cfg["grad_accum_steps"], learning_rate=cfg["learning_rate"],
                    epochs=cfg["epochs"], patience=cfg["patience"], use_class_weights=True, device=device)
                ckpt = transformer_checkpoint_dir(MODEL_KEY, "sarcasm", seed, scope="cross_variety", train_variety=train_variety)
                ckpt.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(ckpt); tokenizer.save_pretrained(ckpt)
                for tv in missing:
                    v_test = test_df[test_df[VARIETY_COL] == tv].reset_index(drop=True)
                    if len(v_test) == 0:
                        continue
                    ds = TextClassificationDataset(v_test[TEXT_COL], v_test[SARCASM_COL], tokenizer, cfg["max_length"])
                    loader = DataLoader(ds, batch_size=cfg["eval_batch_size"], shuffle=False)
                    y_true, y_pred, _ = evaluate_transformer_model(model, loader, device)
                    metrics_full, _, cm = evaluate_predictions(y_true, y_pred, ["Not Sarcastic", "Sarcastic"])
                    plot_confusion_matrix(cm, ["Not Sarcastic", "Sarcastic"],
                                          f"DeBERTa-v3-base: train={train_variety}, test={tv} (seed={seed})",
                                          filename=transformer_cm_filename(MODEL_KEY, "sarcasm", seed, scope="cross_variety",
                                                                           trained_on=train_variety, tested_on=tv), show=False)
                    append_result_row({
                        "model_type": cfg["family"], "model_key": MODEL_KEY, "model_name": cfg["display_name"],
                        "task": "sarcasm", "seed": seed, "trained_on": train_variety, "validated_on": train_variety,
                        "tested_on": tv, "val_macro_f1": history_df["val_macro_f1"].max() if len(history_df) else np.nan,
                        "test_macro_f1": metrics_full["macro_f1"], "test_macro_precision": metrics_full["macro_precision"],
                        "test_macro_recall": metrics_full["macro_recall"],
                        "test_f1_Not Sarcastic": metrics_full.get("f1_Not Sarcastic", 0),
                        "test_f1_Sarcastic": metrics_full.get("f1_Sarcastic", 0),
                        "epochs_trained": len(history_df), "trainable_params_pct": 100.0,
                        "adapter_path": str(ckpt), "is_same_variety": train_variety == tv,
                        "notes": f'hf_id={cfg["hf_id"]}; cross_variety; second_seed_replication'})
            finally:
                del model, tokenizer
                clear_vram(verbose=True)

    print("\nReplication rows appended. Rebuild heatmaps with: python scripts/run_cross_variety.py")


if __name__ == "__main__":
    main()
