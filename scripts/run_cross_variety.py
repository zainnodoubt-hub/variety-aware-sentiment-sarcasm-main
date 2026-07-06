"""Section 5 - Cross-variety evaluation (Q2.2).

DeBERTa-v3-base trained on each variety, tested on all three (sarcasm, two seeds).
Resumable: completed cells are skipped, so on a populated project this regenerates
the class-balance table, the gap analysis, and the two heatmaps from cached results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

import _bootstrap
from _bootstrap import show
from src.config import (
    TEXT_COL, VARIETY_COL, SARCASM_COL, SEEDS, VARIETIES, FIGURES_DIR, device,
    TRANSFORMER_MODEL_CONFIGS, selected_cross_variety_transformer_model_keys,
    is_transformer_crossvariety_key, transformer_checkpoint_dir, transformer_cm_filename,
)
from src.metrics import evaluate_predictions, plot_confusion_matrix, clear_vram
from src.transformers_ft import TextClassificationDataset, train_transformer_model, evaluate_transformer_model
from src.registry import load_master_results, is_experiment_done, append_result_row, TABLES_DIR


def main() -> None:
    _, train_df, val_df, test_df = _bootstrap.load_all_splits()

    # 5.1 Per-variety test-set class balance
    balance = (test_df.groupby(VARIETY_COL)[SARCASM_COL]
               .agg(test_size="count", n_sarcastic="sum", sarcastic_proportion="mean").round(4))
    balance["sarcastic_pct"] = (100 * balance["sarcastic_proportion"]).round(2).astype(str) + "%"
    balance.to_csv(TABLES_DIR / "01_test_class_balance_per_variety.csv")
    print("Per-variety test-set class balance (sarcasm):")
    show(balance)

    # Cross-variety training loop (resumable)
    cross_keys = selected_cross_variety_transformer_model_keys()
    print(f"\nCross-variety encoder transformers (idempotent): {cross_keys}")
    for base_key in cross_keys:
        cfg = TRANSFORMER_MODEL_CONFIGS[base_key]
        model_key = f"{base_key}_crossvariety"
        for train_variety in VARIETIES:
            v_train = train_df[train_df[VARIETY_COL] == train_variety].reset_index(drop=True)
            v_val = val_df[val_df[VARIETY_COL] == train_variety].reset_index(drop=True)
            if len(v_train) < 50 or len(v_val) < 10:
                print(f'Skipping {cfg["display_name"]} {train_variety}: insufficient data')
                continue
            for seed in SEEDS:
                missing = [tv for tv in VARIETIES
                           if not is_experiment_done(model_key, "sarcasm", train_variety, train_variety, tv, seed)]
                if not missing:
                    print(f"  [SKIP] {model_key} | train={train_variety} | seed={seed} (all test varieties done)")
                    continue
                if not _bootstrap.training_enabled():
                    print(f"  [SKIP-NO-TRAIN] {model_key} | train={train_variety} | seed={seed} (RUN_TRAINING=0)")
                    continue
                print(f'\n  [RUN] {cfg["display_name"]} cross-variety train={train_variety}, seed={seed}; missing={missing}')
                model = tokenizer = None
                try:
                    model, tokenizer, history_df = train_transformer_model(
                        model_name=cfg["hf_id"], train_df=v_train, val_df=v_val, text_col=TEXT_COL,
                        label_col=SARCASM_COL, task_name="sarcasm", seed=seed, max_length=cfg["max_length"],
                        batch_size=cfg["batch_size"], eval_batch_size=cfg["eval_batch_size"],
                        grad_accum_steps=cfg["grad_accum_steps"], learning_rate=cfg["learning_rate"],
                        epochs=cfg["epochs"], patience=cfg["patience"], use_class_weights=True, device=device)
                    ckpt = transformer_checkpoint_dir(model_key, "sarcasm", seed, scope="cross_variety", train_variety=train_variety)
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
                                              f'{cfg["display_name"]}: train={train_variety}, test={tv} (seed={seed})',
                                              filename=transformer_cm_filename(model_key, "sarcasm", seed, scope="cross_variety",
                                                                               trained_on=train_variety, tested_on=tv), show=False)
                        append_result_row({
                            "model_type": cfg["family"], "model_key": model_key, "model_name": cfg["display_name"],
                            "task": "sarcasm", "seed": seed, "trained_on": train_variety, "validated_on": train_variety,
                            "tested_on": tv, "val_macro_f1": history_df["val_macro_f1"].max() if len(history_df) else np.nan,
                            "test_macro_f1": metrics_full["macro_f1"], "test_macro_precision": metrics_full["macro_precision"],
                            "test_macro_recall": metrics_full["macro_recall"],
                            "test_f1_Not Sarcastic": metrics_full.get("f1_Not Sarcastic", 0),
                            "test_f1_Sarcastic": metrics_full.get("f1_Sarcastic", 0),
                            "epochs_trained": len(history_df), "trainable_params_pct": 100.0,
                            "adapter_path": str(ckpt), "is_same_variety": train_variety == tv,
                            "notes": f'hf_id={cfg["hf_id"]}; cross_variety'})
                finally:
                    del model, tokenizer
                    clear_vram(verbose=True)

    master = load_master_results()
    cross_df = master[master["model_key"].astype(str).apply(is_transformer_crossvariety_key)].copy()
    cross_df.to_csv(TABLES_DIR / "05_cross_variety_results.csv", index=False)
    print(f"\nCross-variety rows available: {len(cross_df)}")

    if len(cross_df) == 0:
        print("No cross-variety results to summarise.")
        return

    # 5.x Variety gap analysis
    gap_summary = []
    for trained in VARIETIES:
        same = cross_df[(cross_df["trained_on"] == trained) & (cross_df["tested_on"] == trained)]["test_macro_f1"]
        if len(same) == 0:
            continue
        same_v = same.mean()
        for tested in VARIETIES:
            if trained == tested:
                continue
            cross = cross_df[(cross_df["trained_on"] == trained) & (cross_df["tested_on"] == tested)]["test_macro_f1"]
            if len(cross) == 0:
                continue
            gap_summary.append({"trained_on": trained, "tested_on": tested,
                                "same_variety_f1": round(same_v, 3), "cross_variety_f1": round(cross.mean(), 3),
                                "variety_gap": round(same_v - cross.mean(), 3)})
    if gap_summary:
        print("\nVariety Gap (in-variety F1 - cross-variety F1):")
        show(pd.DataFrame(gap_summary))

    # Heatmaps (macro-F1 and sarcastic-F1)
    macro = (cross_df.groupby(["trained_on", "tested_on"])["test_macro_f1"].mean()
             .unstack().reindex(index=VARIETIES, columns=VARIETIES).round(3))
    sarc = (cross_df.groupby(["trained_on", "tested_on"])["test_f1_Sarcastic"].mean()
            .unstack().reindex(index=VARIETIES, columns=VARIETIES).round(3))
    plt.figure(figsize=(7, 5))
    sns.heatmap(macro, annot=True, fmt=".3f", cmap="YlGn", square=True, vmin=0, vmax=1,
                cbar_kws={"label": "Macro-F1 (mean of 2 seeds)"})
    plt.title("Cross-Variety Macro-F1 (sarcasm) - DeBERTa-v3-base, seeds {42, 123}")
    plt.xlabel("Tested on"); plt.ylabel("Trained on")
    plt.tight_layout(); plt.savefig(FIGURES_DIR / "05_cross_variety_heatmap.png", dpi=300, bbox_inches="tight"); plt.close()

    vmax_sarc = float(sarc.max().max()) * 1.05
    plt.figure(figsize=(7, 5))
    sns.heatmap(sarc, annot=True, fmt=".3f", cmap="YlGn", square=True, vmin=0, vmax=vmax_sarc,
                cbar_kws={"label": "Sarcastic-class F1 (mean of 2 seeds)"})
    plt.title("Cross-Variety Sarcastic-F1 (minority class) - anchored scale")
    plt.xlabel("Tested on"); plt.ylabel("Trained on")
    plt.tight_layout(); plt.savefig(FIGURES_DIR / "05_cross_variety_heatmap_sarcastic_f1.png", dpi=300, bbox_inches="tight"); plt.close()
    print("\nSaved heatmaps. Macro-F1 matrix:")
    show(macro)
    print("\nSarcastic-F1 matrix:")
    show(sarc)


if __name__ == "__main__":
    main()
