"""Encoder-transformer fine-tuning: dataset, training loop with early stopping
and memory-aware mixed precision, plus probability prediction and threshold
tuning helpers shared by the encoder and Qwen experiments."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from .config import (
    set_seed, PREDICTIONS_DIR, TEXT_COL, SARCASM_COL, SOURCE_COL, VARIETY_COL, VARIETIES,
)
from .metrics import evaluate_predictions


class TextClassificationDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def cuda_amp_enabled(device):
    return str(device).startswith("cuda") and torch.cuda.is_available()


def get_cuda_amp_dtype():
    return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16


def load_sequence_classifier_fp32(model_name, num_labels):
    try:
        return AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            dtype=torch.float32,
        )
    except TypeError:
        return AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            torch_dtype=torch.float32,
        )


def evaluate_transformer_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", dtype=get_cuda_amp_dtype(), enabled=cuda_amp_enabled(device)):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_p = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_r = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    metrics = {"macro_f1": macro_f1, "macro_precision": macro_p, "macro_recall": macro_r}
    return all_labels, all_preds, metrics


def train_transformer_model(model_name, train_df, val_df, text_col, label_col,
                            task_name, seed=42, max_length=128, batch_size=4,
                            eval_batch_size=None, grad_accum_steps=4, learning_rate=2e-5,
                            epochs=7, patience=2, use_class_weights=False, device="cuda"):
    """Train an encoder classifier with early stopping on Macro-F1 and memory-aware AMP."""
    set_seed(seed)
    eval_batch_size = batch_size if eval_batch_size is None else eval_batch_size

    # Aggressive memory cleanup before starting.
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    print(f"  Training {model_name} on {task_name} (seed={seed}, batch={batch_size}x{grad_accum_steps} accum)")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    num_labels = len(train_df[label_col].unique())
    model = load_sequence_classifier_fp32(model_name, num_labels)
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.to(device)
    model.float()
    trainable_dtypes = sorted({str(p.dtype) for p in model.parameters() if p.requires_grad})
    print(f"    Trainable parameter dtypes after load: {trainable_dtypes}")

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    train_dataset = TextClassificationDataset(train_df[text_col], train_df[label_col], tokenizer, max_length)
    val_dataset = TextClassificationDataset(val_df[text_col], val_df[label_col], tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=0)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    steps_per_epoch = max(1, int(np.ceil(len(train_loader) / grad_accum_steps)))
    num_training_steps = max(1, steps_per_epoch * epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )

    amp_enabled = cuda_amp_enabled(device)
    amp_dtype = get_cuda_amp_dtype() if amp_enabled else torch.float32
    use_grad_scaler = amp_enabled and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler) if amp_enabled else None
    print(f"    AMP enabled: {amp_enabled}, dtype: {amp_dtype}, grad_scaler: {use_grad_scaler}")

    if use_class_weights:
        labels = train_df[label_col].values
        weights = []
        for c in range(num_labels):
            count = np.sum(labels == c)
            weights.append(len(labels) / (num_labels * count) if count > 0 else 1.0)
        class_weights = torch.tensor(weights, dtype=torch.float).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fn = nn.CrossEntropyLoss()

    history = []
    best_val_f1 = -1
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                loss = loss_fn(outputs.logits, batch["labels"].to(device))
                loss = loss / grad_accum_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            should_step = ((step + 1) % grad_accum_steps == 0) or ((step + 1) == len(train_loader))
            if should_step:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()

            train_loss += loss.item() * grad_accum_steps

        _, _, val_metrics = evaluate_transformer_model(model, val_loader, device)
        avg_train_loss = train_loss / max(1, len(train_loader))
        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_macro_f1": val_metrics["macro_f1"],
            "val_macro_precision": val_metrics["macro_precision"],
            "val_macro_recall": val_metrics["macro_recall"],
        })

        print(f'    Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.4f}, val_macro_f1={val_metrics["macro_f1"]:.4f}')

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        del best_state

    history_df = pd.DataFrame(history)
    return model, tokenizer, history_df


# --------------------------------------------------------------------------- #
# Probability prediction + threshold tuning (shared by Qwen + DeBERTa)
# --------------------------------------------------------------------------- #
def predict_transformer_probabilities(model, data_loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.amp.autocast("cuda", dtype=get_cuda_amp_dtype(), enabled=cuda_amp_enabled(device)):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits.float(), dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.vstack(all_probs)


def evaluate_threshold(y_true, probs, threshold: float):
    y_pred = (probs[:, 1] >= threshold).astype(int)
    metrics, _, cm = evaluate_predictions(y_true, y_pred, ["Not Sarcastic", "Sarcastic"])
    return y_pred, metrics, cm


def get_validation_subset_for_scope(val_df: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "pooled_all_varieties":
        return val_df.reset_index(drop=True)
    if scope == "reddit_only":
        return val_df[val_df[SOURCE_COL].astype(str).str.lower() == "reddit"].reset_index(drop=True)
    if scope in VARIETIES:
        return val_df[val_df[VARIETY_COL] == scope].reset_index(drop=True)
    if scope.startswith("reddit_"):
        variety = scope.replace("reddit_", "")
        return val_df[(val_df[SOURCE_COL].astype(str).str.lower() == "reddit") & (val_df[VARIETY_COL] == variety)].reset_index(drop=True)
    return val_df.reset_index(drop=True)


def get_test_subset_for_scope(test_df: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "pooled_all_varieties":
        return test_df.reset_index(drop=True)
    if scope == "reddit_only":
        return test_df[test_df[SOURCE_COL].astype(str).str.lower() == "reddit"].reset_index(drop=True)
    if scope in VARIETIES:
        return test_df[test_df[VARIETY_COL] == scope].reset_index(drop=True)
    if scope.startswith("reddit_"):
        variety = scope.replace("reddit_", "")
        return test_df[(test_df[SOURCE_COL].astype(str).str.lower() == "reddit") & (test_df[VARIETY_COL] == variety)].reset_index(drop=True)
    return test_df.reset_index(drop=True)


def tune_threshold_for_saved_model(model, tokenizer, val_subset, cache_prefix: str, device="cuda"):
    cache_path = PREDICTIONS_DIR / f"{cache_prefix}_val_probs.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        y_val = cached["labels"]
        val_probs = cached["probs"]
    else:
        val_dataset = TextClassificationDataset(val_subset[TEXT_COL], val_subset[SARCASM_COL], tokenizer, 128)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        y_val, val_probs = predict_transformer_probabilities(model, val_loader, device)
        np.savez(cache_path, labels=y_val, probs=val_probs)

    rows = []
    thresholds = np.round(np.arange(0.10, 0.901, 0.05), 2)
    for threshold in thresholds:
        _, metrics, _ = evaluate_threshold(y_val, val_probs, threshold)
        rows.append({"threshold": threshold, "val_macro_f1": metrics["macro_f1"]})
    sweep_df = pd.DataFrame(rows)
    best_row = sweep_df.sort_values("val_macro_f1", ascending=False).iloc[0]
    return float(best_row["threshold"]), sweep_df
