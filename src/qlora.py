"""Qwen2.5-3B Q-LoRA: 4-bit NF4 base + LoRA adapters for sarcasm classification,
plus the 4-bit causal-LM loader used by the Q4 few-shot prompting experiment."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM,
    BitsAndBytesConfig, get_cosine_schedule_with_warmup,
)
from peft import (
    LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training, TaskType,
)

from .metrics import clear_vram
from .transformers_ft import TextClassificationDataset, evaluate_transformer_model


# --------------------------------------------------------------------------- #
# Qwen Q-LoRA configuration
# --------------------------------------------------------------------------- #
LORA_MODEL = "Qwen/Qwen2.5-3B-Instruct"
QWEN_MODEL_KEY = "qwen2.5-3b-qlora"
QWEN_MODEL_NAME = "Qwen2.5-3B-Instruct + QLoRA"
QWEN_ATTENTION_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
QWEN_MLP_TARGET_MODULES = ["gate_proj", "up_proj", "down_proj"]
QWEN_TARGET_MODULES = QWEN_ATTENTION_TARGET_MODULES
QWEN_LORA_TARGET_MODULE_SETS = {
    "attention": QWEN_ATTENTION_TARGET_MODULES,
    "attention_mlp": QWEN_ATTENTION_TARGET_MODULES + QWEN_MLP_TARGET_MODULES,
}
QWEN_DEFAULT_LORA_RANK = 16
QWEN_DEFAULT_LORA_ALPHA = 32
QWEN_DEFAULT_LORA_BIAS = "none"
QWEN_DEFAULT_LORA_DROPOUT = 0.1

# Kept ablation row: attention_mlp targets at rank=16 (used as the en-IN deployment route).
# The full sweep (3 ranks x 2 target-module sets = 6 configs) ran once previously.
QWEN_LORA_TARGET_RANK_SWEEP = [
    {
        "target_modules_key": "attention_mlp",
        "target_modules": QWEN_ATTENTION_TARGET_MODULES + QWEN_MLP_TARGET_MODULES,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_bias": "none",
    },
]
QWEN_LORA_RANK_BIAS_SWEEP = QWEN_LORA_TARGET_RANK_SWEEP  # legacy alias


def qwen_slug_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()


def qwen_lora_variant_slug(lora_rank: int, lora_bias: str = "none", target_modules_key: str = "attention") -> str:
    return f"{qwen_slug_token(target_modules_key)}_rank{lora_rank}_bias_{qwen_slug_token(lora_bias)}"


def qwen_lora_variant_key(lora_rank: int, lora_bias: str = "none", target_modules_key: str = "attention") -> str:
    return f"{QWEN_MODEL_KEY}_{qwen_lora_variant_slug(lora_rank, lora_bias, target_modules_key)}"


# --------------------------------------------------------------------------- #
# Model building + training
# --------------------------------------------------------------------------- #
def count_trainable_parameters(model):
    trainable = 0
    total = 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    return {
        "total": total,
        "trainable": trainable,
        "trainable_percentage": 100 * trainable / total if total > 0 else 0,
    }


def build_qlora_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def require_free_vram(min_free_gb: float = 8.0) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Q-LoRA requires CUDA, but torch.cuda.is_available() is False.")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gb = free_bytes / 1024 ** 3
    total_gb = total_bytes / 1024 ** 3
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"Not enough free GPU memory for Qwen Q-LoRA: {free_gb:.2f} GiB free / {total_gb:.2f} GiB total. "
            "Stop other GPU jobs first with `nvidia-smi`, or reduce batch/max_length."
        )


def build_quantized_sequence_classifier(model_name: str, num_labels: int, tokenizer, device: str = "cuda"):
    require_free_vram(min_free_gb=8.0)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    device_map = {"": 0} if torch.cuda.is_available() and str(device).startswith("cuda") else {"": "cpu"}
    if device_map == {"": "cpu"}:
        raise RuntimeError("Q-LoRA 4-bit inference/training requires a CUDA GPU; CPU offload is disabled.")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=False,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    return model


def train_qlora_model(model_name, train_df, val_df, text_col, label_col,
                      variety, seed=42, max_length=128, batch_size=2,
                      grad_accum_steps=4, learning_rate=1e-4, epochs=7,
                      patience=2, use_class_weights=True, device="cuda", adapter_output_dir=None,
                      lora_rank=16, lora_alpha=None, lora_dropout=0.1,
                      lora_bias="none", target_modules=None):
    """Train Qwen2.5-style 3B sequence classification with 4-bit Q-LoRA.

    Uses NF4 quantization, configurable LoRA rank/target modules, cosine LR schedule,
    gradient accumulation, class weighting, and early stopping on validation Macro-F1.
    """
    from .config import set_seed
    set_seed(seed)
    clear_vram()

    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    target_modules = list(target_modules)
    target_modules_label = "+".join(target_modules)
    if lora_alpha is None:
        lora_alpha = 2 * lora_rank

    print(
        f"  Training Q-LoRA for {variety} "
        f"(seed={seed}, batch={batch_size}x{grad_accum_steps}, rank={lora_rank}, bias={lora_bias}, targets={target_modules_label})"
    )

    tokenizer = build_qlora_tokenizer(model_name)
    num_labels = len(train_df[label_col].unique())
    model = build_quantized_sequence_classifier(model_name, num_labels, tokenizer, device=device)
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=lora_bias,
        task_type=TaskType.SEQ_CLS,
        target_modules=target_modules,
    )

    model = get_peft_model(model, lora_config)
    param_info = count_trainable_parameters(model)
    param_info.update({
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "lora_bias": lora_bias,
        "lora_target_modules": target_modules_label,
    })
    print(f'    Trainable params: {param_info["trainable"]:,} ({param_info["trainable_percentage"]:.2f}%)')

    train_dataset = TextClassificationDataset(train_df[text_col], train_df[label_col], tokenizer, max_length)
    val_dataset = TextClassificationDataset(val_df[text_col], val_df[label_col], tokenizer, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    steps_per_epoch = max(1, int(np.ceil(len(train_loader) / grad_accum_steps)))
    num_training_steps = max(1, steps_per_epoch * epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )

    class_weights = None
    if use_class_weights:
        labels = train_df[label_col].values
        weights = []
        for c in range(num_labels):
            count = np.sum(labels == c)
            weights.append(len(labels) / (num_labels * count) if count > 0 else 1.0)
        class_weights = torch.tensor(weights, dtype=torch.float).to(device)

    history = []
    best_val_f1 = -1
    best_state = None
    epochs_no_improve = 0
    trainable_markers = ["lora", "modules_to_save", "score", "classifier"]
    if lora_bias != "none":
        trainable_markers.append("bias")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            labels = batch["labels"].to(device)
            loss = F.cross_entropy(outputs.logits, labels, weight=class_weights)
            loss = loss / grad_accum_steps
            loss.backward()

            should_step = ((step + 1) % grad_accum_steps == 0) or ((step + 1) == len(train_loader))
            if should_step:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_loss += loss.item() * grad_accum_steps

        _, _, val_metrics = evaluate_transformer_model(model, val_loader, device)
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss / max(1, len(train_loader)),
            "val_macro_f1": val_metrics["macro_f1"],
            "val_macro_precision": val_metrics["macro_precision"],
            "val_macro_recall": val_metrics["macro_recall"],
        })
        print(f'    Epoch {epoch+1}/{epochs}: train_loss={history[-1]["train_loss"]:.4f}, val_macro_f1={val_metrics["macro_f1"]:.4f}')

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
                if any(marker in k.lower() for marker in trainable_markers)
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict({**model.state_dict(), **{k: v.to(device) for k, v in best_state.items()}}, strict=False)
        del best_state

    if adapter_output_dir is not None:
        adapter_output_dir = Path(adapter_output_dir)
        adapter_output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(adapter_output_dir)
        tokenizer.save_pretrained(adapter_output_dir)
        print(f"    Saved adapter to: {adapter_output_dir}")

    history_df = pd.DataFrame(history)
    return model, tokenizer, history_df, param_info


def load_qlora_adapter(model_name: str, adapter_path: Path, num_labels: int = 2, device: str = "cuda"):
    tokenizer = build_qlora_tokenizer(model_name)
    model = build_quantized_sequence_classifier(model_name, num_labels, tokenizer, device=device)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    param_info = count_trainable_parameters(model)
    return model, tokenizer, param_info


def load_4bit_causal_lm(model_name: str, device: str = "cuda"):
    """Load Qwen as a 4-bit causal LM for few-shot prompting (Q4.3)."""
    require_free_vram(min_free_gb=8.0)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=False,
    )
    model.eval()
    return model, tokenizer
