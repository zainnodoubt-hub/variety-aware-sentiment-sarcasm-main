"""Section 9 - Deployment registry + efficiency benchmark.

Builds the 13-route deployment registry via deployment_app.setup() (lazy loaders;
no model is loaded here), saves 09_deployment_model_registry.csv, and times the
CPU classical fallback. The Gradio UI launches only with LAUNCH_GRADIO_APP=1 and
the GPU Qwen latency benchmark only with RUN_QWEN_EFFICIENCY_BENCHMARK=1.
"""
from __future__ import annotations

import time

import pandas as pd

import _bootstrap
from _bootstrap import show
from src import config
from src.config import (
    TEXT_COL, SARCASM_COL, SENTIMENT_COL, VARIETIES, device, env_flag,
    is_transformer_model_key, is_plain_transformer_model_key,
    is_transformer_reddit_key, is_transformer_crossvariety_key, transformer_checkpoint_dir,
)
from src.classical import build_classical_model
from src.metrics import clear_vram
from src.qlora import LORA_MODEL, load_qlora_adapter
from src.registry import load_master_results, CLASSICAL_MODEL_KEYS, TABLES_DIR

TASK_LABEL_COLS = {"sarcasm": SARCASM_COL, "sentiment": SENTIMENT_COL}
BENCHMARK_TEXTS = {
    "short_single": ["Great, the delivery is late again. Just perfect."],
    "long_single":  [" ".join(["Great, the delivery is late again and everyone is thrilled about waiting outside."] * 20)],
    "batch_32":     ["Great, the delivery is late again. Just perfect."] * 32,
}


def main() -> None:
    import deployment_app
    _, train_df, val_df, _ = _bootstrap.load_all_splits()
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)

    deployment_app.setup(
        master_df=load_master_results(), train_val_df=train_val_df,
        text_col=TEXT_COL, sarcasm_col=SARCASM_COL, sentiment_col=SENTIMENT_COL,
        varieties=VARIETIES, classical_model_keys=CLASSICAL_MODEL_KEYS,
        is_transformer_model_key=is_transformer_model_key,
        is_plain_transformer_model_key=is_plain_transformer_model_key,
        is_transformer_reddit_key=is_transformer_reddit_key,
        is_transformer_crossvariety_key=is_transformer_crossvariety_key,
        transformer_checkpoint_dir=transformer_checkpoint_dir,
        build_classical_model=build_classical_model, load_qlora_adapter=load_qlora_adapter,
        clear_vram=clear_vram, lora_model=LORA_MODEL, device=device,
        enable_live_qwen=env_flag("ENABLE_LIVE_QWEN_DEPLOYMENT", "1"),
        enable_live_transformer=env_flag("ENABLE_LIVE_TRANSFORMER_DEPLOYMENT", "1"),
    )

    registry_df = deployment_app.deployment_registry_df
    registry_path = TABLES_DIR / "09_deployment_model_registry.csv"
    registry_df.to_csv(registry_path, index=False)
    print(f"Full deployment registry saved to: {registry_path}")
    route_cols = ["task", "route_label", "ui_variety", "model_key", "trained_on",
                  "seed", "test_macro_f1", "class1_f1", "live_backend", "artifact_status"]
    print("Compact deployment route view:")
    show(registry_df[[c for c in route_cols if c in registry_df.columns]])

    if env_flag("LAUNCH_GRADIO_APP") and deployment_app.deployment_demo is not None:
        # share=True (GRADIO_SHARE=1) gives a public link, handy on a remote server
        # where the local 127.0.0.1:7860 port is not forwarded.
        print("\nLaunching Gradio UI... (Ctrl-C to stop)")
        deployment_app.deployment_demo.launch(share=env_flag("GRADIO_SHARE"), server_name="0.0.0.0")
        return  # serving is blocking; skip the benchmark when running the UI

    print("\nGradio app object built. Set LAUNCH_GRADIO_APP=1 to launch the UI.")
    _efficiency_benchmark(train_val_df, registry_df)


def _avg_latency_ms(fn, inputs, repeats):
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn(inputs)
    return round((time.perf_counter() - t0) * 1000 / repeats, 2)


def _efficiency_benchmark(train_val_df, registry_df) -> None:
    rows = []
    for task in ["sentiment", "sarcasm"]:
        cw = "balanced" if task == "sarcasm" else None
        model = build_classical_model("tfidf_logreg", class_weight=cw, seed=42)
        model.fit(train_val_df[TEXT_COL], train_val_df[TASK_LABEL_COLS[task]])
        for scenario, texts in BENCHMARK_TEXTS.items():
            rows.append({"task": task, "family": "Classical (TF-IDF + LogReg)", "device": "CPU",
                         "scenario": scenario, "batch_size": len(texts),
                         "avg_latency_ms": _avg_latency_ms(model.predict, texts, 100 if scenario == "short_single" else 20)})

    if env_flag("RUN_QWEN_EFFICIENCY_BENCHMARK"):
        import deployment_app
        import torch
        qwen_route = registry_df[(registry_df["route_label"] == "Sarcasm: Qwen same-variety adapter")
                                 & (registry_df["live_backend"] == "qwen_adapter")].sort_values("test_macro_f1", ascending=False).iloc[0]
        qwen_model, qwen_tok = deployment_app.load_deployment_qwen_adapter(qwen_route)

        def _qwen_predict(texts):
            enc = qwen_tok(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
            with torch.no_grad():
                qwen_model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device))
        for scenario, texts in BENCHMARK_TEXTS.items():
            rows.append({"task": "sarcasm", "family": "Qwen QLoRA (4-bit + adapter)", "device": "GPU",
                         "scenario": scenario, "batch_size": len(texts),
                         "avg_latency_ms": _avg_latency_ms(_qwen_predict, texts, 10 if scenario == "short_single" else 3)})
    else:
        print("Qwen GPU benchmark skipped. Set RUN_QWEN_EFFICIENCY_BENCHMARK=1 to include it.")

    efficiency_df = pd.DataFrame(rows)
    efficiency_df.to_csv(TABLES_DIR / "09_efficiency_benchmarks.csv", index=False)
    print("\nEfficiency benchmark:")
    show(efficiency_df)


if __name__ == "__main__":
    main()
