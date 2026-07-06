"""BESSTIE-CW-26 sentiment & sarcasm pipeline.

Submodules:
    config          constants, paths, seeds, encoder model registry + naming helpers
    data            HuggingFace dataset loading and reporting
    metrics         evaluation metrics, confusion-matrix plotting, VRAM cleanup
    classical       TF-IDF / GloVe classical baselines
    transformers_ft encoder fine-tuning + threshold tuning
    qlora           Qwen2.5-3B Q-LoRA training / loading
    registry        master experiment registry (resume-from-cache)

Imports are intentionally lazy: import the submodule you need rather than the
whole package, so that scripts touching only classical/EDA code don't pull in
bitsandbytes/peft.
"""

__all__ = [
    "config", "data", "metrics", "classical", "transformers_ft", "qlora", "registry",
]
