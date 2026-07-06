"""Run the whole pipeline end to end, in section order (2 -> 9).

By default this resumes from the cached master registry and saved checkpoints:
it rebuilds every table and figure without re-training (RUN_TRAINING=0). Set
RUN_TRAINING=1 to re-enable the resumable GPU train/evaluate loops for any rows
that are missing from the registry.

    python scripts/run_all.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (sets sys.path + matplotlib backend + dirs)

import run_eda
import run_classical
import run_transformers
import run_cross_variety
import run_qlora
import run_evaluation
import run_error_analysis
import run_deployment

STAGES = [
    ("Section 2  - EDA / vocabulary", run_eda.main),
    ("Section 3  - Classical baselines", run_classical.main),
    ("Section 4  - Encoder transformers", run_transformers.main),
    ("Section 5  - Cross-variety", run_cross_variety.main),
    ("Section 6  - Qwen Q-LoRA", run_qlora.main),
    ("Section 7  - Evaluation", run_evaluation.main),
    ("Section 8  - Error analysis (Q4)", run_error_analysis.main),
    ("Section 9  - Deployment + efficiency", run_deployment.main),
]


def main() -> None:
    for title, fn in STAGES:
        print("\n" + "=" * 78)
        print(f">>> {title}")
        print("=" * 78)
        fn()
    print("\nAll stages complete.")


if __name__ == "__main__":
    main()
