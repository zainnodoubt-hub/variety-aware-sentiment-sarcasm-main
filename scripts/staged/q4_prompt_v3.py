"""Staged experiment - Q4 prompt v3: CoT + self-consistency + balanced contrastive shots.

v2 reached 3/6 by curing false alarms, but its neutral prior over-corrected and lost one
genuine sarcasm (a precision/recall trade, measured on only 6 items). v3 addresses both
problems:

  1. Balanced *contrastive* few-shot examples (hand-written, leakage-free) covering BOTH
     "looks sarcastic but is literal" AND "looks literal but is sarcastic", so the model
     does not simply learn to default to Not Sarcastic.
  2. Chain-of-Thought with an explicit literal-vs-intended *contradiction check* before the
     label, so subtle sarcasm is not suppressed.
  3. Self-consistency: sample the reasoning k times (default 5) and take the majority label.
  4. A trustworthy denominator: evaluate on the adapter's FULL en-AU error set (131 cases),
     not just 6, with a false-positive / false-negative breakdown - and still report the
     original 6 held-out items for a like-for-like comparison with v1 (2/6) and v2 (3/6).

Inference only (loads the 4-bit Qwen base; no weight updates). Run deliberately on a GPU:

    python scripts/staged/q4_prompt_v3.py                 # all 131 errors, 5 samples each
    python scripts/staged/q4_prompt_v3.py --samples 3 --max-eval 30   # quicker smoke run
    python scripts/staged/q4_prompt_v3.py --write-prompt-only         # just write the prompt
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPTS_DIR.parent
for p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import _bootstrap  # noqa: E402
from src.config import device  # noqa: E402
from src.metrics import clear_vram  # noqa: E402

ERROR_DIR = REPO_ROOT / "outputs" / "tables" / "08_sarcasm_error_analysis"
ADAPTER_PREDS_PATH = ERROR_DIR / "qwen_best_adapter_predictions.csv"
HELDOUT6_PATH = ERROR_DIR / "qwen_fewshot_remaining_6_examples.csv"
V3_PROMPT_PATH = ERROR_DIR / "qwen_sarcasm_cot_contrastive_prompt_v3.txt"
V3_RESULTS_PATH = ERROR_DIR / "qwen_fewshot_v3_results.csv"
SARCASM_ID_TO_LABEL = {0: "Not Sarcastic", 1: "Sarcastic"}

# Hand-written, balanced, leakage-free contrastive demonstrations. Each shows the exact
# Chain-of-Thought format we want back: a literal reading, an intended reading, an explicit
# contradiction verdict, then the label. Both error directions are represented so the model
# learns the *distinction*, not a default answer.
CONTRASTIVE_SHOTS = [
    {  # hard negative: emphatic / SHOUTY but a sincere complaint -> NOT sarcastic
        "variety": "en-AU",
        "text": "WORST burger ever. COLD chips, NO sauce, waited 40 mins. Never coming back!!!",
        "literal": "An angry complaint about a bad meal and long wait.",
        "intended": "The author genuinely means it was bad - the anger is sincere, not ironic.",
        "contradiction": "no",
        "label": 0,
    },
    {  # subtle positive: over-positive wording about a bad outcome -> SARCASTIC
        "variety": "en-AU",
        "text": "Love that the train's cancelled again. Brilliant start to the week, truly.",
        "literal": "The writer says they love the cancellation and calls it a brilliant start.",
        "intended": "They are annoyed; praising a cancellation reverses the polarity.",
        "contradiction": "yes",
        "label": 1,
    },
    {  # hard negative: rhetorical question, sincere -> NOT sarcastic
        "variety": "en-UK",
        "text": "Why is rent so high now? Genuinely struggling to keep up with it.",
        "literal": "A question expressing real difficulty with rising rent.",
        "intended": "A sincere expression of struggle; no opposite meaning is implied.",
        "contradiction": "no",
        "label": 0,
    },
    {  # subtle positive: mock-agreement, looks literal -> SARCASTIC
        "variety": "en-IN",
        "text": "Sure, because adding one more form will definitely fix the whole system.",
        "literal": "Agreement that one more form will fix the system.",
        "intended": "The author thinks it will not help at all; 'sure...definitely' is mock-agreement.",
        "contradiction": "yes",
        "label": 1,
    },
    {  # hard negative: blunt factual statement -> NOT sarcastic
        "variety": "en-AU",
        "text": "Walls are paper thin in these units, you can hear everything next door.",
        "literal": "A factual complaint that the walls are thin.",
        "intended": "Read literally; the literal and intended meanings agree.",
        "contradiction": "no",
        "label": 0,
    },
    {  # subtle positive: faux-praise of an absurd outcome -> SARCASTIC
        "variety": "en-UK",
        "text": "Great, so the fix is to do nothing and hope it sorts itself out. Genius plan.",
        "literal": "Calling 'do nothing' a great, genius plan.",
        "intended": "The author thinks the plan is bad; the praise is ironic.",
        "contradiction": "yes",
        "label": 1,
    },
]


def safe(value) -> str:
    return str(value).replace("{", "{{").replace("}", "}}").replace("\n", " ")


def build_cot_prompt(shots) -> str:
    lines = [
        "Task: Decide whether a piece of text is sarcastic for the given variety of English.",
        "Definition: a text is SARCASTIC only if its literal wording and its intended meaning",
        "point in OPPOSITE directions (a polarity reversal / incongruity).",
        "Base rate: sarcasm is RARE here - only about 1 post in 14 is sarcastic, so most posts are sincere.",
        "Anger, CAPS, exclamation marks or rhetorical questions are NOT sarcasm by themselves;",
        "a blunt sincere complaint, a real question, or a factual statement is Not Sarcastic.",
        "Label Sarcastic ONLY when there is a CLEAR polarity reversal - the author plainly means the",
        "opposite of the words (e.g. praising a bad outcome, mock-agreement). If the reading is unclear",
        "or the post is merely emphatic/angry, answer Not Sarcastic.",
        "",
        "Reason in three steps, then answer:",
        "  Literal: what the words say on the surface.",
        "  Intended: what the author actually means in context.",
        "  Contradiction: yes only if the two CLEARLY conflict, otherwise no.",
        "Contradiction = yes -> Label: 1.   Contradiction = no (or unsure) -> Label: 0.",
        "",
        "Worked examples:",
    ]
    for i, s in enumerate(shots, 1):
        lines += [
            f"Example {i}:",
            f"Variety: {s['variety']}",
            f'Text: "{safe(s["text"])}"',
            f"Literal: {safe(s['literal'])}",
            f"Intended: {safe(s['intended'])}",
            f"Contradiction: {s['contradiction']}",
            f"Label: {s['label']}",
            "",
        ]
    lines += [
        "Now classify the new item with the same three steps.",
        "Variety: {variety}",
        'Text: "{text}"',
        "Reply with the four lines: Literal:, Intended:, Contradiction:, Label: (0 or 1).",
        "Literal:",
    ]
    return "\n".join(lines)


def parse_label(response_text: str):
    """Take the LAST explicit `Label: x` (the final decision after CoT); then fall back."""
    text = str(response_text)
    matches = re.findall(r"label\s*[:\-]?\s*([01])", text, flags=re.IGNORECASE)
    if matches:
        return int(matches[-1])
    contra = re.findall(r"contradiction\s*[:\-]?\s*(yes|no)", text, flags=re.IGNORECASE)
    if contra:
        return 1 if contra[-1].lower() == "yes" else 0
    low = text.lower()
    if "not sarcastic" in low:
        return 0
    if "sarcastic" in low:
        return 1
    return None


def majority_label(labels):
    valid = [x for x in labels if x is not None]
    if not valid:
        return None
    ones = sum(valid)
    zeros = len(valid) - ones
    if ones == zeros:
        return None  # genuine tie -> treat as undecided (counts as not recovered)
    return 1 if ones > zeros else 0


def main() -> None:
    samples = _arg_int("--samples", 5)
    max_eval = _arg_int("--max-eval", 0)  # 0 = all

    prompt_template = build_cot_prompt(CONTRASTIVE_SHOTS)
    V3_PROMPT_PATH.write_text(prompt_template)
    print(f"Wrote v3 CoT + contrastive prompt: {V3_PROMPT_PATH}\n")
    print("--- v3 prompt preview ---")
    print(prompt_template[:1400] + "...\n")

    if "--write-prompt-only" in sys.argv:
        print("--write-prompt-only: prompt written; skipping GPU inference.")
        return

    if not ADAPTER_PREDS_PATH.exists():
        print(f"Missing adapter prediction cache: {ADAPTER_PREDS_PATH}")
        return
    preds = pd.read_csv(ADAPTER_PREDS_PATH)
    errors = preds[preds["gold_label"] != preds["predicted_label"]].reset_index(drop=True)
    heldout_texts = set(pd.read_csv(HELDOUT6_PATH)["text"].astype(str)) if HELDOUT6_PATH.exists() else set()
    if max_eval and max_eval < len(errors):
        errors = errors.head(max_eval)
    print(f"Evaluating v3 on {len(errors)} adapter errors "
          f"(self-consistency = {samples} samples, majority vote)\n")

    import torch
    from src.qlora import LORA_MODEL, load_4bit_causal_lm

    clear_vram(verbose=True)
    model, tokenizer = load_4bit_causal_lm(LORA_MODEL, device=device)
    torch.manual_seed(42)

    rows = []
    for i, row in errors.iterrows():
        prompt = prompt_template.format(variety=row["variety"], text=str(row["text"]).replace("\n", " "))
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072).to(model.device)
        with torch.no_grad():
            if samples > 1:
                out = model.generate(**inputs, max_new_tokens=160, do_sample=True, temperature=0.7,
                                     top_p=0.9, num_return_sequences=samples, pad_token_id=tokenizer.eos_token_id)
            else:
                out = model.generate(**inputs, max_new_tokens=160, do_sample=False,
                                     temperature=None, pad_token_id=tokenizer.eos_token_id)
        gen_len = inputs["input_ids"].shape[1]
        labels = [parse_label(tokenizer.decode(seq[gen_len:], skip_special_tokens=True)) for seq in out]
        pred = majority_label(labels)
        gold = int(row["gold_label"])
        rows.append({
            "text": row["text"], "variety": row["variety"], "error_type": row["error_type"],
            "gold_label": gold, "gold_label_name": SARCASM_ID_TO_LABEL[gold],
            "adapter_prediction": int(row["predicted_label"]),
            "v3_prediction": pred, "v3_prediction_name": SARCASM_ID_TO_LABEL.get(pred, "Undecided"),
            "v3_was_correct": pred == gold, "votes": str(labels),
            "in_original_6": str(row["text"]) in heldout_texts,
        })
        if (i + 1) % 20 == 0:
            print(f"  ...{i + 1}/{len(errors)} done")

    results = pd.DataFrame(rows)
    results.to_csv(V3_RESULTS_PATH, index=False)
    del model, tokenizer
    clear_vram(verbose=True)

    _report(results)
    print(f"\nSaved: {V3_RESULTS_PATH}")


def _report(results: pd.DataFrame) -> None:
    total = len(results)
    rec = int(results["v3_was_correct"].sum())
    fp = results[results["error_type"].str.startswith("False Positive")]
    fn = results[results["error_type"].str.startswith("False Negative")]
    print("\n================= v3 RESULTS =================")
    print(f"Full en-AU adapter error set: {rec}/{total} recovered ({100*rec/total:.1f}%)")
    if len(fp):
        print(f"  False alarms fixed (gold=Not Sarcastic): {int(fp['v3_was_correct'].sum())}/{len(fp)}")
    if len(fn):
        print(f"  Missed sarcasm recovered (gold=Sarcastic): {int(fn['v3_was_correct'].sum())}/{len(fn)}")
    six = results[results["in_original_6"]]
    if len(six):
        print(f"\nOriginal 6 held-out (compare: v1=2/6, v2=3/6): v3 = {int(six['v3_was_correct'].sum())}/{len(six)}")


def _arg_int(flag: str, default: int) -> int:
    if flag in sys.argv:
        try:
            return int(sys.argv[sys.argv.index(flag) + 1])
        except (IndexError, ValueError):
            pass
    return default


if __name__ == "__main__":
    main()
