"""Staged experiment - a deliberate Q4 prompt revision (v2).

The original explanation-rich 4-shot prompt recovered 2/6 held-out errors. This is a
genuine revision of that prompt (two-step literal-vs-intended reasoning, a strict
`Label: x` output contract with a hardened parser, sharper cue-specific exemplar
explanations, and a neutral-prior nudge against the adapter's false-positive bias).

It is NOT run as part of the pipeline. Execute it deliberately on a GPU (inference
only - no weight updates):

    python scripts/staged/q4_prompt_v2.py                    # write prompt + run the 6 held-out items
    python scripts/staged/q4_prompt_v2.py --write-prompt-only # just write the revised prompt, no GPU

It reuses the cached 10-error sample / 4 explained / 6 held-out examples, re-runs the
6 held-out items with the v2 prompt, and prints v2 recovery (x/6) next to the original
2/6 for a like-for-like comparison. Report whatever it produces - including no gain.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

# scripts/staged/ -> put repo root AND scripts/ on the path so `import _bootstrap` works.
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPTS_DIR.parent
for p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import _bootstrap  # noqa: E402
from src.config import TEXT_COL, VARIETY_COL, SOURCE_COL, device  # noqa: E402
from src.metrics import clear_vram  # noqa: E402

ERROR_DIR = REPO_ROOT / "outputs" / "tables" / "08_sarcasm_error_analysis"
EXPLAINED_PATH = ERROR_DIR / "qwen_explained_4_examples.csv"
EVAL_PATH = ERROR_DIR / "qwen_fewshot_remaining_6_examples.csv"
ORIGINAL_RESULTS_PATH = ERROR_DIR / "qwen_fewshot_remaining_6_results.csv"
V2_PROMPT_PATH = ERROR_DIR / "qwen_sarcasm_4shot_explanation_prompt_v2.txt"
V2_RESULTS_PATH = ERROR_DIR / "qwen_fewshot_v2_results.csv"
ADAPTER_PREDS_PATH = ERROR_DIR / "qwen_best_adapter_predictions.csv"
V2_FULL_RESULTS_PATH = ERROR_DIR / "qwen_fewshot_v2_full131_results.csv"
SARCASM_ID_TO_LABEL = {0: "Not Sarcastic", 1: "Sarcastic"}


def linguistic_sarcasm_explanation_v2(row: pd.Series) -> str:
    """Cue-specific explanation: name the actual surface/intended contrast."""
    text = str(row[TEXT_COL])
    lower = text.lower()
    variety = str(row.get(VARIETY_COL, "unknown"))
    source = str(row.get(SOURCE_COL, "unknown"))
    gold = int(row["gold_label"])

    cues = []
    if any(t in lower for t in ["great", "brilliant", "legend", "love", "excellent", "amazing", "wow", "perfect", "a++", "good onya"]):
        cues.append("over-positive wording used about a negative situation (irony)")
    if any(t in lower for t in ["yeah right", "as if", "thanks for nothing", "what a", "obviously", "sure"]):
        cues.append("an explicit ironic marker")
    if any(t in lower for t in ["coz", "yaar", "mate", "ute", "arvo", "innit"]):
        cues.append("variety-specific slang that shifts tone")
    if text.isupper() or text.count("!") >= 2:
        cues.append("shouting/emphatic punctuation signalling attitude")
    if not cues:
        cues.append("no lexical irony marker - the cue is purely pragmatic and needs context")

    if gold == 1:
        literal = "Read literally, the words look positive or neutral."
        intended = "The intended meaning is negative/mocking, so the polarity is reversed -> Sarcastic."
    else:
        literal = "Read literally, the statement is a sincere complaint, question, or factual claim."
        intended = "There is no polarity reversal; the literal and intended readings agree -> Not Sarcastic."
    return (f"Context {variety}/{source}. {literal} {intended} "
            f"Decisive cue(s): {'; '.join(cues)}.")


def safe(value) -> str:
    return str(value).replace("{", "{{").replace("}", "}}").replace("\n", " ")


def build_explanation_prompt_v2(examples_df: pd.DataFrame) -> str:
    lines = [
        "Task: Decide whether a piece of text is sarcastic for the given variety of English.",
        "Sarcasm = the literal wording and the intended meaning point in OPPOSITE directions.",
        "If the literal and intended readings AGREE, it is NOT sarcastic, even if the text is angry or emphatic.",
        "Many of these come from en-AU Reddit, where blunt sincere complaints are common - do not mistake bluntness for sarcasm.",
        "",
        "For each item, reason in two steps before answering:",
        "  (1) Literal reading: what the words say on the surface.",
        "  (2) Intended reading: what the author actually means in context.",
        "If (1) and (2) conflict -> label 1 (Sarcastic). If they agree -> label 0 (Not Sarcastic).",
        "",
        "Worked examples:",
    ]
    for idx, row in examples_df.iterrows():
        lines += [
            f"Example {idx + 1}:",
            f"Variety: {row[VARIETY_COL]}",
            f'Text: "{safe(row[TEXT_COL])}"',
            f"Reasoning: {safe(row['manual_explanation_v2'])}",
            f"Label: {int(row['gold_label'])}",
            "",
        ]
    lines += [
        "Now classify the new item using the same two-step reasoning.",
        "Variety: {variety}",
        'Text: "{text}"',
        "Output format (exactly): one line `Label: 0` or `Label: 1`, then one short reasoning line.",
        "Label:",
    ]
    return "\n".join(lines)


def parse_prompt_label_v2(response_text: str):
    """Hardened parser: prefer an explicit `Label: x`, then fall back to word cues."""
    text = str(response_text).strip()
    m = re.search(r"label\s*[:\-]?\s*([01])", text, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    low = text.lower()
    if "not sarcastic" in low:
        return 0
    if "sarcastic" in low:
        return 1
    m2 = re.search(r"(^|\D)([01])(\D|$)", text)
    return int(m2.group(2)) if m2 else None


def main() -> None:
    if not (EXPLAINED_PATH.exists() and EVAL_PATH.exists()):
        print("Cached Q4 explained/held-out examples not found under outputs/tables/08_sarcasm_error_analysis/.")
        return

    explained = pd.read_csv(EXPLAINED_PATH).reset_index(drop=True)
    explained["manual_explanation_v2"] = explained.apply(linguistic_sarcasm_explanation_v2, axis=1)
    prompt_template = build_explanation_prompt_v2(explained)
    V2_PROMPT_PATH.write_text(prompt_template)
    print(f"Wrote revised v2 prompt: {V2_PROMPT_PATH}")
    print("\n--- v2 prompt preview ---")
    print(prompt_template[:1100] + "...\n")

    if "--write-prompt-only" in sys.argv:
        print("--write-prompt-only: v2 prompt written; skipping the GPU inference run.")
        return

    from src.qlora import LORA_MODEL, load_4bit_causal_lm
    import torch

    clear_vram(verbose=True)
    model, tokenizer = load_4bit_causal_lm(LORA_MODEL, device=device)

    if "--full" in sys.argv:
        _run_full(model, tokenizer, torch, prompt_template)
    else:
        _run_heldout6(model, tokenizer, torch, prompt_template)

    del model, tokenizer
    clear_vram(verbose=True)


def _generate_label(model, tokenizer, torch, prompt_template, variety, text):
    prompt = prompt_template.format(variety=variety, text=str(text).replace("\n", " "))
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=60, do_sample=False,
                             temperature=None, pad_token_id=tokenizer.eos_token_id)
    gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return parse_prompt_label_v2(gen), gen


def _run_heldout6(model, tokenizer, torch, prompt_template):
    eval_df = pd.read_csv(EVAL_PATH)
    rows = []
    for _, row in eval_df.iterrows():
        pred, gen = _generate_label(model, tokenizer, torch, prompt_template, row[VARIETY_COL], row[TEXT_COL])
        rows.append({
            "case_id": row["case_id"], TEXT_COL: row[TEXT_COL], VARIETY_COL: row[VARIETY_COL],
            "gold_label": int(row["gold_label"]), "gold_label_name": row["gold_label_name"],
            "adapter_prediction": int(row["predicted_label"]),
            "v2_prediction": pred, "v2_prediction_name": SARCASM_ID_TO_LABEL.get(pred, "Unparsed"),
            "v2_was_correct": pred == int(row["gold_label"]), "raw_qwen_response": gen,
        })
    results = pd.DataFrame(rows)
    results.to_csv(V2_RESULTS_PATH, index=False)
    v2_correct = int(results["v2_was_correct"].sum())
    orig = "2"
    if ORIGINAL_RESULTS_PATH.exists():
        orig = str(int(pd.read_csv(ORIGINAL_RESULTS_PATH)["fewshot_was_correct"].sum()))
    print(f"\nOriginal 4-shot prompt: {orig}/{len(results)} recovered")
    print(f"Revised v2 prompt:      {v2_correct}/{len(results)} recovered")
    print(f"Saved: {V2_RESULTS_PATH}")


def _run_full(model, tokenizer, torch, prompt_template):
    """Evaluate the v2 prompt on the adapter's FULL en-AU error set (trustworthy denominator)."""
    if not ADAPTER_PREDS_PATH.exists():
        print(f"Missing adapter prediction cache: {ADAPTER_PREDS_PATH}")
        return
    preds = pd.read_csv(ADAPTER_PREDS_PATH)
    errors = preds[preds["gold_label"] != preds["predicted_label"]].reset_index(drop=True)
    heldout = set(pd.read_csv(EVAL_PATH)["text"].astype(str)) if EVAL_PATH.exists() else set()
    print(f"\nEvaluating v2 prompt on {len(errors)} adapter errors (greedy)...")
    rows = []
    for i, row in errors.iterrows():
        pred, gen = _generate_label(model, tokenizer, torch, prompt_template, row["variety"], row["text"])
        gold = int(row["gold_label"])
        rows.append({
            "text": row["text"], "variety": row["variety"], "error_type": row["error_type"],
            "gold_label": gold, "gold_label_name": SARCASM_ID_TO_LABEL[gold],
            "adapter_prediction": int(row["predicted_label"]),
            "v2_prediction": pred, "v2_prediction_name": SARCASM_ID_TO_LABEL.get(pred, "Unparsed"),
            "v2_was_correct": pred == gold, "in_original_6": str(row["text"]) in heldout,
            "raw_qwen_response": gen,
        })
        if (i + 1) % 25 == 0:
            print(f"  ...{i + 1}/{len(errors)} done")
    results = pd.DataFrame(rows)
    results.to_csv(V2_FULL_RESULTS_PATH, index=False)

    total = len(results)
    rec = int(results["v2_was_correct"].sum())
    fp = results[results["error_type"].str.startswith("False Positive")]
    fn = results[results["error_type"].str.startswith("False Negative")]
    six = results[results["in_original_6"]]
    print("\n================= v2 FULL RESULTS =================")
    print(f"Full en-AU adapter error set: {rec}/{total} recovered ({100 * rec / total:.1f}%)")
    if len(fp):
        print(f"  False alarms fixed (gold=Not Sarcastic):   {int(fp['v2_was_correct'].sum())}/{len(fp)}")
    if len(fn):
        print(f"  Missed sarcasm recovered (gold=Sarcastic): {int(fn['v2_was_correct'].sum())}/{len(fn)}")
    if len(six):
        print(f"Original 6 held-out (v1=2/6): v2 = {int(six['v2_was_correct'].sum())}/{len(six)}")
    print(f"Saved: {V2_FULL_RESULTS_PATH}")


if __name__ == "__main__":
    main()
