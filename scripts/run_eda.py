"""Section 2 - Dataset analysis (Q1.1 label distributions, Q1.2 vocabulary).

Regenerates the EDA figures and summaries from the dataset. No training.

Label distributions are shown split by SOURCE (Google vs Reddit), which is the
dataset's dominant confound: sarcasm is almost entirely a Reddit phenomenon and
sentiment polarity flips with source, so a by-variety-only view is misleading.
"""
from __future__ import annotations

from collections import Counter
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

import _bootstrap
from _bootstrap import show
from src import config
from src.config import (
    VARIETY_COL, SOURCE_COL, SENTIMENT_COL, SARCASM_COL, TEXT_COL, VARIETIES, SOURCES, FIGURES_DIR,
)
from src.registry import MASTER_RESULTS_PATH

SENTIMENT_NAMES = ["Negative", "Positive"]
SARCASM_NAMES = ["Not Sarcastic", "Sarcastic"]
SENTIMENT_COLORS = ["#d1495b", "#2e8b8b"]   # muted red / teal
SARCASM_COLORS = ["#b8bcc2", "#e8833a"]      # grey / orange


def _style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "axes.titleweight": "bold", "axes.titlesize": 14, "axes.labelsize": 12,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    })


def _stacked_distribution_by_source(df, label_col, label_names, colors, title, fname):
    """100%-stacked class proportions per variety, faceted Google | Reddit."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    for ax, source in zip(axes, SOURCES):
        sub = df[df[SOURCE_COL] == source]
        ct = (pd.crosstab(sub[VARIETY_COL], sub[label_col], normalize="index")
              .reindex(index=VARIETIES).reindex(columns=[0, 1], fill_value=0) * 100)
        counts = sub[VARIETY_COL].value_counts().reindex(VARIETIES).fillna(0).astype(int)
        x = np.arange(len(VARIETIES))
        ax.bar(x, ct[0], color=colors[0], label=label_names[0], edgecolor="white", linewidth=1.2)
        ax.bar(x, ct[1], bottom=ct[0], color=colors[1], label=label_names[1], edgecolor="white", linewidth=1.2)
        for i, v in enumerate(VARIETIES):
            pct = ct.loc[v, 1]
            y = min(ct.loc[v, 0] + ct.loc[v, 1] / 2, 94)
            ax.text(i, y, f"{pct:.1f}%", ha="center", va="center", fontsize=11,
                    fontweight="bold", color="white" if 8 < pct < 92 else "#222")
        ax.set_title(f"{source}", loc="left")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v}\n(n={counts[v]:,})" for v in VARIETIES])
        ax.set_ylim(0, 100)
        ax.set_ylabel("Share of texts (%)")
        ax.grid(axis="x", visible=False)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=colors[0], label=label_names[0]),
               Patch(facecolor=colors[1], label=label_names[1])]
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01), fontsize=12)
    fig.savefig(FIGURES_DIR / fname)
    plt.close(fig)
    print(f"  saved {fname}")


def _text_length_figure(df, fname):
    df = df.copy()
    df["word_count"] = df[TEXT_COL].astype(str).str.split().apply(len)
    cap = float(df["word_count"].quantile(0.95))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))

    sns.boxplot(data=df, x=VARIETY_COL, y="word_count", hue=SOURCE_COL, order=VARIETIES,
                showfliers=False, palette="Set2", ax=axes[0])
    axes[0].set_ylim(0, cap * 1.18)
    axes[0].set_title(f"Words per text by variety & source\n(outliers above the 95th pct, {cap:.0f} words, hidden)", fontsize=12)
    axes[0].set_xlabel(""); axes[0].set_ylabel("Word count")
    axes[0].legend(title="Source", loc="upper center", ncol=2, fontsize=10, title_fontsize=10)

    sns.ecdfplot(data=df, x="word_count", hue=VARIETY_COL, hue_order=VARIETIES, palette="Set1", ax=axes[1])
    axes[1].set_xlim(0, cap)
    axes[1].axhline(0.5, color="grey", ls="--", lw=1, alpha=0.6)
    axes[1].set_title("Cumulative distribution of length\n(median = 0.5 line)", fontsize=12)
    axes[1].set_xlabel("Word count"); axes[1].set_ylabel("Proportion of texts ≤ x")

    fig.suptitle("Text length distribution", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / fname)
    plt.close(fig)
    print(f"  saved {fname}")


def main() -> None:
    _style()
    df, _, _, _ = _bootstrap.load_all_splits()
    df["word_count"] = df[TEXT_COL].astype(str).str.split().apply(len)
    df["char_length"] = df[TEXT_COL].astype(str).str.len()

    # ---- Clean tabular summaries -------------------------------------------------
    print("\n" + "=" * 64 + "\nDATASET COMPOSITION\n" + "=" * 64)
    print("\nRows by variety x source:")
    show(pd.crosstab(df[VARIETY_COL], df[SOURCE_COL], margins=True, margins_name="Total"))

    print("\nSentiment - % Positive by variety x source:")
    show((df.groupby([VARIETY_COL, SOURCE_COL])[SENTIMENT_COL].mean() * 100)
         .round(1).unstack().reindex(VARIETIES))
    print("\nSarcasm - % Sarcastic by variety x source:")
    show((df.groupby([VARIETY_COL, SOURCE_COL])[SARCASM_COL].mean() * 100)
         .round(1).unstack().reindex(VARIETIES))
    print("\n-> Sarcasm is concentrated in Reddit; Google is overwhelmingly non-sarcastic & positive.")
    print("   Use Macro-F1 + class weighting throughout; treat source as a confound (see Q1.2 ablation).")

    # ---- Q1.1 label distributions (split by source) -----------------------------
    print("\n" + "=" * 64 + "\nQ1.1  LABEL DISTRIBUTIONS\n" + "=" * 64)
    _stacked_distribution_by_source(
        df, SENTIMENT_COL, SENTIMENT_NAMES, SENTIMENT_COLORS,
        "Sentiment distribution by variety and source", "01_sentiment_distribution.png")
    _stacked_distribution_by_source(
        df, SARCASM_COL, SARCASM_NAMES, SARCASM_COLORS,
        "Sarcasm distribution by variety and source", "02_sarcasm_distribution.png")

    # ---- Text length ------------------------------------------------------------
    print("\nText length (words) by variety:")
    show(df.groupby(VARIETY_COL)["word_count"].describe()[["mean", "50%", "75%", "max"]].round(1))
    print("Text length (words) by source:")
    show(df.groupby(SOURCE_COL)["word_count"].describe()[["mean", "50%", "75%", "max"]].round(1))
    _text_length_figure(df, "03_text_length.png")

    # ---- Q1.2 vocabulary overlap ------------------------------------------------
    print("\n" + "=" * 64 + "\nQ1.2  VOCABULARY OVERLAP\n" + "=" * 64)

    def simple_tokenize(text):
        return re.findall(r"[a-zA-Z]+(?:'[a-z]+)?", str(text).lower())

    def vocab(variety):
        s = set()
        for t in df[df[VARIETY_COL] == variety][TEXT_COL].astype(str):
            s.update(simple_tokenize(t))
        return s

    vocabularies = {v: vocab(v) for v in VARIETIES}
    for v in VARIETIES:
        print(f"  {v}: {len(vocabularies[v]):,} unique words")

    def jaccard(a, b):
        u = a | b
        return len(a & b) / len(u) if u else 0.0

    jac = pd.DataFrame([[jaccard(vocabularies[a], vocabularies[b]) for b in VARIETIES] for a in VARIETIES],
                       index=VARIETIES, columns=VARIETIES).round(3)
    show(jac)
    plt.figure(figsize=(6.5, 5.2))
    sns.heatmap(jac, annot=True, fmt=".3f", cmap="Blues", square=True, vmin=0,
                cbar_kws={"label": "Jaccard similarity"}, linewidths=0.5, linecolor="white")
    plt.title("Vocabulary overlap between varieties", fontweight="bold")
    plt.tight_layout(); plt.savefig(FIGURES_DIR / "04_jaccard_similarity.png"); plt.close()
    print("  saved 04_jaccard_similarity.png")

    # Variety-specific wordclouds
    counters = {v: Counter(tok for t in df[df[VARIETY_COL] == v][TEXT_COL].astype(str) for tok in simple_tokenize(t))
                for v in VARIETIES}

    def specific(target, top_n=60):
        tc = counters[target]
        other = Counter()
        for v in VARIETIES:
            if v != target:
                other.update(counters[v])
        rows = [{"word": w, "specificity": c / (other.get(w, 0) + 1)} for w, c in tc.items() if c >= 2]
        return pd.DataFrame(rows).nlargest(top_n, "specificity")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, v in zip(axes, VARIETIES):
        sp = specific(v)
        wc = WordCloud(width=420, height=320, background_color="white", colormap="viridis") \
            .generate_from_frequencies(dict(zip(sp["word"], sp["specificity"])))
        ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
        ax.set_title(f"{v} — variety-specific terms", fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "05_variety_wordclouds.png"); plt.close(fig)
    print("  saved 05_variety_wordclouds.png")

    # ---- Q1.2 domain-specificity ablation (cached; no training) -----------------
    if MASTER_RESULTS_PATH.exists():
        master = pd.read_csv(MASTER_RESULTS_PATH)
        q12 = master[master["model_key"].astype(str).isin(["roberta-large", "roberta-large_reddit"])
                     & master["task"].eq("sarcasm")].copy()
        if len(q12):
            summ = (q12.groupby(["model_key", "trained_on", "tested_on"], as_index=False)
                    .agg(seeds=("seed", "nunique"), macro_f1_mean=("test_macro_f1", "mean"),
                         macro_f1_std=("test_macro_f1", "std"), sarcastic_f1_mean=("test_f1_Sarcastic", "mean")).round(4))
            summ.insert(0, "regime", summ["trained_on"].map(
                {"pooled_all_varieties": "Pooled training", "reddit_only": "Reddit-only training"}))
            print("\nQ1.2 domain ablation - RoBERTa-large pooled vs Reddit-only (sarcasm test):")
            show(summ[["regime", "tested_on", "seeds", "macro_f1_mean", "macro_f1_std", "sarcastic_f1_mean"]])

    print(f"\nEDA complete. Figures written to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
