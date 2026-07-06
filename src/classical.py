"""Classical baselines: TF-IDF + LogReg and GloVe-mean + Linear SVM."""
from __future__ import annotations

import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

from .config import set_seed


class GloveTransformer(BaseEstimator, TransformerMixin):
    """Mean-pooled GloVe embeddings as a scikit-learn vectorizer step."""

    def __init__(self, glove_model=None, dim=100):
        self.glove_model = glove_model
        self.dim = dim

    def fit(self, X, y=None):
        if self.glove_model is None:
            try:
                import gensim.downloader as gensim_api
                print("Loading GloVe (glove-wiki-gigaword-100)...")
                self.glove_model = gensim_api.load("glove-wiki-gigaword-100")
                print("GloVe loaded")
            except Exception as e:
                print(f"GloVe load failed: {e}, using random embeddings as fallback")
                self.glove_model = None
        return self

    def transform(self, X):
        vectors = []
        for text in X:
            tokens = re.findall(r"[a-zA-Z]+(?:'[a-z]+)?", str(text).lower())
            if self.glove_model is not None:
                vecs = [self.glove_model[t] for t in tokens if t in self.glove_model]
                if vecs:
                    vectors.append(np.mean(vecs, axis=0))
                else:
                    vectors.append(np.zeros(self.dim))
            else:
                # Fallback: hash-based pseudo-embeddings
                vec = np.zeros(self.dim)
                for t in tokens:
                    np.random.seed(hash(t) % (2 ** 32))
                    vec += np.random.randn(self.dim) * 0.1
                if len(tokens) > 0:
                    vec /= len(tokens)
                vectors.append(vec)
        return np.array(vectors)


# Cache GloVe model globally to avoid reloading
_GLOVE_MODEL = None


def get_glove_model():
    global _GLOVE_MODEL
    if _GLOVE_MODEL is None:
        try:
            import gensim.downloader as gensim_api
            print("Loading GloVe model (one-time)...")
            _GLOVE_MODEL = gensim_api.load("glove-wiki-gigaword-100")
            print("GloVe loaded successfully")
        except Exception as e:
            print(f"GloVe unavailable: {e}")
            _GLOVE_MODEL = False
    return _GLOVE_MODEL if _GLOVE_MODEL else None


def build_classical_model(model_key, class_weight=None, seed=42):
    set_seed(seed)

    if model_key == "tfidf_logreg":
        return Pipeline([
            ("vec", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=2000, class_weight=class_weight, random_state=seed)),
        ])
    elif model_key == "glove_svm":
        return Pipeline([
            ("vec", GloveTransformer(glove_model=get_glove_model(), dim=100)),
            ("clf", LinearSVC(max_iter=3000, class_weight=class_weight, random_state=seed, dual=False)),
        ])
    else:
        raise ValueError(f"Unknown model: {model_key}")
