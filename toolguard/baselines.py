"""Non-neural baselines: keyword/regex rules and TF-IDF + logistic regression."""
from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# Written from general knowledge of injection phrasing before looking at the test set.
# "password" is deliberately excluded because the test source (Gandalf) is about passwords.
PATTERNS = [
    r"\b(ignore|disregard|forget|skip|override)\b.{0,40}\b(previous|prior|above|earlier|all|preceding|your)\b.{0,30}\b(instructions?|prompts?|rules|directions|context|text)",
    r"\bnew (instructions?|task|rules)\b",
    r"\byou are now\b",
    r"\b(act|pretend|behave) (as|like|to be)\b",
    r"\bsystem prompt\b",
    r"\bjailbr(eak|oken)\b",
    r"\bdo anything now\b|\bDAN\b",
    r"\bdeveloper mode\b",
    r"\b(reveal|print|repeat|show|output) (your|the) (instructions|prompt|rules)\b",
    r"\bno (restrictions|limitations|rules|filters)\b",
]
RX = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in PATTERNS]


def regex_score(text: str) -> int:
    return sum(1 for r in RX if r.search(text))


def train_tfidf(texts, labels, seed=0):
    pipe = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced", random_state=seed),
    )
    pipe.fit(texts, labels)
    return pipe


def tfidf_score(pipe, texts):
    return np.asarray(pipe.predict_proba(texts)[:, 1])
