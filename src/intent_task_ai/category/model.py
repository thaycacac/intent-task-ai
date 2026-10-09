from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from intent_task_ai.category.map_intents import CATEGORIES


@dataclass
class CategoryPrediction:
    category: str
    confidence: float
    explanation: str


class CategoryModel:
    """TF-IDF + LogisticRegression category classifier (CPU-friendly MVP)."""

    def __init__(self, pipeline: Pipeline, version: str, labels: list[str]):
        self.pipeline = pipeline
        self.version = version
        self.labels = labels

    @classmethod
    def build(cls, version: str = "v1-tfidf-lr") -> CategoryModel:
        pipe = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=2,
                        max_features=50_000,
                        lowercase=True,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        solver="saga",
                    ),
                ),
            ]
        )
        return cls(pipeline=pipe, version=version, labels=list(CATEGORIES))

    def fit(self, texts: list[str], labels: list[str]) -> CategoryModel:
        self.pipeline.fit(texts, labels)
        self.labels = sorted(set(labels))
        return self

    def predict(self, text: str) -> CategoryPrediction:
        proba = self.pipeline.predict_proba([text])[0]
        classes = list(self.pipeline.classes_)
        best_i = int(proba.argmax())
        category = str(classes[best_i])
        confidence = float(proba[best_i])
        return CategoryPrediction(
            category=category,
            confidence=confidence,
            explanation=f"category_model={self.version}; top={category}:{confidence:.3f}",
        )

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        model_path = directory / "category_model.joblib"
        meta_path = directory / "category_meta.json"
        joblib.dump(self.pipeline, model_path)
        meta = {"version": self.version, "labels": self.labels, "backend": "sklearn-tfidf-lr"}
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return model_path

    @classmethod
    def load(cls, directory: Path) -> CategoryModel:
        meta = json.loads((directory / "category_meta.json").read_text(encoding="utf-8"))
        pipeline = joblib.load(directory / "category_model.joblib")
        return cls(
            pipeline=pipeline,
            version=meta.get("version", "unknown"),
            labels=list(meta.get("labels") or CATEGORIES),
        )


_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return _WHITESPACE.sub(" ", text.strip())


def majority_baseline_accuracy(labels: list[str]) -> float:
    if not labels:
        return 0.0
    from collections import Counter

    majority = Counter(labels).most_common(1)[0][0]
    return sum(1 for x in labels if x == majority) / len(labels)


def evaluate_predictions(
    y_true: list[str], y_pred: list[str]
) -> dict[str, Any]:
    from collections import Counter

    n = len(y_true)
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    accuracy = correct / n if n else 0.0
    baseline = majority_baseline_accuracy(y_true)
    per_class: dict[str, dict[str, float]] = {}
    for label in sorted(set(y_true) | set(y_pred)):
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == label and b == label)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != label and b == label)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == label and b != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        support = sum(1 for a in y_true if a == label)
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
    return {
        "n": n,
        "accuracy": round(accuracy, 4),
        "majority_baseline_accuracy": round(baseline, 4),
        "lift_over_majority": round(accuracy - baseline, 4),
        "label_distribution": dict(Counter(y_true)),
        "per_class": per_class,
    }
