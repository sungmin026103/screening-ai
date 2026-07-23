from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC


@dataclass
class ScreeningResult:
    predictions: pd.DataFrame
    metrics: dict
    threshold: float


def _find_col(df: pd.DataFrame, options: list[str]) -> str | None:
    lookup = {str(c).strip().lower(): c for c in df.columns}
    return next((lookup[x] for x in options if x in lookup), None)


def prepare_screening_data(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    title_col = _find_col(df, ["title", "제목"])
    abstract_col = _find_col(df, ["abstract", "초록"])
    label_col = _find_col(df, ["human_label", "include", "label", "decision", "포함", "라벨"])
    if not title_col or not label_col:
        raise ValueError("제목(Title/제목)과 라벨(Human_Label/Include/Label) 열이 필요합니다.")
    out = pd.DataFrame()
    out["Title"] = df[title_col].fillna("").astype(str)
    out["Abstract"] = df[abstract_col].fillna("").astype(str) if abstract_col else ""
    out["Human_Label"] = df[label_col]
    mapping = {"include": 1, "included": 1, "yes": 1, "y": 1, "o": 1, "1": 1,
               "exclude": 0, "excluded": 0, "no": 0, "n": 0, "x": 0, "0": 0}
    out["Human_Label"] = out["Human_Label"].map(lambda x: mapping.get(str(x).strip().lower(), x))
    out["Human_Label"] = pd.to_numeric(out["Human_Label"], errors="coerce")
    out["Text"] = (out["Title"] + " " + out["Abstract"]).str.strip()
    return out, label_col


def train_and_predict(df: pd.DataFrame, target_recall: float = 0.95) -> ScreeningResult:
    data, _ = prepare_screening_data(df)
    labeled = data[data["Human_Label"].isin([0, 1])].copy()
    if len(labeled) < 20 or labeled["Human_Label"].nunique() < 2:
        raise ValueError("학습을 위해 Include와 Exclude가 모두 포함된 최소 20개 라벨이 필요합니다.")

    y = labeled["Human_Label"].astype(int).to_numpy()
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=50000, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=50000, sublinear_tf=True)
    features = FeatureUnion([("word", word), ("char", char)])
    base = LinearSVC(class_weight="balanced")
    model = CalibratedClassifierCV(base, method="sigmoid", cv=3)

    min_class = int(labeled["Human_Label"].value_counts().min())
    folds = max(2, min(5, min_class))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    X = features.fit_transform(labeled["Text"])
    probs = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, probs)
    valid = np.where(recall[:-1] >= target_recall)[0]
    threshold = float(thresholds[valid[-1]]) if len(valid) else 0.5
    pred = (probs >= threshold).astype(int)
    metrics = {
        "recall": float(recall_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probs)),
        "average_precision": float(average_precision_score(y, probs)),
        "labeled_n": int(len(labeled)),
        "include_n": int(y.sum()),
    }

    model.fit(X, y)
    all_X = features.transform(data["Text"])
    all_probs = model.predict_proba(all_X)[:, 1]
    result = df.copy().reset_index(drop=True)
    result["AI_Probability"] = all_probs
    result["AI_Probability_%"] = (all_probs * 100).round(2)
    result["AI_Recommendation"] = np.where(all_probs >= threshold, "Include candidate", "Low probability")
    result = result.sort_values("AI_Probability", ascending=False).reset_index(drop=True)
    return ScreeningResult(result, metrics, threshold)
