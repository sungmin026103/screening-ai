from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


@dataclass
class ScreeningResult:
    predictions: pd.DataFrame
    metrics: dict
    threshold: float
    pr_curve: dict = field(default_factory=dict)   # {"precision": [...], "recall": [...], "thresholds": [...]}
    roc_curve: dict = field(default_factory=dict)   # {"fpr": [...], "tpr": [...]}
    confusion: dict = field(default_factory=dict)   # {"tn":.., "fp":.., "fn":.., "tp":..}


def _find_col(df: pd.DataFrame, options: list[str]) -> str | None:
    lookup = {str(c).strip().lower(): c for c in df.columns}
    return next((lookup[x] for x in options if x in lookup), None)


def _normalize_label_value(value):
    """사용자 라벨을 0/1로 표준화한다. O/X, 숫자, 한글 판정을 모두 허용한다."""
    if pd.isna(value):
        return np.nan
    key = str(value).strip().lower()
    mapping = {
        "1": 1, "1.0": 1, "o": 1, "○": 1, "ㅇ": 1,
        "include": 1, "included": 1, "yes": 1, "y": 1, "포함": 1,
        "0": 0, "0.0": 0, "x": 0, "×": 0,
        "exclude": 0, "excluded": 0, "no": 0, "n": 0, "제외": 0,
    }
    return mapping.get(key, np.nan)


def _reviewer_label_columns(df: pd.DataFrame, excluded: set[str]) -> list[str]:
    """O/X 또는 0/1 값이 주로 들어 있는 검토자 열을 자동 탐지한다."""
    found = []
    for col in df.columns:
        if col in excluded:
            continue
        vals = df[col].dropna()
        if len(vals) == 0:
            continue
        normalized = vals.map(_normalize_label_value)
        valid_rate = float(normalized.notna().mean())
        if valid_rate >= 0.70 and normalized.notna().sum() >= 2:
            found.append(col)
    return found


def prepare_screening_data(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    title_col = _find_col(df, ["title", "제목"])
    abstract_col = _find_col(df, ["abstract", "초록"])
    label_col = _find_col(df, [
        "human_label", "human label", "include", "label", "decision",
        "포함", "라벨", "판정", "최종판정", "최종라벨",
    ])
    if not title_col:
        raise ValueError("제목(Title/제목) 열이 필요합니다.")

    out = pd.DataFrame()
    out["Title"] = df[title_col].fillna("").astype(str)
    out["Abstract"] = df[abstract_col].fillna("").astype(str) if abstract_col else ""

    if label_col:
        out["Human_Label"] = df[label_col].map(_normalize_label_value)
        label_source = str(label_col)
    else:
        excluded = {title_col}
        if abstract_col:
            excluded.add(abstract_col)
        reviewer_cols = _reviewer_label_columns(df, excluded)
        if not reviewer_cols:
            raise ValueError(
                "라벨 열을 찾지 못했습니다. Human_Label/Label 열을 추가하거나, "
                "검토자 열에 O(포함)와 X(제외)를 입력해 주세요."
            )
        normalized = pd.DataFrame({c: df[c].map(_normalize_label_value) for c in reviewer_cols})
        if len(reviewer_cols) == 1:
            out["Human_Label"] = normalized.iloc[:, 0]
        else:
            # 두 명 이상이 모두 같은 판정을 내린 행만 학습에 사용하고, 불일치는 미합의로 둔다.
            n_valid = normalized.notna().sum(axis=1)
            unanimous = normalized.nunique(axis=1, dropna=True).eq(1) & n_valid.ge(2)
            out["Human_Label"] = np.where(unanimous, normalized.bfill(axis=1).iloc[:, 0], np.nan)
        label_source = "검토자 자동합의: " + ", ".join(map(str, reviewer_cols))

    out["Human_Label"] = pd.to_numeric(out["Human_Label"], errors="coerce")
    out["Text"] = (out["Title"] + " " + out["Abstract"]).str.strip()
    return out, label_source


class CriteriaSimilarity(BaseEstimator, TransformerMixin):
    """코사인 유사도 기반 PICO/배제기준 근접도 피처.

    각 문헌 텍스트와 사용자가 입력한 PICO+배제기준 텍스트 사이의 TF-IDF
    코사인 유사도를 하나의 숫자 피처로 만든다. 반드시 Pipeline 안에 넣어
    fit/transform이 매 CV fold마다 train 텍스트로만 다시 이루어지도록 해야
    검증 fold의 텍스트가 IDF 통계에 새어 들어가지 않는다 (다른 프로젝트에서
    잡았던 것과 동일한 리키지 패턴).
    """

    def __init__(self, criteria_text: str = ""):
        self.criteria_text = criteria_text

    def fit(self, X, y=None):
        text = (self.criteria_text or "").strip()
        corpus = list(X) + [text if text else " "]
        self.vectorizer_ = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        self.vectorizer_.fit(corpus)
        self.criteria_vec_ = self.vectorizer_.transform([text if text else " "])
        return self

    def transform(self, X):
        doc_vecs = self.vectorizer_.transform(X)
        return cosine_similarity(doc_vecs, self.criteria_vec_)


def _build_pipeline(criteria_text: str = "") -> Pipeline:
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=50000, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=50000, sublinear_tf=True)
    transformers = [("word", word), ("char", char)]
    if criteria_text and criteria_text.strip():
        transformers.append(("criteria", CriteriaSimilarity(criteria_text=criteria_text)))
    features = FeatureUnion(transformers)
    base = LinearSVC(class_weight="balanced")
    model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    return Pipeline([("features", features), ("model", model)])


def _priority_labels(probabilities: np.ndarray, threshold: float, very_low_cutoff: float = 0.005) -> np.ndarray:
    """확률을 실제 검토 목적에 맞는 3단계로 구분한다."""
    cutoff = min(float(very_low_cutoff), max(float(threshold) * 0.95, 0.0))
    return np.select(
        [probabilities >= threshold, probabilities < cutoff],
        ["우선 검토", "매우 낮은 확률"],
        default="후순위 검토",
    )


def apply_recall_target(result: ScreeningResult, target_recall: float, very_low_cutoff: float = 0.005) -> ScreeningResult:
    """저장된 교차검증 확률을 사용해 재학습 없이 임계값과 화면 분류를 갱신한다."""
    updated = deepcopy(result)
    precision = np.asarray(updated.pr_curve.get("precision", []), dtype=float)
    recall = np.asarray(updated.pr_curve.get("recall", []), dtype=float)
    thresholds = np.asarray(updated.pr_curve.get("thresholds", []), dtype=float)
    if len(thresholds) == 0 or len(recall) < 2:
        return updated
    valid = np.where(recall[:-1] >= float(target_recall))[0]
    threshold = float(thresholds[valid[-1]]) if len(valid) else float(thresholds[0])
    updated.threshold = threshold

    pred_df = updated.predictions.copy()
    probs = pd.to_numeric(pred_df["AI_Probability"], errors="coerce").fillna(0).to_numpy()
    pred_df["AI_Recommendation"] = _priority_labels(probs, threshold, very_low_cutoff)

    if {"CV_Probability", "Human_Label_Normalized"}.issubset(pred_df.columns):
        mask = pred_df["CV_Probability"].notna() & pred_df["Human_Label_Normalized"].isin([0, 1])
        cv_probs = pd.to_numeric(pred_df.loc[mask, "CV_Probability"], errors="coerce").to_numpy()
        y = pred_df.loc[mask, "Human_Label_Normalized"].astype(int).to_numpy()
        cv_pred = (cv_probs >= threshold).astype(int)
        pred_df.loc[:, "CV_Prediction"] = np.nan
        pred_df.loc[mask, "CV_Prediction"] = cv_pred
        pred_df.loc[:, "False_Negative"] = False
        pred_df.loc[mask, "False_Negative"] = (y == 1) & (cv_pred == 0)
        tn, fp, fn, tp = confusion_matrix(y, cv_pred, labels=[0, 1]).ravel()
        rec = float(recall_score(y, cv_pred, zero_division=0))
        pre = float(precision_score(y, cv_pred, zero_division=0))
        updated.confusion = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
        updated.metrics.update({
            "recall": rec, "precision": pre,
            "accuracy": float(accuracy_score(y, cv_pred)),
            "f1": float(2 * pre * rec / (pre + rec)) if pre + rec > 0 else 0.0,
        })
    order = pd.Categorical(pred_df["AI_Recommendation"], ["우선 검토", "후순위 검토", "매우 낮은 확률"], ordered=True)
    pred_df = pred_df.assign(_priority_order=order).sort_values(
        ["_priority_order", "AI_Probability"], ascending=[True, False]
    ).drop(columns="_priority_order").reset_index(drop=True)
    updated.predictions = pred_df
    return updated


def train_and_predict(df: pd.DataFrame, target_recall: float = 0.95, criteria_text: str = "") -> ScreeningResult:
    data, _ = prepare_screening_data(df)
    labeled = data[data["Human_Label"].isin([0, 1])].copy()
    if len(labeled) < 20 or labeled["Human_Label"].nunique() < 2:
        raise ValueError("학습을 위해 Include와 Exclude가 모두 포함된 최소 20개 라벨이 필요합니다.")

    y = labeled["Human_Label"].astype(int).to_numpy()
    texts = labeled["Text"].to_numpy()

    min_class = int(labeled["Human_Label"].value_counts().min())
    folds = max(2, min(5, min_class))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    # 전체 파이프라인(TF-IDF + PICO 유사도 + 분류기)을 fold마다 처음부터 다시
    # 학습한다. train만으로 학습하기 때문에 검증 성능이 부풀려지지 않는다.
    probs = cross_val_predict(_build_pipeline(criteria_text), texts, y, cv=cv, method="predict_proba")[:, 1]

    precision, recall, thresholds = precision_recall_curve(y, probs)
    valid = np.where(recall[:-1] >= target_recall)[0]
    threshold = float(thresholds[valid[-1]]) if len(valid) else 0.5
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(y, probs)

    recall_v = float(recall_score(y, pred, zero_division=0))
    precision_v = float(precision_score(y, pred, zero_division=0))
    metrics = {
        "recall": recall_v,
        "precision": precision_v,
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(2 * precision_v * recall_v / (precision_v + recall_v)) if (precision_v + recall_v) > 0 else 0.0,
        "roc_auc": float(roc_auc_score(y, probs)),
        "average_precision": float(average_precision_score(y, probs)),
        "labeled_n": int(len(labeled)),
        "include_n": int(y.sum()),
    }

    final_pipeline = _build_pipeline(criteria_text)
    final_pipeline.fit(texts, y)
    all_probs = final_pipeline.predict_proba(data["Text"].to_numpy())[:, 1]
    result_df = df.copy().reset_index(drop=True)
    result_df["AI_Probability"] = all_probs
    result_df["AI_Probability_%"] = (all_probs * 100).round(2)
    result_df["Human_Label_Normalized"] = data["Human_Label"].to_numpy()
    result_df["CV_Probability"] = np.nan
    result_df.loc[labeled.index, "CV_Probability"] = probs
    result_df["CV_Prediction"] = np.nan
    result_df.loc[labeled.index, "CV_Prediction"] = pred
    result_df["False_Negative"] = False
    result_df.loc[labeled.index, "False_Negative"] = (y == 1) & (pred == 0)
    result_df["AI_Recommendation"] = _priority_labels(all_probs, threshold, 0.005)
    order = pd.Categorical(result_df["AI_Recommendation"], ["우선 검토", "후순위 검토", "매우 낮은 확률"], ordered=True)
    result_df = result_df.assign(_priority_order=order).sort_values(
        ["_priority_order", "AI_Probability"], ascending=[True, False]
    ).drop(columns="_priority_order").reset_index(drop=True)

    return ScreeningResult(
        predictions=result_df,
        metrics=metrics,
        threshold=threshold,
        pr_curve={"precision": precision.tolist(), "recall": recall.tolist(), "thresholds": thresholds.tolist()},
        roc_curve={"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        confusion={"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    )
