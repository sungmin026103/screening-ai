from __future__ import annotations

import io
import re
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass, field
from copy import deepcopy

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.utils.dataframe import dataframe_to_rows
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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
from scipy.stats import beta as _beta_dist
from scipy.stats import t as _t_dist

# 문장 임베딩(의미 기반) 신호는 선택적 의존성이다. requirements.txt에
# sentence-transformers가 없거나 배포 환경에서 모델 다운로드가 막혀 있어도
# 앱 전체가 죽지 않도록 임포트 실패를 흡수하고, 사용 가능 여부를 플래그로 남긴다.
try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except Exception:  # pragma: no cover
    SentenceTransformer = None
    _HAS_SENTENCE_TRANSFORMERS = False

# 화면에 표시되는 3단계 우선순위 (정렬 순서 그대로 사용)
MANUAL_REVIEW_TIER = "초록 없음 · 수기 확인"
PRIORITY_ORDER = ["우선 검토", "경계 문헌", MANUAL_REVIEW_TIER, "안전 제외 후보"]
# 초록이 이보다 짧으면 제목만으로는 PECO 판정이 불가능한 레코드로 보고,
# 자동 제외에서 빼고 임계값 추정에서도 제외한 뒤 별도 수기 확인 더미로 보낸다.
ABSTRACT_MIN_CHARS = 30

# 지도학습(재현율 통계적 보장) 모드로 전환되는 최소 라벨 수. 이 미만이면 앱은
# 자동으로 zero-shot 모드로 동작한다 — 사람이 매번 "어느 모드로 할지" 고르는 게
# 아니라, 라벨 존재 여부라는 데이터 상태로 결정되는 고정 기준이다.
MIN_LABELS_FOR_SUPERVISED = 100
TRAINING_SAMPLE_SIZE = 200
MIN_INCLUDE_FOR_SUPERVISED = 10
VALIDATION_RECALL_TARGET = 0.95
VALIDATION_CONFIDENCE = 0.95
ALGORITHM_VERSION = "V36.0"
# 재현성: 같은 입력(코퍼스 + 라벨)이면 항상 같은 결과가 나와야 한다.
# 난수를 쓰는 모든 지점(폴드 분할, 캘리브레이션, SVM 좌표하강, 표본추출)에 이 seed를 건다.
RANDOM_SEED = 42
# 층화 추출: PICO 점수 순위 경계(상위 10%, 상위 40%)와 층별 표본 배분(합 1.0)
STRATUM_BOUNDS = (None, 0.40)  # High는 상위 n_high편 전수, Mid는 그 아래~상위 40%
# 적격 문헌은 PICO 순위 상단에 몰린다. 200편을 50/35/15로 나누면 상위 전수가 100편에 그쳐
# 유병률 1% 미만인 주제에서는 Include가 2~3편밖에 잡히지 않고, 그러면 교차검증 자체가 불가능하다.
# 상단 비중을 키워 '같은 라벨 노동으로 Include를 더 확보'하도록 기본값을 바꾼다.
# 하단 층 표본이 줄면 그 층의 가중치가 커져 분산이 늘지만, 애초에 Include가 없으면
# 분산을 논할 지표 자체가 만들어지지 않는다. 필요하면 plan_validation_extension으로 보강한다.
STRATUM_ALLOCATION = (0.70, 0.20, 0.10)

# 엑셀 다운로드 시 구간 순서와 배경색. False Negative는 실제 라벨이 Include인데
# 컷오프 밖으로 밀려난, 눈에 띄어야 하는 문헌이라 원래 버킷에서 따로 떼어내
# 독립된 구간으로 모은다 (행이 두 번 나오지 않도록 한 구간에만 배치한다).
EXPORT_GROUP_ORDER = ["우선 검토", "경계 문헌", MANUAL_REVIEW_TIER, "False Negative", "안전 제외 후보"]
EXPORT_GROUP_COLORS = {
    "우선 검토": "FFFFFF",       # 흰색
    "경계 문헌": "E9ECF1",       # 중간 회색
    MANUAL_REVIEW_TIER: "FFF2CC",  # 노란색 — 텍스트가 없어 자동 판정 불가
    "False Negative": "F7D6D2",  # 경고용 붉은색 (기존 FN 탭과 동일 색)
    "안전 제외 후보": "B8BDC6",  # 진한 회색
}


@dataclass
class ScreeningResult:
    predictions: pd.DataFrame
    metrics: dict
    threshold: float
    pr_curve: dict = field(default_factory=dict)   # {"precision": [...], "recall": [...], "thresholds": [...]} (참고용 차트)
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
    """Human_Label을 표준화한다.

    - 라벨 열이 하나면 O/o/○/ㅇ = Include(1), X/x/× = Exclude(0)로 인식한다.
    - 검토자 열이 2개 이상 자동 탐지되면, 모두 Include로 일치할 때만 1,
      모두 Exclude로 일치할 때만 0으로 두고, 불일치(의견 불일치)는 NaN으로
      두어 학습에서 제외한다 (합의되지 않은 판정을 모델이 배우지 않도록).
    """
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
    out["StructuredText"] = [_serialize_structured(t, a) for t, a in zip(out["Title"], out["Abstract"])]
    return out, label_source


def detect_label_count(df: pd.DataFrame) -> int:
    """업로드된 파일에 유효 라벨(Include/Exclude 둘 다 있는)이 몇 개 있는지 반환한다.
    라벨 열이 아예 없거나 형식이 안 맞으면 0을 반환한다 (예외를 던지지 않음).
    app.py가 이 값으로 지도학습/zero-shot 모드를 자동 판별하는 데 쓴다."""
    try:
        data, _ = prepare_screening_data(df)
        return int(data["Human_Label"].isin([0, 1]).sum())
    except Exception:
        return 0


def _stable_record_id(source_index: int, title: str) -> str:
    """Human-validation 파일에서 행을 안전하게 다시 연결하기 위한 안정적 ID."""
    payload = f"{int(source_index)}|{str(title).strip().casefold()}".encode("utf-8", errors="ignore")
    return hashlib.sha256(payload).hexdigest()[:20]


def _validation_set_id(record_ids: list[str]) -> str:
    payload = "|".join(sorted(map(str, record_ids))).encode("utf-8")
    return "VAL-" + hashlib.sha256(payload).hexdigest()[:16].upper()


def build_training_sample(
    df: pd.DataFrame,
    criteria_text: str,
    exclusion_text: str = "",
    sample_size: int = TRAINING_SAMPLE_SIZE,
    random_state: int = 42,
    seed_texts: list[str] | None = None,
) -> pd.DataFrame:
    """전체 코퍼스에서 1회성 human-validation 표본을 만든다.

    기본 200편 = High 100편(상위 PICO 적합도 전수) + Mid 70편 + Low 30편(층화 무작위).
    High는 certainty stratum이라 weight=1이며, Mid/Low는 역추출확률 가중치(IPW)를 저장한다.
    이 200편은 모델 학습과 out-of-fold 내부 검증에 동시에 사용되며 추가 라벨링은 필수가 아니다.
    """
    if not criteria_text or not criteria_text.strip():
        raise ValueError("검증용 문헌을 선정하려면 PICO/PECO 기준이 필요합니다.")
    if len(df) == 0:
        raise ValueError("문헌 데이터가 비어 있습니다.")

    base = df.copy().reset_index(drop=True)
    title_col = _find_col(base, ["title", "제목"])
    abstract_col = _find_col(base, ["abstract", "초록"])
    if title_col is None:
        raise ValueError("제목(Title/제목) 열이 필요합니다.")
    titles = base[title_col].fillna("").astype(str)
    abstracts = base[abstract_col].fillna("").astype(str) if abstract_col else pd.Series([""] * len(base))
    docs = (titles + " " + abstracts).str.strip().tolist()
    if not any(x.strip() for x in docs):
        raise ValueError("Title/Abstract 텍스트가 비어 있어 표본을 선정할 수 없습니다.")
    base["_Source_Index"] = np.arange(len(base), dtype=int)

    sections = _parse_pico_sections(criteria_text)
    queries = [q.strip() for q in sections.values() if q and q.strip()]
    if not queries:
        queries = [criteria_text.strip()]
    exclusion_items = _split_bullet_items(exclusion_text)
    all_queries = queries + exclusion_items

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=60000,
                          sublinear_tf=True, stop_words="english")
    mat = vec.fit_transform(docs + all_queries)
    doc_mat = mat[:len(docs)]
    query_mat = mat[len(docs):]
    pico_q = query_mat[:len(queries)]
    pico_sim = cosine_similarity(doc_mat, pico_q)
    pico_score = 0.65 * pico_sim.mean(axis=1) + 0.35 * pico_sim.min(axis=1)
    if exclusion_items:
        excl_q = query_mat[len(queries):]
        excl_score = cosine_similarity(doc_mat, excl_q).max(axis=1)
    else:
        excl_score = np.zeros(len(docs), dtype=float)
    scores = np.asarray(pico_score - 0.75 * excl_score, dtype=float)

    # 이미 적격임을 아는 문헌(seed)이 있으면 그 문헌과의 유사도를 순위에 섞는다.
    # PICO 문장은 연구자가 쓴 '기준'이고 seed는 실제 적격 '문헌'이라, 적격 문헌을 끌어올리는
    # 힘이 훨씬 세다. 유병률이 1% 미만인 주제에서 Include를 먼저 확보하는 가장 싼 방법이다.
    # seed는 PECO 기준을 바꾸지 않는다 — 어떤 문헌을 '먼저 읽을지'만 바꾸며,
    # 층별 가중치(N/n)가 그대로 적용되므로 추정의 불편성은 유지된다.
    if seed_texts:
        seeds = [str(t).strip() for t in seed_texts if str(t).strip()]
        if seeds:
            seed_mat = vec.transform(seeds)
            seed_sim = cosine_similarity(doc_mat, seed_mat).max(axis=1)
            def _z(v):
                v = np.asarray(v, dtype=float)
                sd = v.std()
                return (v - v.mean()) / sd if sd > 0 else np.zeros_like(v)
            scores = 0.5 * _z(scores) + 0.5 * _z(seed_sim)

    n_total = len(base)
    n = min(int(sample_size), n_total)
    order = np.argsort(-scores, kind="stable")
    rng = np.random.default_rng(random_state)

    # 코퍼스가 200편 이하라면 표본추출이 아니라 전수 validation이다.
    if n == n_total:
        selected = [(int(i), "Census", 1.0, 1.0, n_total, n_total) for i in order]
    else:
        alloc = [int(round(n * a)) for a in STRATUM_ALLOCATION]
        alloc[-1] = n - sum(alloc[:-1])
        high_n = min(alloc[0], n_total)
        cut_mid = max(high_n, min(n_total, int(np.ceil(n_total * STRATUM_BOUNDS[1]))))
        high_pool = order[:high_n]
        mid_pool = order[high_n:cut_mid]
        low_pool = order[cut_mid:]

        # 보통 N>>200이면 정확히 100/70/30. 작은 층이 부족한 예외에서는 남는 표본을 다른 층으로 이동.
        mid_take = min(alloc[1], len(mid_pool))
        low_take = min(alloc[2], len(low_pool))
        deficit = n - (len(high_pool) + mid_take + low_take)
        if deficit > 0:
            add_mid = min(deficit, len(mid_pool) - mid_take)
            mid_take += add_mid; deficit -= add_mid
        if deficit > 0:
            add_low = min(deficit, len(low_pool) - low_take)
            low_take += add_low; deficit -= add_low
        if deficit > 0:
            # 이 경우는 사실상 n_total<=n에 가까운 경계 상황. 남은 행을 순위대로 채운다.
            remaining = [i for i in order if i not in set(high_pool.tolist())]
            chosen = set()
            mid_pick = list(rng.choice(mid_pool, size=mid_take, replace=False)) if mid_take else []
            low_pick = list(rng.choice(low_pool, size=low_take, replace=False)) if low_take else []
            chosen.update(map(int, mid_pick + low_pick))
            extra = [int(i) for i in remaining if int(i) not in chosen][:deficit]
        else:
            mid_pick = list(rng.choice(mid_pool, size=mid_take, replace=False)) if mid_take else []
            low_pick = list(rng.choice(low_pool, size=low_take, replace=False)) if low_take else []
            extra = []

        selected = []
        # High는 정의상 top high_n 전수이므로 inclusion probability=1.
        for i in high_pool:
            selected.append((int(i), "High PICO relevance", 1.0, 1.0, len(high_pool), len(high_pool)))
        if mid_take:
            w = len(mid_pool) / mid_take
            pi = mid_take / len(mid_pool)
            selected.extend((int(i), "Mid PICO relevance", w, pi, len(mid_pool), mid_take) for i in mid_pick)
        if low_take:
            w = len(low_pool) / low_take
            pi = low_take / len(low_pool)
            selected.extend((int(i), "Low PICO relevance", w, pi, len(low_pool), low_take) for i in low_pick)
        # extra는 매우 드문 경계 fallback이며 보수적으로 weight=1 처리하고 별도 표기한다.
        selected.extend((int(i), "Fallback", 1.0, 1.0, len(extra), len(extra)) for i in extra)

    rows = []
    for idx, stratum, weight, pi, stratum_n, sampled_n in selected[:n]:
        row = base.iloc[idx].copy()
        row["Training_Stratum"] = stratum
        row["Sampling_Weight"] = round(float(weight), 8)
        row["Sampling_Probability"] = round(float(pi), 8)
        row["Sampling_Stratum_N"] = int(stratum_n)
        row["Sampling_Stratum_n"] = int(sampled_n)
        row["Validation_Record_ID"] = _stable_record_id(idx, titles.iloc[idx])
        rows.append(row)

    out = pd.DataFrame(rows).reset_index(drop=True)
    out.insert(0, "Training_No", np.arange(1, len(out) + 1))
    set_id = _validation_set_id(out["Validation_Record_ID"].astype(str).tolist())
    out.insert(1, "Validation_Set_ID", set_id)
    out["Human_Label"] = ""
    return out


def validate_human_validation_file(expected_sample_df: pd.DataFrame, uploaded_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """다운로드한 validation 표본과 업로드된 O/X 파일의 무결성과 완전성을 검사한다.

    Sampling_Weight/stratum 등 설계 메타데이터는 업로드 파일을 신뢰하지 않고 앱이 보관한
    expected_sample_df에서 다시 가져온다. 사용자는 Human_Label만 수정할 수 있다.
    """
    expected = expected_sample_df.copy().reset_index(drop=True)
    uploaded = uploaded_df.copy().reset_index(drop=True)
    if expected.empty:
        raise ValueError("앱에 저장된 validation 표본이 없습니다. 200편 표본을 다시 생성해 주세요.")
    if "Human_Label" not in uploaded.columns:
        # prepare_screening_data가 인식하는 별칭을 허용하되 최종적으로 Human_Label로 복사한다.
        label_col = _find_col(uploaded, ["human_label", "label", "decision", "판정", "라벨"])
        if label_col is None:
            raise ValueError("Human_Label 열을 찾지 못했습니다. 다운로드한 파일의 열 이름을 유지해 주세요.")
        uploaded["Human_Label"] = uploaded[label_col]

    key = "Validation_Record_ID" if "Validation_Record_ID" in expected.columns and "Validation_Record_ID" in uploaded.columns else "_Source_Index"
    if key not in expected.columns or key not in uploaded.columns:
        raise ValueError("Validation_Record_ID 또는 _Source_Index가 없어 원본 200편과 안전하게 대조할 수 없습니다.")
    if uploaded[key].duplicated().any():
        raise ValueError(f"업로드 파일의 {key}에 중복 행이 있습니다. 원본 validation 파일을 사용해 주세요.")

    exp_keys = expected[key].astype(str)
    up_keys = uploaded[key].astype(str)
    missing = sorted(set(exp_keys) - set(up_keys))
    extra = sorted(set(up_keys) - set(exp_keys))
    if missing or extra:
        raise ValueError(
            f"업로드 파일이 생성된 validation 표본과 일치하지 않습니다. 누락 {len(missing)}편, 추가 {len(extra)}편입니다. "
            "처음 다운로드한 파일에서 Human_Label 열만 수정해 주세요."
        )

    label_map = dict(zip(up_keys, uploaded["Human_Label"]))
    canonical = expected.copy()
    canonical["Human_Label"] = canonical[key].astype(str).map(label_map)
    normalized = canonical["Human_Label"].map(_normalize_label_value)
    invalid_mask = ~normalized.isin([0, 1])
    if invalid_mask.any():
        bad_n = int(invalid_mask.sum())
        bad_rows = canonical.loc[invalid_mask, "Training_No"].head(10).astype(str).tolist() if "Training_No" in canonical.columns else []
        detail = f" (예: {', '.join(bad_rows)})" if bad_rows else ""
        raise ValueError(f"200편 모두 O 또는 X로 판정해야 합니다. 미판정/잘못된 값 {bad_n}편{detail}")

    canonical["Human_Label"] = normalized.astype(int)
    include_n = int((canonical["Human_Label"] == 1).sum())
    exclude_n = int((canonical["Human_Label"] == 0).sum())
    return canonical, {
        "labeled_n": int(len(canonical)),
        "expected_n": int(len(expected)),
        "include_n": include_n,
        "exclude_n": exclude_n,
        "complete": True,
        "validation_set_id": str(expected.get("Validation_Set_ID", pd.Series([""])).iloc[0]) if len(expected) else "",
    }


def merge_training_labels(
    full_df: pd.DataFrame,
    labeled_sample_df: pd.DataFrame,
    expected_sample_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Human-validation O/X를 전체 코퍼스에 병합한다.

    V30에서는 expected_sample_df가 주어지면 200/200 완전 라벨링과 표본 ID 무결성을 강제하고,
    Sampling_Weight는 앱이 원래 생성한 표본에서만 가져온다. 하위 호환을 위해 None도 허용한다.
    """
    full = full_df.copy().reset_index(drop=True)
    if expected_sample_df is not None:
        sample, stats = validate_human_validation_file(expected_sample_df, labeled_sample_df)
    else:
        sample = labeled_sample_df.copy().reset_index(drop=True)
        sample_prepared, _ = prepare_screening_data(sample)
        sample["Human_Label"] = sample_prepared["Human_Label"]
        valid_tmp = sample["Human_Label"].isin([0, 1])
        stats = {
            "labeled_n": int(valid_tmp.sum()),
            "expected_n": int(len(sample)),
            "include_n": int((sample.loc[valid_tmp, "Human_Label"] == 1).sum()),
            "exclude_n": int((sample.loc[valid_tmp, "Human_Label"] == 0).sum()),
            "complete": bool(valid_tmp.all()),
            "validation_set_id": "",
        }

    full["Human_Label"] = np.nan
    full["Sampling_Weight"] = np.nan
    full["Training_Stratum"] = pd.Series([""] * len(full), dtype="object")
    full["Validation_Record_ID"] = pd.Series([""] * len(full), dtype="object")
    matched = 0

    if "_Source_Index" in sample.columns:
        for _, row in sample.iterrows():
            lab = _normalize_label_value(row.get("Human_Label"))
            if lab not in (0, 1):
                continue
            try:
                idx = int(row["_Source_Index"])
            except Exception:
                continue
            if 0 <= idx < len(full):
                full.loc[idx, "Human_Label"] = int(lab)
                full.loc[idx, "Sampling_Weight"] = pd.to_numeric(row.get("Sampling_Weight", 1.0), errors="coerce")
                full.loc[idx, "Training_Stratum"] = row.get("Training_Stratum", "")
                full.loc[idx, "Validation_Record_ID"] = row.get("Validation_Record_ID", "")
                matched += 1
    else:
        title_full = _find_col(full, ["title", "제목"])
        title_sample = _find_col(sample, ["title", "제목"])
        if not title_full or not title_sample:
            raise ValueError("라벨 파일을 병합하려면 _Source_Index 또는 Title/제목 열이 필요합니다.")
        lookup = {}
        for idx, t in enumerate(full[title_full].fillna("").astype(str)):
            lookup.setdefault(t.strip().casefold(), idx)
        for _, row in sample.iterrows():
            lab = _normalize_label_value(row.get("Human_Label"))
            if lab not in (0, 1):
                continue
            key_title = str(row[title_sample]).strip().casefold()
            idx = lookup.get(key_title)
            if idx is not None:
                full.loc[idx, "Human_Label"] = int(lab)
                full.loc[idx, "Sampling_Weight"] = pd.to_numeric(row.get("Sampling_Weight", 1.0), errors="coerce")
                full.loc[idx, "Training_Stratum"] = row.get("Training_Stratum", "")
                full.loc[idx, "Validation_Record_ID"] = row.get("Validation_Record_ID", "")
                matched += 1

    valid = full["Human_Label"].isin([0, 1])
    stats = dict(stats)
    stats.update({
        "matched": int(matched),
        "include_n": int((full.loc[valid, "Human_Label"] == 1).sum()),
        "exclude_n": int((full.loc[valid, "Human_Label"] == 0).sum()),
        "labeled_n": int(valid.sum()),
    })
    if expected_sample_df is not None and matched != len(expected_sample_df):
        raise ValueError(f"전체 코퍼스와 validation 표본 매칭에 실패했습니다: {matched}/{len(expected_sample_df)}편")
    return full, stats


# ---------------------------------------------------------------------------
# 의미 기반(임베딩) 신호. TF-IDF 계열(word/char/PICO 유사도)은 결국 전부
# 표면적 단어 일치에 의존하기 때문에 "서로 다른 근거"라고 보기 어렵다.
# 사전학습된 문장 임베딩(SBERT 계열)은 동의어·다른 표현으로 쓰인 문헌도
# 의미로 포착하므로, TF-IDF 신호들과 상관관계가 낮은 진짜 독립적인 근거가 된다.
# 또한 이 인코더는 우리 데이터로 다시 학습(fit)되지 않는 고정 가중치이므로,
# 전체 문헌에 대해 미리 한 번만 계산해도 검증 fold 누수가 전혀 발생하지 않는다
# (TF-IDF는 데이터 의존적으로 fit되므로 fold마다 다시 학습해야 하는 것과 대조적).
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"  # 생의학 초록 특화. 배포 환경 리소스가 빠듯하면
# "sentence-transformers/all-MiniLM-L6-v2" (가볍고 범용, CPU에서 훨씬 빠름)로 교체 가능.

_embed_model_singleton: dict[str, "SentenceTransformer"] = {}


def embeddings_available() -> bool:
    return _HAS_SENTENCE_TRANSFORMERS


def _get_embed_model():
    if EMBEDDING_MODEL_NAME not in _embed_model_singleton:
        _embed_model_singleton[EMBEDDING_MODEL_NAME] = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embed_model_singleton[EMBEDDING_MODEL_NAME]


def build_embedding_lookup(all_texts: np.ndarray) -> dict[str, np.ndarray] | None:
    """전체 문헌 텍스트에 대해 임베딩을 한 번만 계산해 {텍스트: 벡터} 딕셔너리로 반환한다.
    사전학습 인코더가 이 데이터로 학습되는 게 아니므로, CV fold 밖에서 한 번만
    계산해도 안전하다 (TF-IDF 벡터라이저처럼 fold마다 다시 fit할 필요가 없음)."""
    if not _HAS_SENTENCE_TRANSFORMERS:
        return None
    model = _get_embed_model()
    unique_texts = list(dict.fromkeys(all_texts.tolist()))  # 중복 제거, 순서 보존
    vectors = model.encode(unique_texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    return {t: v for t, v in zip(unique_texts, vectors)}


class EmbeddingLookup(BaseEstimator, TransformerMixin):
    """사전 계산된 임베딩을 텍스트로 조회만 하는 변환기. fit에서 아무것도 학습하지
    않으므로(고정 가중치 인코더), Pipeline 안에 있어도 매 fold 재계산이 필요 없다."""

    def __init__(self, lookup: dict[str, np.ndarray] | None = None, dim: int = 768):
        self.lookup = lookup or {}
        self.dim = dim

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not self.lookup:
            return np.zeros((len(X), self.dim))
        any_vec = next(iter(self.lookup.values()))
        dim = any_vec.shape[0]
        return np.vstack([self.lookup.get(t, np.zeros(dim)) for t in X])



STRUCTURED_SEP = "\n<ABSTRACT>\n"


def _serialize_structured(title: str, abstract: str) -> str:
    return f"{str(title).strip()}{STRUCTURED_SEP}{str(abstract).strip()}"


class TextPartExtractor(BaseEstimator, TransformerMixin):
    """StructuredText에서 Title 또는 Abstract만 분리해 별도 TF-IDF가 학습되도록 한다."""
    def __init__(self, part: str = "title"):
        self.part = part

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = []
        for x in X:
            text = str(x)
            if STRUCTURED_SEP in text:
                title, abstract = text.split(STRUCTURED_SEP, 1)
            else:
                title, abstract = text, ""
            value = title if self.part == "title" else abstract
            # Title-only screening files are valid. If an entire CV fold has no abstracts,
            # sklearn's TfidfVectorizer would otherwise raise
            # "empty vocabulary; perhaps the documents only contain stop words".
            # A constant placeholder keeps the feature branch structurally valid while
            # contributing no discriminative information.
            if not str(value).strip():
                value = "missing_abstract" if self.part == "abstract" else "missing_title"
            out.append(value)
        return np.asarray(out, dtype=object)


class RuleSignalFeatures(BaseEstimator, TransformerMixin):
    """초록에서 명백한 연구설계/비대상 신호를 숫자 feature로 변환한다.
    단일 키워드만으로 자동 배제하지 않고 최종 분류기의 보조 feature로만 사용한다."""
    # V36: 주제 특이적 토큰(c2c12, myoblast, arabidopsis, observational 등)을 제거하고
    # 어떤 SR에도 공통인 '연구 형태' 신호만 남긴다. 값은 분류기의 보조 feature일 뿐이며
    # 가중치는 매 리뷰의 라벨로 학습된다(제외 규칙이 아님).
    PATTERNS = [
        r"\bin\s*vitro\b|cell\s+culture|cultured\s+cells?|cell\s+lines?\b",
        r"systematic\s+review|meta[- ]analysis|narrative\s+review|scoping\s+review|literature\s+review",
        r"study\s+protocol|trial\s+protocol|protocol\s+for\s+a",
        r"case\s+report|case\s+series",
        r"editorial|commentary|letter\s+to\s+the\s+editor|erratum|corrigendum",
    ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for x in X:
            t = str(x).lower()
            vals = [1.0 if re.search(p, t, flags=re.I) else 0.0 for p in self.PATTERNS]
            vals.append(float(sum(vals)))
            rows.append(vals)
        return np.asarray(rows, dtype=float)


class PrototypeSimilarity(BaseEstimator, TransformerMixin):
    """훈련 fold의 Include/Exclude 문헌 centroid와의 유사도를 계산한다.
    fold 안에서만 prototype을 만들기 때문에 교차검증 누수가 없다."""
    def __init__(self, max_features: int = 25000):
        self.max_features = max_features

    def fit(self, X, y=None):
        self.vectorizer_ = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=self.max_features, sublinear_tf=True)
        mat = self.vectorizer_.fit_transform(list(X))
        y = np.asarray(y) if y is not None else np.zeros(mat.shape[0], dtype=int)
        self.pos_ = np.asarray(mat[y == 1].mean(axis=0) if np.any(y == 1) else mat.mean(axis=0))
        self.neg_ = np.asarray(mat[y == 0].mean(axis=0) if np.any(y == 0) else mat.mean(axis=0))
        return self

    def transform(self, X):
        mat = self.vectorizer_.transform(list(X))
        pos = cosine_similarity(mat, self.pos_).ravel()
        neg = cosine_similarity(mat, self.neg_).ravel()
        return np.column_stack([pos, neg, pos - neg])


class NumericLookup(BaseEstimator, TransformerMixin):
    """사전 계산된 고정 numeric features를 StructuredText key로 조회한다."""
    def __init__(self, lookup=None, dim: int = 4):
        self.lookup = lookup or {}
        self.dim = dim

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not self.lookup:
            return np.zeros((len(X), self.dim), dtype=float)
        any_vec = np.asarray(next(iter(self.lookup.values())), dtype=float)
        dim = int(any_vec.size)
        return np.vstack([np.asarray(self.lookup.get(str(x), np.zeros(dim)), dtype=float) for x in X])

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


# ---------------------------------------------------------------------------
# PICO 섹션 파서. 사용자가 기준 텍스트를 'P: ...' / 'I: ...' / 'C: ...' / 'O: ...'
# (또는 한글 '대상:'/'중재:'/'대조:'/'결과:') 형식으로 쓰면 항목별로 분리한다.
# 인식되는 접두어가 하나도 없으면 전체를 "criteria"라는 단일 키로 반환해
# 기존(통짜 유사도) 동작과 완전히 호환된다. LLM 호출 없이 문자열 파싱만 사용.
# ---------------------------------------------------------------------------
_PICO_PREFIXES = [
    ("P", ["p:", "population:", "patient:", "대상:", "환자:", "인구집단:"]),
    ("I", ["i:", "intervention:", "중재:", "노출:"]),
    ("C", ["c:", "comparison:", "comparator:", "대조:", "비교:"]),
    ("O", ["o:", "outcome:", "결과:", "결과지표:"]),
]


def _parse_pico_sections(criteria_text: str) -> dict[str, str]:
    text = (criteria_text or "").strip()
    if not text:
        return {"criteria": ""}
    sections: dict[str, list[str]] = {}
    current = None
    matched_any = False
    for line in text.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        found_key, remainder = None, stripped
        for key, prefixes in _PICO_PREFIXES:
            for p in prefixes:
                if lowered.startswith(p):
                    found_key = key
                    remainder = stripped[len(p):].strip()
                    break
            if found_key:
                break
        if found_key:
            matched_any = True
            current = found_key
            sections.setdefault(current, []).append(remainder)
        elif current is not None and stripped:
            sections[current].append(stripped)
    if not matched_any:
        return {"criteria": text}
    return {k: " ".join(v).strip() for k, v in sections.items() if " ".join(v).strip()}


# ---------------------------------------------------------------------------
# Zero-shot 스크리닝: 라벨링된 문헌이 하나도 없는 프로젝트 초기 단계를 위한 모드.
# 지도학습 파이프라인(train_and_predict)은 최소 20개 이상의 라벨이 있어야 재현율을
# 통계적으로 보장할 수 있다. 이 함수는 PICO 기준 + 배제기준 텍스트만으로 모든
# 문헌에 순위를 매긴다 — 다만 정답(라벨)이 전혀 없으므로 "재현율 X% 보장" 같은
# 통계적 검증은 원리적으로 불가능하다는 점이 지도학습 모드와의 근본적 차이다.
#
# 대신 다음 두 가지로 최대한 원칙 있게 동작하게 했다:
#   1) PICO 각 항목(P/I/C/O)·배제기준 각 항목에 대한 의미 유사도를 따로 계산
#      (뭉뚱그린 유사도 하나보다 어느 항목이 안 맞는지 설명 가능)
#   2) Otsu 임계값 방법(영상 이진화 기법을 1차원 점수 분포에 적용)으로 임의의
#      퍼센트/임계값을 사용자가 고르지 않아도 "안전 제외 / 경계 / 우선 검토"
#      3단계를 점수 분포 스스로 자연스럽게 나누게 한다 — 이번 요청에서 강조하신
#      "매번 선택하게 하지 말고 고정" 원칙을 그대로 따른 것.
#
# 중요: 안전 제외 후보 중 일부(예: 20~30편)를 무작위로 뽑아 사람이 직접 확인하고,
# 그 판정을 Human_Label로 입력해 지도학습 모드로 넘어가는 것을 강력히 권장한다.
# 그래야 비로소 재현율 보장이 통계적으로 성립한다.
# ---------------------------------------------------------------------------

ZERO_SHOT_DISCLAIMER = (
    "Zero-shot 모드는 라벨(정답) 없이 PICO/배제기준 텍스트 유사도만으로 순위를 매깁니다. "
    "재현율 목표를 통계적으로 보장하지 않습니다. 안전 제외 후보 중 일부를 무작위로 "
    "직접 확인한 뒤 그 판정을 라벨로 입력하면, 지도학습 모드(train_and_predict)로 넘어가 "
    "통계적으로 검증된 재현율 보장을 받을 수 있습니다."
)


def prepare_unlabeled_data(df: pd.DataFrame) -> pd.DataFrame:
    """라벨 열이 전혀 없어도 되는 zero-shot 전용 데이터 준비 함수.
    prepare_screening_data와 달리 라벨 열을 요구하지 않는다."""
    title_col = _find_col(df, ["title", "제목"])
    abstract_col = _find_col(df, ["abstract", "초록"])
    if not title_col:
        raise ValueError("제목(Title/제목) 열이 필요합니다.")
    out = pd.DataFrame()
    out["Title"] = df[title_col].fillna("").astype(str)
    out["Abstract"] = df[abstract_col].fillna("").astype(str) if abstract_col else ""
    out["Text"] = (out["Title"] + " " + out["Abstract"]).str.strip()
    out["StructuredText"] = [_serialize_structured(t, a) for t, a in zip(out["Title"], out["Abstract"])]
    return out


def _split_bullet_items(text: str) -> list[str]:
    """배제기준처럼 여러 항목이 줄바꿈/불릿으로 나열된 텍스트를 개별 항목으로 분리.
    분리되는 항목이 없으면(불릿 없는 한 문단) 전체를 항목 하나로 취급한다."""
    if not text or not text.strip():
        return []
    items = []
    for line in text.splitlines():
        stripped = line.strip()
        stripped = re.sub(r"^[\-\*\u2022\u25CF\u25AA]\s*", "", stripped)
        stripped = re.sub(r"^\d+[\.\)]\s*", "", stripped)
        if stripped:
            items.append(stripped)
    return items or [text.strip()]


def _cosine_sim_matrix(doc_texts: list[str], query_texts: list[str]) -> np.ndarray:
    """문헌 목록과 질의(PICO 항목/배제기준 항목) 사이의 코사인 유사도 행렬
    (n_docs, n_queries). 임베딩 모델이 있으면 의미 기반, 없으면 TF-IDF로 대체
    (둘 다 무료·로컬, API 비용 없음)."""
    if not query_texts:
        return np.zeros((len(doc_texts), 0))
    if embeddings_available():
        try:
            model = _get_embed_model()
            doc_vecs = model.encode(list(doc_texts), batch_size=32, show_progress_bar=False, normalize_embeddings=True)
            query_vecs = model.encode(list(query_texts), batch_size=32, show_progress_bar=False, normalize_embeddings=True)
            return doc_vecs @ query_vecs.T
        except Exception:
            # 배포 서버가 오프라인이거나 모델 캐시가 없어도 전체 앱은 TF-IDF로 계속 동작한다.
            pass
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    vectorizer.fit(list(doc_texts) + list(query_texts))
    doc_vecs = vectorizer.transform(doc_texts)
    query_vecs = vectorizer.transform(query_texts)
    return cosine_similarity(doc_vecs, query_vecs)


def _otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """1차원 점수 분포를 두 그룹으로 가장 잘 가르는 임계값 (Otsu, 1979).
    영상 이진화 기법을 연속값 분포에 그대로 적용한다 — '몇 %를 자를지'를
    사람이 정하는 대신, 그룹 간 분산이 최대가 되는 지점을 데이터 스스로
    찾게 한다. 라벨이 전혀 없는 zero-shot 모드에서 임계값을 고정 상수로
    하드코딩하지 않기 위한 용도."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or np.allclose(values.min(), values.max()):
        return float(np.median(values)) if len(values) else 0.0
    hist, edges = np.histogram(values, bins=bins)
    hist = hist.astype(float)
    total = hist.sum()
    if total == 0:
        return float(np.median(values))
    prob = hist / total
    bin_centers = (edges[:-1] + edges[1:]) / 2
    cumulative_prob = np.cumsum(prob)
    cumulative_mean = np.cumsum(prob * bin_centers)
    global_mean = cumulative_mean[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        between_var = (global_mean * cumulative_prob - cumulative_mean) ** 2 / (
            cumulative_prob * (1 - cumulative_prob)
        )
    between_var = np.nan_to_num(between_var, nan=-1.0, posinf=-1.0, neginf=-1.0)
    best_idx = int(np.argmax(between_var))
    return float(bin_centers[best_idx])



def _split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [p.strip() for p in parts if len(p.strip()) >= 12] or [text]


def build_sentence_pico_lookup(keys: np.ndarray, abstracts: np.ndarray, criteria_text: str) -> dict[str, np.ndarray] | None:
    """각 Abstract 문장에서 P/I/C/O와 가장 유사한 문장을 찾아 4개 feature로 저장한다.
    임베딩이 있으면 의미 유사도, 없으면 TF-IDF 유사도로 자동 폴백한다."""
    sections = _parse_pico_sections(criteria_text)
    pico_queries = [sections.get(k, "") for k in ["P", "I", "C", "O"]]
    if not any(q.strip() for q in pico_queries):
        return None
    lookup = {}
    for key, abstract in zip(keys, abstracts):
        sents = _split_sentences(abstract)
        if not sents:
            lookup[str(key)] = np.zeros(4, dtype=float)
            continue
        vals = []
        for q in pico_queries:
            if not q.strip():
                vals.append(0.0)
            else:
                sims = _cosine_sim_matrix(sents, [q])
                vals.append(float(np.max(sims[:, 0])) if sims.size else 0.0)
        lookup[str(key)] = np.asarray(vals, dtype=float)
    return lookup


def _obvious_exclusion_reason(title: str, abstract: str) -> str:
    t = f"{title} {abstract}".lower()
    checks = [
        (r"\bin\s*vitro\b|cell\s+culture|cultured\s+cells?|cell\s+lines?\b", "In vitro / cell study signal"),
        (r"systematic\s+review|meta[- ]analysis|narrative\s+review|scoping\s+review|literature\s+review", "Review article signal"),
        (r"study\s+protocol|\bprotocol\b", "Protocol signal"),
        (r"case\s+report|case\s+series", "Case report/series signal"),
    ]
    reasons = [label for pattern, label in checks if re.search(pattern, t, flags=re.I)]
    return "; ".join(reasons)



# ---------------------------------------------------------------------------
# PECO 규칙 게이트 (ML 앞단의 결정론적 사전 제외)
# ---------------------------------------------------------------------------
# 배경: 층화 표본(High/Mid/Low PICO)에서 Low/Mid 층 Include 1~2편이 큰 표본가중치를
# 갖기 때문에, 목표 재현율을 가중 기준으로 맞추면 임계값과 안전제외 컷오프가 동시에
# 붕괴해 '안전 제외 후보'가 거의 나오지 않는다. 이때 검토량을 줄이는 가장 확실한
# 수단은 모델을 더 돌리는 것이 아니라, 적격 문헌이라면 제목·초록에 반드시 등장할 수밖에
# 없는 용어군(노출어/결과어/연구설계어)을 AND 조건으로 걸어 corpus를 먼저 줄이는 것이다.
#
# 안전장치:
#   1) 각 규칙은 라벨 Include를 한 편이라도 떨어뜨리면 자동으로 비활성화된다
#      (사람 라벨 위에서 FN=0인 규칙만 살아남는다).
#   2) 살아남은 규칙이 없거나, 게이트 통과 Include가 너무 적으면 게이트 전체가 꺼진다.
#   3) 게이트에서 떨어진 문헌은 삭제되지 않고 '안전 제외 후보'로 분류되며,
#      Gate_Fail_Reason 열에 어떤 규칙에 걸렸는지 남는다 (감사·무작위 검증 가능).
# ---------------------------------------------------------------------------

GATE_MIN_INCLUDE_AFTER = 4     # 게이트 통과 라벨 Include가 이보다 적으면 게이트 사용 안 함
GATE_MIN_LABELED_AFTER = 20    # 게이트 통과 라벨이 이보다 적으면 게이트 사용 안 함

# V36: 기본 규칙은 비어 있다. 주제 특이적 정규식은 소스 코드가 아니라 '입력'이다.
# 프로젝트별로 PICO 화면의 「자동 제외 규칙」 칸에 검색식 개념 블록을 적으면
# parse_gate_rules_text()가 (이름, 정규식) 목록으로 바꿔 train_and_predict / rule_only_screen에 넘긴다.
# 입력 문자열은 run_fingerprint와 실행 manifest에 그대로 기록된다.
GATE_RULES_DEFAULT: list[tuple[str, str]] = []


def _missing_abstract_mask(abstracts) -> np.ndarray:
    """초록이 없거나 지나치게 짧은 레코드(제목만 있는 레코드)를 표시한다."""
    ser = pd.Series(abstracts).fillna("").astype(str).str.strip()
    return (ser.str.len() < ABSTRACT_MIN_CHARS).to_numpy()

# 니트로사민/HCA–심혈관–동물실험 SR에서 쓰던 규칙. 기본 적용되지 않으며, PICO 화면에서
# 「이전 니트로사민 SR 규칙 불러오기」를 눌러 해당 프로젝트의 입력으로만 사용할 수 있다.
LEGACY_NITROSAMINE_CVD_GATE_RULES: list[tuple[str, str]] = [
    (
        "노출어 없음",
        r"nitrosamin|nitrosodi|nitroso|\bndma\b|\bndea\b|\bnpyr\b|\bnpip\b|\bndba\b|\bnmor\b|\bndela\b|"
        r"\bdena\b|\bden\b|diethylnitro|dimethylnitro|heterocyclic\s+amine|heterocyclic\s+aromatic\s+amine|"
        r"\bhcas?\b|\bphip\b|\bmeiqx\b|\bdimeiqx\b|\bmeiq\b|trp-p|glu-p|a-?alpha-?c|aminoimidazo|"
        r"imidazo\[|pyrido\[|dipyrido|quinoxaline|quinoline",
    ),
    (
        "심혈관 결과어 없음",
        r"atheroscler|\baort|vascul|endotheli|\bcardi|\bheart\b|myocard|troponin|ck-?mb|vcam|icam|"
        r"\benos\b|plaque|vasodil|vasorelax|blood\s+pressure|lipoprotein|\bldl\b|\bhdl\b|\bvldl\b|"
        r"dyslipid|hyperlipid|foam\s+cell"
        r"|(?:serum|plasma|blood|circulating)[^.]{0,60}(?:lipid|cholesterol|triglycerid)"
        r"|(?:lipid|cholesterol|triglycerid)[^.]{0,60}(?:serum|plasma|blood\s+level)"
        r"|lipid\s+profile|lipid\s+panel|total\s+cholesterol|\btc\b|\btg\b|"
        r"free\s+fatty\s+acid|\bnefa\b",
    ),
    (
        "동물실험어 없음",
        r"\brats?\b|\bmice\b|\bmouse\b|\brabbits?\b|hamster|guinea\s+pig|\bpigs?\b|\bswine\b|"
        r"\bdogs?\b|canine|\bin\s*vivo\b|c57|balb|wistar|f344|fischer|sprague|ldlr|apoe|\bgavage\b|"
        r"\bintraperitoneal\b|\bchow\b|\bdiet\b|animal\s+model|\bmurine\b|\brodent",
    ),
]

GATE_RULE_REGEX_PREFIX = "re:"


def _term_to_regex(term: str) -> str:
    """검색어 한 개를 정규식으로. '*'는 절단(truncation), 공백은 공백 정규식, 앞쪽은 단어 경계.
    're:'로 시작하면 사용자가 쓴 정규식을 그대로 쓴다."""
    term = str(term).strip()
    if not term:
        return ""
    if term.lower().startswith(GATE_RULE_REGEX_PREFIX):
        return term[len(GATE_RULE_REGEX_PREFIX):].strip()
    trunc = term.endswith("*")
    core = term.rstrip("*").strip().lower()
    parts = [re.escape(p) for p in core.split()]
    body = r"\s+".join(parts)
    lead = r"\b" if core[:1].isalnum() else ""
    tail = "" if trunc else (r"\b" if core[-1:].isalnum() else "")
    return f"{lead}{body}{tail}"


def parse_gate_rules_text(text: str) -> list[tuple[str, str]]:
    """「자동 제외 규칙」 입력을 (이름, 정규식) 목록으로 바꾼다.

    한 줄 = 규칙 하나 = '이름: 용어1, 용어2, ...' (이름이 없으면 '규칙 N').
    같은 줄의 용어는 OR, 줄끼리는 AND(적격이면 모든 줄의 용어가 하나 이상 등장해야 함).
    '*' = 절단 검색어(nitrosamin* → nitrosamine, nitrosamines …). 're:패턴' = 정규식 직접 입력.
    '#'으로 시작하는 줄과 빈 줄은 무시한다. 잘못된 정규식이 있으면 ValueError.
    """
    rules: list[tuple[str, str]] = []
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line and not line.lower().startswith(GATE_RULE_REGEX_PREFIX):
            name, terms = line.split(":", 1)
            name = name.strip() or f"규칙 {len(rules) + 1}"
        else:
            name, terms = f"규칙 {len(rules) + 1}", line
        if terms.strip().lower().startswith(GATE_RULE_REGEX_PREFIX):
            pats = [_term_to_regex(terms.strip())]
        else:
            pats = [_term_to_regex(t) for t in re.split(r"[,;|]", terms)]
        pats = [p for p in pats if p]
        if not pats:
            continue
        pattern = "|".join(f"(?:{p})" for p in pats)
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"'{name}' 규칙의 정규식 오류: {exc}") from exc
        label = name if name.endswith("없음") else f"{name} 없음"
        rules.append((label, pattern))
    return rules


def gate_rules_to_text(rules: list[tuple[str, str]]) -> str:
    """(이름, 정규식) 목록을 입력 칸 형식('이름: re:패턴')으로 되돌린다."""
    out = []
    for name, pattern in rules:
        base = name[:-2].strip() if name.endswith("없음") else name
        out.append(f"{base}: {GATE_RULE_REGEX_PREFIX}{pattern}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 자동 개념 게이트 (V36) — 주제별 정규식을 사람이 쓰지 않아도 되게 한다.
#
# 적격 문헌이라면 제목·초록에 PICO의 각 요소(P, I/E, O)를 가리키는 말이 하나 이상 나온다.
# 그 '말 묶음(개념 블록)'을 코드나 사람이 아니라 (1) PICO 문장의 영어 핵심어와
# (2) 그 핵심어가 나오는 문헌에서 함께 많이 나오고(lift) 실제 Include 라벨에도 나오는 단어로 만든다.
#   * 블록은 라벨된 Include를 모두 포함할 때만 쓴다(coverage 100%).
#   * 블록을 만드는 '절차 전체'를 Include 한 편씩 빼고 다시 수행해(jackknife) 빠진 Include가
#     새 블록을 통과하는지 확인한다. 한 편이라도 떨어지면 그 블록은 쓰지 않는다.
#   * 남은 블록은 일반 규칙과 똑같이 build_gate(FN=0, 최소 Include 수, nested 평가)를 다시 거친다.
# PICO가 한국어뿐이면 영어 핵심어가 없으므로 블록이 만들어지지 않는다(= 규칙 없이 모델만 사용).
# ---------------------------------------------------------------------------
CONCEPT_ELEMENTS = [("P", "대상(P)"), ("I", "중재·노출(I/E)"), ("O", "결과(O)")]
CONCEPT_MAX_TERMS = 24
CONCEPT_MIN_LIFT = 2.0
_GENERIC_WORDS = {
    "study", "studies", "effect", "effects", "group", "groups", "result", "results", "method", "methods",
    "significant", "significantly", "increase", "increased", "decrease", "decreased", "level", "levels",
    "using", "used", "use", "based", "compared", "control", "controls", "treatment", "treated", "model",
    "models", "analysis", "data", "showed", "shown", "show", "found", "observed", "associated", "association",
    "high", "low", "higher", "lower", "induced", "including", "include", "included", "however", "also",
    "may", "well", "new", "different", "total", "change", "changes", "related", "role", "potential",
    "response", "responses", "test", "tested", "week", "weeks", "day", "days", "year", "years", "time",
    "outcome", "outcomes", "intervention", "interventions", "population", "comparison", "comparator",
    "exposure", "exposed", "patients", "subjects", "participants", "human", "humans", "factor", "factors",
}


def _concept_vectorizer():
    from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
    stop = sorted(set(ENGLISH_STOP_WORDS) | _GENERIC_WORDS)
    return CountVectorizer(binary=True, lowercase=True, ngram_range=(1, 2), min_df=2,
                           token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]{2,}\b", stop_words=stop)


_IRREGULAR = {"mice": "mouse", "mouse": "mice", "feet": "foot", "teeth": "tooth", "children": "child", "child": "children",
              "men": "man", "women": "woman"}


def _concept_term(t: str, abbrs: set | frozenset = frozenset(), vocab=None) -> str:
    """블록 용어 → 규칙 입력 형식. 긴 단어는 절단(*), 짧은 단어는 단수·복수(불규칙 포함)를 적는다.
    약어(NDEA 등)는 그대로 둔다. 복수형/단수형은 코퍼스에 실제로 있을 때만 덧붙인다."""
    if " " in t or t in abbrs:
        return t
    if len(t) >= 6:
        stem = t[:-1] if (t.endswith("s") and not t.endswith("ss") and (vocab is None or t[:-1] in vocab)) else t
        return f"{stem}*"
    forms = [t]
    alt = _IRREGULAR.get(t) or (t[:-1] if t.endswith("s") and not t.endswith("ss") else t + "s")
    if alt and (vocab is None or alt in vocab or t in _IRREGULAR):
        forms.append(alt)
    return ", ".join(dict.fromkeys(forms))


def _seed_terms(text: str, vocab: dict) -> list[str]:
    raw = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", str(text).lower())
    toks = []
    for tk in raw:
        toks.append(tk)
        if "-" in tk:
            toks += [p_ for p_ in tk.split("-") if len(p_) >= 3]
    seeds = []
    for i, tk in enumerate(toks):
        variants = [tk, tk[:-1] if tk.endswith("s") else tk + "s", _IRREGULAR.get(tk)]
        for cand in variants:
            if cand and cand in vocab and cand not in _GENERIC_WORDS and cand not in seeds:
                seeds.append(cand)
        if i + 1 < len(toks):
            bg = f"{tk} {toks[i + 1]}"
            if bg in vocab and bg not in seeds:
                seeds.append(bg)
    return seeds


_ABBR_RE = re.compile(r"([A-Za-z][\w\-]+(?:\s+[\w\-]+){0,5})\s*\(\s*([A-Za-z][A-Za-z0-9\-]{1,9})\s*\)")


def harvest_abbreviations(texts) -> list[tuple[str, str]]:
    """'long form (ABBR)' 쌍을 코퍼스에서 모은다(예: N-nitrosodiethylamine (NDEA)). 소문자로 반환."""
    pairs = {}
    for t in texts:
        for lf, ab in _ABBR_RE.findall(str(t)):
            ab_l = ab.lower()
            if len(ab_l) < 2 or ab_l.isdigit():
                continue
            pairs.setdefault(ab_l, set()).add(lf.lower())
    return [(lf, ab) for ab, lfs in pairs.items() for lf in lfs]


def _definitional_terms(seeds, terms, vocab, abbr_pairs):
    """씨앗어의 형태 가족(앞 6글자 공유: nitrosamine → nitrosodiethylamine)과 약어(NDEA)."""
    out = []
    uni = [t for t in terms if " " not in t]
    for sd in seeds:
        if " " in sd or len(sd) < 7:
            continue
        pre = sd[:6]
        fam = [t for t in uni if t.startswith(pre) or ("-" + pre) in t]
        out += fam[:12]
    stems = [sd[:6] if len(sd) >= 7 else sd for sd in seeds + out if " " not in sd]
    for lf, ab in abbr_pairs:
        tail = " ".join(lf.split()[-2:])             # 약어 바로 앞 두 단어(긴 형태의 끝)만 본다
        if ab in vocab and any(st_ in tail for st_ in stems):
            out.append(ab)
    return list(dict.fromkeys(out))


def _build_blocks(X, terms, vocab, inc_rows, sections, max_terms=CONCEPT_MAX_TERMS, abbr_pairs=()):
    """X: 코퍼스 binary 문서-용어 행렬(csr), inc_rows: 라벨 Include의 코퍼스 행 번호."""
    n_docs = X.shape[0]
    p_all = np.asarray(X.mean(axis=0)).ravel()
    X_inc = X[inc_rows] if len(inc_rows) else None
    inc_has = (np.asarray(X_inc.sum(axis=0)).ravel() if X_inc is not None else np.zeros(X.shape[1]))
    blocks = []
    for key, label in CONCEPT_ELEMENTS:
        text = sections.get(key, "")
        seeds = _seed_terms(text, vocab)
        if not seeds:
            continue
        defin = _definitional_terms(seeds, terms, vocab, abbr_pairs)
        sidx = list(dict.fromkeys([vocab[t] for t in seeds] + [vocab[t] for t in defin if t in vocab]))
        seed_doc = np.asarray(X[:, sidx].sum(axis=1)).ravel() > 0
        if seed_doc.sum() == 0:
            continue
        p_seed = np.asarray(X[seed_doc].mean(axis=0)).ravel()
        with np.errstate(divide="ignore", invalid="ignore"):
            lift = np.where(p_all > 0, p_seed / p_all, 0.0)
        cand = np.flatnonzero((inc_has > 0) & (lift >= CONCEPT_MIN_LIFT) & (p_seed >= 0.01))
        score = lift[cand] * (inc_has[cand] / max(len(inc_rows), 1))
        order = cand[np.argsort(-score, kind="stable")]
        chosen = list(sidx)
        covered = (np.asarray(X_inc[:, chosen].sum(axis=1)).ravel() > 0) if X_inc is not None else np.array([], bool)
        for j in order:                                   # 1) Include를 모두 덮을 때까지
            if X_inc is None or covered.all() or len(chosen) >= max_terms:
                break
            if j in chosen:
                continue
            col = np.asarray(X_inc[:, j].todense()).ravel() > 0
            if (col & ~covered).any():
                chosen.append(int(j))
                covered |= col
        if X_inc is not None and not covered.all():       # 1b) 아직 안 덮인 Include: 그 문헌에 특이적인 단어(보완어)
            with np.errstate(divide="ignore", invalid="ignore"):
                lift_inc = np.where(p_all > 0, (inc_has / max(len(inc_rows), 1)) / p_all, 0.0)
            unigram = np.array([" " not in t for t in terms])
            for r_i in np.flatnonzero(~covered):
                if len(chosen) >= max_terms:
                    break
                row_terms = X_inc[r_i].indices
                ok = row_terms[(p_all[row_terms] < 0.20) & unigram[row_terms]]
                if ok.size == 0:
                    continue
                j = int(ok[np.argmax(lift_inc[ok])])
                if j not in chosen:
                    chosen.append(j)
                covered |= np.asarray(X_inc[:, j].todense()).ravel() > 0
        unigram_mask = [" " not in terms[j] for j in order]
        order = order[np.asarray(unigram_mask, dtype=bool)] if len(order) else order
        for j in order:                                   # 2) 남는 자리는 동의어 폭을 넓히는 데(단일어만)
            if len(chosen) >= max_terms:
                break
            if j not in chosen:
                chosen.append(int(j))
        cov = float(covered.mean()) if covered.size else 0.0
        pass_rate = float((np.asarray(X[:, chosen].sum(axis=1)).ravel() > 0).mean()) if n_docs else 1.0
        blocks.append({"key": key, "label": label, "terms": [terms[j] for j in chosen], "idx": chosen,
                       "seeds": seeds, "include_coverage": cov, "corpus_pass_rate": pass_rate})
    return blocks


def derive_auto_gate(df: pd.DataFrame, criteria_text: str) -> dict:
    """PICO + 라벨된 Include로 개념 블록을 만들고 jackknife로 검증한다.
    반환: {"rules", "rules_text", "blocks"(표), "usable", "reasons", "n_include"}"""
    data, _ = prepare_screening_data(df)
    sections = _parse_pico_sections(criteria_text)
    texts = (data["Title"].fillna("") + " " + data["Abstract"].fillna("")).tolist()
    no_abs = _missing_abstract_mask(data["Abstract"].to_numpy())
    lab = data["Human_Label"]
    inc_rows = np.flatnonzero((lab == 1).to_numpy() & ~no_abs)
    out = {"rules": [], "rules_text": "", "blocks": pd.DataFrame(), "usable": False, "reasons": [],
           "n_include": int(len(inc_rows))}
    if not any(re.search(r"[a-zA-Z]{3,}", sections.get(k, "")) for k, _ in CONCEPT_ELEMENTS):
        out["reasons"].append("PICO(P·I/E·O)에 영어 핵심어가 없어 블록을 만들 수 없습니다. 영어로 적으면 자동 제안됩니다.")
        return out
    if len(inc_rows) < GATE_MIN_INCLUDE_AFTER:
        out["reasons"].append(f"초록 있는 Include가 {len(inc_rows)}편 — 최소 {GATE_MIN_INCLUDE_AFTER}편이 있어야 블록을 검증할 수 있습니다.")
        return out
    vec = _concept_vectorizer()
    try:
        X = vec.fit_transform(texts).tocsr()
    except ValueError:
        out["reasons"].append("코퍼스에서 용어를 추출하지 못했습니다.")
        return out
    terms = vec.get_feature_names_out().tolist()
    vocab = {t: i for i, t in enumerate(terms)}
    abbr_pairs = harvest_abbreviations(texts)
    blocks = _build_blocks(X, terms, vocab, inc_rows, sections, abbr_pairs=abbr_pairs)
    if not blocks:
        out["reasons"].append("PICO 핵심어가 코퍼스에 나오지 않아 블록을 만들 수 없습니다.")
        return out
    # 절차 전체 jackknife: Include i를 빼고 블록을 다시 만든 뒤, i가 그 블록을 통과하는가
    jk_fail = {b["key"]: 0 for b in blocks}
    for i in inc_rows:
        others = inc_rows[inc_rows != i]
        for b2 in _build_blocks(X, terms, vocab, others, sections, abbr_pairs=abbr_pairs):
            if b2["key"] in jk_fail and X[i, b2["idx"]].sum() == 0:
                jk_fail[b2["key"]] += 1
    rows, kept = [], []
    for b in blocks:
        ok = b["include_coverage"] >= 1.0 and jk_fail[b["key"]] == 0 and b["corpus_pass_rate"] < 0.98
        why = []
        if b["include_coverage"] < 1.0:
            why.append(f"Include 포함률 {b['include_coverage']*100:.0f}%")
        if jk_fail[b["key"]]:
            why.append(f"jackknife 탈락 {jk_fail[b['key']]}편")
        if b["corpus_pass_rate"] >= 0.98:
            why.append("거의 모든 문헌이 통과(거르는 효과 없음)")
        rows.append({"개념": b["label"], "PICO 핵심어": ", ".join(b["seeds"]), "용어 수": len(b["terms"]),
                     "용어": ", ".join(b["terms"]), "Include 포함률": f"{b['include_coverage']*100:.0f}%",
                     "코퍼스 통과율": f"{b['corpus_pass_rate']*100:.1f}%",
                     "Jackknife 탈락": jk_fail[b["key"]], "사용": "사용" if ok else "제외 — " + "; ".join(why)})
        if ok:
            kept.append(b)
    abbr_set = frozenset(ab for _lf, ab in abbr_pairs)
    lines = []
    for b in kept:
        forms = []
        for t in b["terms"]:
            for f_ in _concept_term(t, abbr_set).split(", "):   # 안전 쪽: 단수·복수 모두
                if f_ not in forms:
                    forms.append(f_)
        lines.append(f"{b['label']}: " + ", ".join(forms))
    out["rules_text"] = "\n".join(lines)
    out["rules"] = parse_gate_rules_text(out["rules_text"]) if lines else []
    out["blocks"] = pd.DataFrame(rows)
    out["usable"] = bool(kept)
    if not kept:
        out["reasons"].append("검증을 통과한 블록이 없습니다. 규칙 없이 모델 순위만 사용합니다.")
    return out


def _gate_rule_masks(texts: np.ndarray, rules: list[tuple[str, str]]) -> dict[str, np.ndarray]:
    """규칙별 '통과(해당 용어군이 존재)' 불린 마스크를 만든다."""
    lowered = pd.Series([str(t).lower() for t in texts])
    return {
        name: lowered.str.contains(pattern, regex=True, na=False).to_numpy()
        for name, pattern in rules
    }


def build_gate(
    all_texts: np.ndarray,
    labeled_pos: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None = None,
    rules: list[tuple[str, str]] | None = None,
    exempt: np.ndarray | None = None,
) -> dict:
    """규칙 게이트를 만들고 사람 라벨 위에서 검증한다.

    반환 dict:
        active        게이트를 실제로 적용해도 되는지
        pass_mask     전체 문헌에 대한 통과 여부 (active=False면 전부 True)
        reasons       전체 문헌에 대한 탈락 사유 문자열 ("" = 통과)
        kept_rules    FN=0으로 검증을 통과해 실제 적용된 규칙 이름
        dropped_rules {규칙명: 떨어뜨린 라벨 Include 편수}
        stats         라벨/가중 기준 제거량과 FN
    """
    rules = list(rules if rules is not None else GATE_RULES_DEFAULT)
    n_all = len(all_texts)
    if not rules:
        return {"active": False, "pass_mask": np.ones(n_all, dtype=bool),
                "reasons": np.array([""] * n_all, dtype=object), "kept_rules": [],
                "dropped_rules": {}, "stats": {}, "masks": {}}
    masks = _gate_rule_masks(all_texts, rules)
    # 초록이 없는 레코드는 용어 기반 규칙으로 판정할 수 없다. 규칙을 면제해 통과시키고,
    # 규칙 검증(FN 집계)에서도 제외한다. 이 레코드들은 별도 수기 확인 더미로 간다.
    if exempt is not None:
        exempt = np.asarray(exempt, dtype=bool)
        for name in masks:
            masks[name] = masks[name] | exempt
    else:
        exempt = np.zeros(n_all, dtype=bool)

    y = np.asarray(y, dtype=int)
    inc = y == 1
    kept: list[tuple[str, str]] = []
    dropped: dict[str, int] = {}
    for name, pattern in rules:
        lab_mask = masks[name][labeled_pos]
        fn = int((inc & ~lab_mask).sum())
        if fn == 0:
            kept.append((name, pattern))
        else:
            dropped[name] = fn

    info = {
        "active": False,
        "pass_mask": np.ones(n_all, dtype=bool),
        "reasons": np.array([""] * n_all, dtype=object),
        "kept_rules": [n for n, _ in kept],
        "dropped_rules": dropped,
        "stats": {},
        "masks": masks,
    }
    if not kept:
        return info

    pass_mask = np.ones(n_all, dtype=bool)
    reasons = [[] for _ in range(n_all)]
    for name, _ in kept:
        m = masks[name]
        pass_mask &= m
        for i in np.flatnonzero(~m):
            reasons[i].append(name)

    lab_pass = pass_mask[labeled_pos]
    if int((inc & lab_pass).sum()) < GATE_MIN_INCLUDE_AFTER or int(lab_pass.sum()) < GATE_MIN_LABELED_AFTER:
        info["dropped_rules"] = {**dropped, "(게이트 통과 라벨 부족으로 미적용)": 0}
        return info

    w = _weights_or_ones(weights, len(y))
    total_w = float(w.sum()) if float(w.sum()) > 0 else 1.0
    info.update({
        "active": True,
        "pass_mask": pass_mask,
        "reasons": np.array(["; ".join(r) for r in reasons], dtype=object),
        "stats": {
            "corpus_n": int(n_all),
            "corpus_removed_n": int((~pass_mask).sum()),
            "corpus_removed_pct": float((~pass_mask).sum() / n_all) if n_all else 0.0,
            "labeled_n": int(len(y)),
            "labeled_removed_n": int((~lab_pass).sum()),
            "labeled_include_removed_n": int((inc & ~lab_pass).sum()),
            "weighted_removed_pct": float(w[~lab_pass].sum() / total_w),
            "labeled_prevalence_before": float(w[inc].sum() / total_w),
            "labeled_prevalence_after": float(
                w[inc & lab_pass].sum() / w[lab_pass].sum()) if w[lab_pass].sum() > 0 else 0.0,
        },
    })
    return info


def _weighted_screening_metrics(y: np.ndarray, pred: np.ndarray, weights=None) -> dict:
    """게이트까지 포함한 파이프라인 전체의 (가중) Recall / WSS / 검토부담."""
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=bool)
    w = _weights_or_ones(weights, len(y))
    tp = w[(y == 1) & pred].sum(); fn = w[(y == 1) & ~pred].sum()
    fp = w[(y == 0) & pred].sum(); tn = w[(y == 0) & ~pred].sum()
    rec = float(tp / (tp + fn)) if (tp + fn) else 0.0
    return {
        "recall_weighted": rec,
        "wss_weighted": work_saved_over_sampling(tn, fn, tp, fp),
        "burden_weighted": float((tp + fp) / w.sum()) if w.sum() else 1.0,
    }


def _crossfold_policy_evaluation(
    probs: np.ndarray,
    y: np.ndarray,
    fold_ids: np.ndarray,
    recall_target: float,
    weights=None,
    gate_pass: np.ndarray | None = None,
    manual_review: np.ndarray | None = None,
    gate_rule_masks: dict | None = None,
) -> dict:
    """Threshold/cutoff를 평가 행 자체의 라벨로 정하지 않도록 fold별 정책 검증을 수행한다.

    각 fold의 문헌은 나머지 fold에서 정한 priority threshold와 safe-exclude cutoff로만 판정한다.
    모델 score 자체는 OOF score를 사용한다. 이는 full nested CV보다 계산량이 작으면서도
    '같은 200편 전체에서 cutoff를 맞춘 뒤 같은 200편으로 평가'하는 직접적인 재사용 편향을 줄인다.
    """
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=int)
    folds = np.asarray(fold_ids, dtype=int)
    w = _weights_or_ones(weights, len(y))
    gate = np.ones(len(y), dtype=bool) if gate_pass is None else np.asarray(gate_pass, dtype=bool)
    manual = np.zeros(len(y), dtype=bool) if manual_review is None else np.asarray(manual_review, dtype=bool)
    priority_pred = np.zeros(len(y), dtype=bool)
    safe_excluded = np.zeros(len(y), dtype=bool)
    details = []

    inc_all = y == 1
    for fold_no in sorted(int(x) for x in np.unique(folds) if int(x) > 0):
        va = folds == fold_no
        cal = ~va
        # V36: 규칙 게이트도 calibration fold 라벨만으로 고른다(nested). 전체 라벨로 고른
        # 규칙을 같은 라벨로 평가하면 게이트 FN이 정의상 0이 되는 누수를 막는다.
        fold_rules: list[str] = []
        if gate_rule_masks:
            fold_gate = np.ones(len(y), dtype=bool)
            for name, m in gate_rule_masks.items():
                m = np.asarray(m, dtype=bool)
                if int((inc_all & cal & ~m).sum()) == 0:
                    fold_rules.append(name)
                    fold_gate &= m
            if fold_rules and (int((inc_all & cal & fold_gate).sum()) < GATE_MIN_INCLUDE_AFTER
                               or int((cal & fold_gate).sum()) < GATE_MIN_LABELED_AFTER):
                fold_rules, fold_gate = [], np.ones(len(y), dtype=bool)
            gate = fold_gate
        cal_gate = cal & gate & ~manual
        # calibration fold에 두 클래스가 없으면 가장 보수적으로 전부 human review.
        if cal_gate.sum() < 4 or np.unique(y[cal_gate]).size < 2:
            thr = 0.0
            safe_cut = 0.0
        else:
            thr, _ = _optimize_threshold_wss(
                probs[cal_gate], y[cal_gate], recall_target,
                w[cal_gate] if weights is not None else None,
            )
            safe_cut = min(_safe_exclude_cutoff(probs[cal_gate], y[cal_gate], SAFE_RECALL_TARGET), thr)
        priority_pred[va] = ((probs[va] >= thr) & gate[va]) | manual[va]
        safe_excluded[va] = ((probs[va] < safe_cut) | ~gate[va]) & ~manual[va]

        pos = va & (y == 1)
        fold_safe_recall = float((~safe_excluded[pos]).mean()) if pos.any() else np.nan
        fold_priority_recall = float(priority_pred[pos].mean()) if pos.any() else np.nan
        details.append({
            "fold": fold_no,
            "priority_threshold": float(thr),
            "safe_cutoff": float(safe_cut),
            "validation_n": int(va.sum()),
            "include_n": int(pos.sum()),
            "priority_recall": fold_priority_recall,
            "safe_recall": fold_safe_recall,
            "safe_fn": int((pos & safe_excluded).sum()),
            "safe_excluded_n": int((va & safe_excluded).sum()),
            "gate_rules_in_fold": ", ".join(fold_rules),
        })

    priority_metrics = _weighted_screening_metrics(y, priority_pred, weights)
    safe_metrics = _weighted_screening_metrics(y, ~safe_excluded, weights)
    priority_metrics_unw = _weighted_screening_metrics(y, priority_pred, None)
    safe_metrics_unw = _weighted_screening_metrics(y, ~safe_excluded, None)
    safe_fn = int(((y == 1) & safe_excluded).sum())
    priority_fn = int(((y == 1) & ~priority_pred).sum())
    valid_safe_fold_recalls = [d["safe_recall"] for d in details if np.isfinite(d["safe_recall"])]
    valid_priority_fold_recalls = [d["priority_recall"] for d in details if np.isfinite(d["priority_recall"])]
    return {
        "priority_pred": priority_pred,
        "safe_excluded": safe_excluded,
        "priority_recall_weighted": float(priority_metrics["recall_weighted"]),
        "priority_recall_unweighted": float(priority_metrics_unw["recall_weighted"]),
        "priority_wss_weighted": float(priority_metrics["wss_weighted"]),
        "priority_burden_weighted": float(priority_metrics["burden_weighted"]),
        "priority_fn": priority_fn,
        "safe_recall_weighted": float(safe_metrics["recall_weighted"]),
        "safe_recall_unweighted": float(safe_metrics_unw["recall_weighted"]),
        "safe_wss_weighted": float(safe_metrics["wss_weighted"]),
        "safe_burden_weighted": float(safe_metrics["burden_weighted"]),
        "safe_fn": safe_fn,
        "safe_excluded_n": int(safe_excluded.sum()),
        "min_fold_safe_recall": float(min(valid_safe_fold_recalls)) if valid_safe_fold_recalls else 0.0,
        "min_fold_priority_recall": float(min(valid_priority_fold_recalls)) if valid_priority_fold_recalls else 0.0,
        "details": details,
    }


@dataclass
class ZeroShotResult:
    predictions: pd.DataFrame
    metrics: dict = field(default_factory=dict)


def zero_shot_screen(
    df: pd.DataFrame,
    criteria_text: str,
    exclusion_text: str = "",
) -> ZeroShotResult:
    """라벨 없이 PICO 기준 + 배제기준 텍스트만으로 문헌을 3단계로 자동 분류한다.

    사용 예:
        result = zero_shot_screen(
            df,
            criteria_text="P: 우주비행/미세중력 동물 또는 인체 모델\\nI: 영양 중재(비타민/미네랄/단백질 등)\\nO: 근골격계 지표(근위축, 골밀도 등)",
            exclusion_text="식물 대상 연구\\n미생물/세포주만 대상\\n운동/기구 중재만 있고 영양 중재 없음",
        )
        result.predictions  # AI_Recommendation 열 포함, 우선순위 정렬된 DataFrame
        result.metrics["disclaimer"]  # 통계적 보장이 없다는 안내 문구
    """
    if not criteria_text or not criteria_text.strip():
        raise ValueError("PICO 기준 텍스트가 필요합니다.")

    data = prepare_unlabeled_data(df)
    doc_texts = data["Text"].tolist()

    sections = _parse_pico_sections(criteria_text)
    section_names = list(sections.keys())
    section_texts = [sections[k] if sections[k].strip() else " " for k in section_names]
    # Abstract가 있으면 문장 단위에서 각 PICO 항목과 가장 가까운 문장을 사용한다.
    pico_sims = np.zeros((len(data), len(section_names)), dtype=float)
    for r, abstract in enumerate(data["Abstract"].tolist()):
        sents = _split_sentences(abstract) or [data.iloc[r]["Title"]]
        for i, q in enumerate(section_texts):
            sims = _cosine_sim_matrix(sents, [q])
            pico_sims[r, i] = float(np.max(sims[:, 0])) if sims.size else 0.0

    for i, name in enumerate(section_names):
        data[f"Similarity_{name}"] = pico_sims[:, i]

    # PICO 각 항목은 AND 조건(전부 맞아야 진짜 관련) -> 최솟값을 대표 점수로 사용.
    # 평균을 쓰면 한 항목만 매우 높아도 다른 항목이 안 맞는 문헌이 높은 점수를
    # 받는 왜곡이 생긴다.
    pico_score = pico_sims.min(axis=1) if pico_sims.shape[1] else np.zeros(len(doc_texts))
    data["PICO_Score"] = pico_score

    exclusion_items = _split_bullet_items(exclusion_text)
    if exclusion_items:
        excl_sims = _cosine_sim_matrix(doc_texts, exclusion_items)  # (n_docs, n_items)
        # 배제기준은 OR 조건(하나라도 걸리면 제외) -> 최댓값
        exclusion_score = excl_sims.max(axis=1)
    else:
        exclusion_score = np.zeros(len(doc_texts))
    data["Exclusion_Score"] = exclusion_score

    combined_score = pico_score - exclusion_score
    data["Combined_Score"] = combined_score

    # Otsu 임계값으로 자동 3단계 분류: 먼저 전체를 안전제외/나머지로 나누고,
    # 나머지를 다시 우선검토/경계문헌으로 재귀적으로 나눈다 (임의의 퍼센트나
    # 임계값을 사용자가 매번 고를 필요가 없다).
    threshold_1 = _otsu_threshold(combined_score)
    rest_mask = combined_score >= threshold_1
    rest_scores = combined_score[rest_mask]
    threshold_2 = _otsu_threshold(rest_scores) if len(rest_scores) >= 2 else threshold_1

    recommendation = np.full(len(doc_texts), "우선 검토", dtype=object)
    recommendation[~rest_mask] = "안전 제외 후보"
    borderline_mask = rest_mask & (combined_score < threshold_2)
    recommendation[borderline_mask] = "경계 문헌"
    data["AI_Recommendation"] = pd.Categorical(recommendation, categories=PRIORITY_ORDER, ordered=True)
    data["AI_Exclusion_Signal"] = [_obvious_exclusion_reason(t, a) for t, a in zip(data["Title"], data["Abstract"])]
    # 정렬 전에 입력 행 순서를 남긴다(확장 표본 설계에서 코퍼스와 정렬할 때 필요).
    data["_Corpus_Row"] = np.arange(len(data), dtype=int)

    data = data.sort_values(
        ["AI_Recommendation", "Combined_Score"], ascending=[True, False]
    ).reset_index(drop=True)

    metrics = {
        "mode": "zero_shot",
        "n_total": int(len(data)),
        "embedding_signal_used": embeddings_available(),
        "sections_detected": [n for n in section_names if n != "criteria"],
        "exclusion_items_n": len(exclusion_items),
        "safe_exclude_n": int((data["AI_Recommendation"] == "안전 제외 후보").sum()),
        "borderline_n": int((data["AI_Recommendation"] == "경계 문헌").sum()),
        "priority_n": int((data["AI_Recommendation"] == "우선 검토").sum()),
        "otsu_threshold_exclude": float(threshold_1),
        "otsu_threshold_priority": float(threshold_2),
        "disclaimer": ZERO_SHOT_DISCLAIMER,
    }
    return ZeroShotResult(predictions=data, metrics=metrics)


# ---------------------------------------------------------------------------
# 메인 랭킹 모델 (기존과 동일: Word+Char TF-IDF [+ PICO 유사도] -> Calibrated LinearSVC)
# ---------------------------------------------------------------------------

def _build_feature_union(
    criteria_text: str = "",
    embedding_lookup: dict | None = None,
    sentence_pico_lookup: dict | None = None,
) -> FeatureUnion:
    # Combined text
    combined_word = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=45000, sublinear_tf=True)
    combined_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=40000, sublinear_tf=True)

    # Title과 Abstract를 별도의 feature space로 학습한다.
    title_pipe = Pipeline([
        ("title", TextPartExtractor("title")),
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=20000, sublinear_tf=True)),
    ])
    abstract_pipe = Pipeline([
        ("abstract", TextPartExtractor("abstract")),
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=35000, sublinear_tf=True)),
    ])

    transformers = [
        ("combined_word", combined_word),
        ("combined_char", combined_char),
        ("title_word", title_pipe),
        ("abstract_word", abstract_pipe),
        ("rules", RuleSignalFeatures()),
        ("prototype", PrototypeSimilarity()),
    ]
    if criteria_text and criteria_text.strip():
        transformers.append(("criteria", CriteriaSimilarity(criteria_text=criteria_text)))
    if sentence_pico_lookup:
        transformers.append(("sentence_pico", NumericLookup(lookup=sentence_pico_lookup, dim=4)))
    if embedding_lookup:
        transformers.append(("embedding", EmbeddingLookup(lookup=embedding_lookup)))
    return FeatureUnion(transformers)


def _build_pipeline(
    criteria_text: str = "",
    embedding_lookup: dict | None = None,
    sentence_pico_lookup: dict | None = None,
    calib_cv: int = 3,
) -> Pipeline:
    features = _build_feature_union(criteria_text, embedding_lookup, sentence_pico_lookup)
    base = LinearSVC(class_weight="balanced", random_state=RANDOM_SEED)
    model = CalibratedClassifierCV(base, method="sigmoid",
                                   cv=StratifiedKFold(n_splits=calib_cv, shuffle=True, random_state=RANDOM_SEED))
    return Pipeline([("features", features), ("model", model)])

# ---------------------------------------------------------------------------
# 안전 제외 후보를 위한 보조 뷰 모델들.
# 메인 랭킹 모델(위 _build_pipeline)은 그대로 유지하고, 확률 하나만으로
# 안전 제외를 결정하지 않기 위해 서로 다른 근거(단어 TF-IDF만, 문자 TF-IDF만,
# 전체 피처 기반 로지스틱 회귀, PICO 유사도만)로 학습한 표준 sklearn 모델의
# 의견을 추가로 모은다. 모두 검증 fold 누수 없이 cross_val_predict로 계산한다.
# ---------------------------------------------------------------------------

def _build_word_only_pipeline() -> Pipeline:
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=50000, sublinear_tf=True)
    return Pipeline([("word", word), ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED))])


def _build_char_only_pipeline() -> Pipeline:
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=50000, sublinear_tf=True)
    return Pipeline([("char", char), ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED))])


def _build_pico_only_pipeline(criteria_text: str) -> Pipeline:
    return Pipeline([
        ("pico", CriteriaSimilarity(criteria_text=criteria_text)),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED)),
    ])


def _build_embedding_only_pipeline(embedding_lookup: dict) -> Pipeline:
    """의미 임베딩만으로 학습한 로지스틱 회귀. 어휘 중복(TF-IDF 계열)과 독립적인
    '의미가 비슷한가'라는 근거를 안전 제외 후보 판정에 추가한다."""
    return Pipeline([
        ("embedding", EmbeddingLookup(lookup=embedding_lookup)),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED)),
    ])


def _build_sentence_pico_only_pipeline(sentence_pico_lookup: dict) -> Pipeline:
    return Pipeline([
        ("sentence_pico", NumericLookup(lookup=sentence_pico_lookup, dim=4)),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED)),
    ])


def _build_logreg_full_pipeline(criteria_text: str = "", embedding_lookup: dict | None = None, sentence_pico_lookup: dict | None = None) -> Pipeline:
    features = _build_feature_union(criteria_text, embedding_lookup, sentence_pico_lookup)
    return Pipeline([("features", features), ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED))])


def _compute_safety_signals(
    texts: np.ndarray, y: np.ndarray, cv: StratifiedKFold, all_texts: np.ndarray, criteria_text: str = "",
    embedding_lookup: dict | None = None, sentence_pico_lookup: dict | None = None,
) -> dict:
    """Word TF-IDF, Character TF-IDF, 전체 피처 로지스틱 회귀, PICO 유사도, (가능하면)
    의미 임베딩 각각에 대해 (라벨 데이터의 교차검증 확률, 전체 데이터 확률)을 계산해 반환한다.
    TF-IDF 계열 뷰는 fold마다 처음부터 다시 학습되므로 검증 fold 누수가 없고,
    임베딩 뷰는 고정 가중치 인코더라 애초에 데이터 의존적 fit이 없어 누수 자체가 불가능하다.
    """
    signals: dict[str, dict] = {}

    def _run(name: str, pipeline: Pipeline):
        cv_probs, all_probs = _crossfit(pipeline, texts, y, cv, all_texts)
        signals[name] = {"cv": cv_probs, "all": all_probs}

    _run("word_tfidf", _build_word_only_pipeline())
    _run("char_tfidf", _build_char_only_pipeline())
    _run("logistic_regression", _build_logreg_full_pipeline(criteria_text, embedding_lookup, sentence_pico_lookup))
    if criteria_text and criteria_text.strip():
        _run("pico_similarity", _build_pico_only_pipeline(criteria_text))
    if sentence_pico_lookup:
        _run("sentence_pico", _build_sentence_pico_only_pipeline(sentence_pico_lookup))
    if embedding_lookup:
        _run("semantic_embedding", _build_embedding_only_pipeline(embedding_lookup))
    return signals


# ---------------------------------------------------------------------------
# 허용 False Negative(FN) 개수 기반 컷오프.
#
# "재현율 몇 %" 대신, 사람이 이해하기 쉬운 절대 개수("Include 문헌을 최대 N편까지만
# 놓치는 것을 허용")로 임계값을 정한다. 라벨된 Include 문헌들의 교차검증 확률을
# 오름차순 정렬해서, 가장 낮은 N개까지만 컷오프 아래로 떨어지도록 컷오프를 잡으면
# "이 컷오프를 쓰는 한 최대 N개까지만 놓친다"는 것이 라벨 데이터 위에서 수학적으로
# 보장된다. 안전 제외 후보 판정에 쓰이는 5개 신호 모두 같은 방식으로 컷오프를 잡고
# "모두 동의(교집합)"할 때만 안전 제외로 인정하므로, 안전 제외 후보 버킷의 FN 개수는
# 항상 이 허용치 이하로 유지된다 (교집합이므로 개별 신호의 FN 개수를 넘을 수 없다).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 목표 재현율(recall target) 기반 정책. "허용 FN 개수"를 매번 사람이 감으로
# 입력하는 대신, 문헌 스크리닝 자동화 분야에서 흔히 쓰는 고정된 재현율 목표
# (예: 95%)를 정책으로 두고, 현재 라벨 수로부터 allowed_fn을 결정론적으로
# 자동 계산한다. 같은 재현율 목표를 쓰는 한, 데이터셋이 달라져도 "무엇을
# 보장하는가"의 의미는 항상 동일하다 (허용 FN의 절대 개수만 라벨 수에 따라
# 달라질 뿐, 기준 자체가 흔들리는 게 아니다).
# ---------------------------------------------------------------------------

RECALL_TARGET_PRESETS = [0.99, 0.95, 0.90]
DEFAULT_RECALL_TARGET = 0.95

# ---------------------------------------------------------------------------
# 세 구간은 신호들을 스태킹한 단일 확률 하나로 나눈다.
#  - 우선 검토: 확률 ≥ 목표 Recall(기본 95%)에서 WSS 최대인 임계값
#  - 안전 제외: 확률 < 라벨 Include 점수의 99% 단측 예측구간 하한(외삽 여유 포함)
#  - 경계 문헌: 그 사이
# 이전 버전의 신호별 투표(정족수) 방식은 확률이 더 높은 문헌이 더 낮은 구간에 놓이는
# 역전이 생겼고, 정족수를 데이터로 고르면 같은 라벨로 선택·검증하는 과적합이 확인되어 폐기했다.
# ---------------------------------------------------------------------------
SAFE_RECALL_TARGET = 0.99


def allowed_fn_from_recall_target(n_include: int, recall_target: float) -> int:
    """목표 재현율을 만족하는 가장 관대한(=검토량을 가장 많이 줄이는) 허용 FN 개수.
    내림(floor)을 사용해 실제 달성 재현율이 목표를 절대 밑돌지 않도록 보수적으로 잡는다."""
    if n_include <= 0:
        return 0
    # round()로 부동소수점 오차(예: 0.9*20이 1.9999999996이 되는 경우)를 먼저 보정한 뒤
    # floor를 적용해, 0.90/0.95처럼 딱 떨어져야 할 값이 한 개씩 어긋나지 않도록 한다.
    raw = round((1.0 - float(recall_target)) * n_include, 6)
    allowed = int(np.floor(raw))
    return max(0, min(n_include, allowed))


def recall_lower_confidence_bound(n_include: int, allowed_fn: int, confidence: float = 0.95) -> float:
    """Clopper-Pearson(정확 이항) 하한. 현재 라벨 표본에서 이 allowed_fn을 썼을 때
    관측된 성공률(포착률)에 표본 크기 불확실성을 반영해, '모집단 재현율이 이 값
    이상일 것이라고 confidence 신뢰수준으로 말할 수 있는' 하한선을 계산한다.
    주의: 라벨 표본이 전체 문헌(라벨 없는 문헌 포함)을 대표한다는 가정이 전제이며,
    라벨 수가 적을수록 이 하한은 크게 내려간다 (표본이 작을수록 불확실성이 크다는
    사실을 감추지 않고 그대로 보여주기 위함)."""
    n = int(n_include)
    if n <= 0:
        return 0.0
    k = n - int(allowed_fn)  # 성공(=포착)한 라벨 Include 개수
    if k <= 0:
        return 0.0
    if k >= n:
        # 전부 포착(allowed_fn=0)한 경우의 하한: Jeffreys/Clopper-Pearson 상한쪽 특수 케이스
        return float(_beta_dist.ppf(1 - confidence, n, 1))
    return float(_beta_dist.ppf(1 - confidence, k, n - k + 1))


def work_saved_over_sampling(tn: int, fn: int, tp: int, fp: int) -> float:
    """WSS (Work Saved over Sampling), Cohen et al. 2006 정의.

    WSS = (TN+FN)/N - (1 - 실제 달성 재현율)

    무작위로 문헌을 훑는 것 대비, 이 스크리닝 파이프라인이 실제로 절감한 검토 비율을
    나타내는 SR 자동화 스크리닝 분야의 표준 지표다 (ASReview, CLEF eHealth TAR 등에서
    보고하는 방식). 목표 재현율(recall_target) 자체가 아니라 라벨 데이터에서 '실제 측정된'
    재현율을 쓴다 — allowed_fn은 floor로 보수적으로 잡히므로 실제 달성 재현율이 목표보다
    같거나 높을 수 있고, WSS 정의는 항상 실측 재현율을 기준으로 하기 때문이다.
    """
    n = tn + fp + fn + tp
    if n <= 0:
        return 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float((tn + fn) / n - (1.0 - recall))


def _weights_or_ones(weights, n: int) -> np.ndarray:
    if weights is None:
        return np.ones(n, dtype=float)
    w = np.asarray(weights, dtype=float)
    return np.where(np.isfinite(w) & (w > 0), w, 1.0)


def _optimize_threshold_wss(cv_probs: np.ndarray, y: np.ndarray, recall_target: float = 0.95, weights=None) -> tuple[float, dict]:
    """교차검증 Recall 제약을 만족하는 threshold 중 WSS가 가장 높은 값을 자동 선택한다.
    weights(Sampling_Weight)가 있으면 Recall·WSS를 층화 표본 가중치로 계산해 코퍼스 기준 추정치로 쓴다."""
    probs = np.asarray(cv_probs, dtype=float)
    y = np.asarray(y, dtype=int)
    w = _weights_or_ones(weights, len(y))
    candidates = np.unique(np.r_[0.0, probs, 1.0])
    best = None
    for thr in candidates:
        pred = probs >= thr
        tp = w[(y == 1) & pred].sum(); fn = w[(y == 1) & ~pred].sum()
        fp = w[(y == 0) & pred].sum(); tn = w[(y == 0) & ~pred].sum()
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        if rec + 1e-12 < float(recall_target):
            continue
        wss = work_saved_over_sampling(tn, fn, tp, fp)
        burden = (tp + fp) / w.sum() if w.sum() else 1.0
        score = (wss, -burden, thr)
        if best is None or score > best[0]:
            best = (score, float(thr), {"tn": float(tn), "fp": float(fp), "fn": float(fn), "tp": float(tp), "recall": float(rec), "wss": float(wss), "burden": float(burden)})
    if best is None:
        return 0.0, {"recall": 1.0, "wss": 0.0, "burden": 1.0}
    return best[1], best[2]


def _fn_budget_cutoff(cv_probs: np.ndarray, y: np.ndarray, allowed_fn: int, weights=None) -> float:
    """라벨 Include 중 확률이 낮은 쪽부터 허용 FN만큼만 컷오프 아래로 두는 가장 관대한 컷오프.
    weights가 있으면 '허용 FN 편수'를 Include 가중합의 같은 비율(allowed_fn / n_include)로 환산한다."""
    probs = np.asarray(cv_probs, dtype=float)
    y = np.asarray(y)
    inc_mask = y == 1
    if not inc_mask.any():
        return 0.5
    order = np.argsort(probs[inc_mask], kind="stable")
    inc_probs = probs[inc_mask][order]
    inc_w = _weights_or_ones(None if weights is None else np.asarray(weights)[inc_mask], inc_mask.sum())[order]
    allowed_fn = max(0, int(allowed_fn))
    if allowed_fn >= len(inc_probs):
        return 0.0
    allowed_mass = allowed_fn / len(inc_probs) * inc_w.sum()
    k = int(np.searchsorted(np.cumsum(inc_w), allowed_mass + 1e-9, side="right"))
    return float(inc_probs[min(k, len(inc_probs) - 1)])


def _safe_exclude_cutoff(cv_probs: np.ndarray, y: np.ndarray, level: float = 0.99) -> float:
    """안전 제외 컷오프. 라벨 Include의 CV 점수(logit)에 정규분포를 가정하고, 새 Include 한 편이
    이 값보다 낮을 확률이 (1-level)이 되는 단측 예측구간 하한을 쓴다.
        cut = sigmoid(mean - t_{level, n-1} · sd · sqrt(1 + 1/n))
    라벨 Include 중 최저값보다 더 아래로 외삽하므로, 라벨 Include가 적거나 점수가 흩어져
    있을수록 기준선이 자동으로 내려가 안전 제외가 줄어든다(라벨 Include 3편 미만이면 0편).
    컷오프는 관찰된 Include 점수의 단순 최솟값보다 보수적으로 설정해, 낮은 점수의
    잠재적 Include를 자동 제외하는 위험을 줄인다."""
    probs = np.asarray(cv_probs, dtype=float)
    inc = probs[np.asarray(y) == 1]
    if len(inc) < 3:
        return 0.0
    z = _logit(inc)
    n = len(z)
    lower = z.mean() - _t_dist.ppf(level, n - 1) * z.std(ddof=1) * np.sqrt(1 + 1 / n)
    return float(min(1 / (1 + np.exp(-lower)), inc.min()))


def _crossfit(pipeline, texts: np.ndarray, y: np.ndarray, cv: StratifiedKFold, all_texts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """fold 모델로 (라벨 문헌 OOF 확률, 전체 문헌에 대한 fold 모델 평균 확률)을 함께 계산한다.
    임계값을 정한 확률 분포(fold 모델)와 비라벨 문헌에 적용하는 확률 분포를 일치시키기 위함이다
    (전체 재학습 모델은 확률 분포가 fold 모델과 달라 CV 임계값의 의미가 흔들린다)."""
    oof = np.zeros(len(y), dtype=float)
    all_sum = np.zeros(len(all_texts), dtype=float)
    k = 0
    for tr, va in cv.split(texts, y):
        model = clone(pipeline)
        model.fit(texts[tr], y[tr])
        oof[va] = model.predict_proba(texts[va])[:, 1]
        all_sum += model.predict_proba(all_texts)[:, 1]
        k += 1
    return oof, all_sum / max(k, 1)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _stack_signals(signals: dict, y: np.ndarray, cv: StratifiedKFold, labeled_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """신호별 OOF 확률을 입력으로 하는 메타 로지스틱 회귀(stacking)로 단일 확률을 만든다.
    투표 대신 하나의 연속 점수로 세 구간을 나누므로, 확률이 높은 문헌이 더 낮은 구간에
    배치되는 역전이 생기지 않는다."""
    names = sorted(signals)
    x_cv = np.column_stack([_logit(signals[n]["cv"]) for n in names])
    x_all = np.column_stack([_logit(signals[n]["all"]) for n in names])
    meta = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=RANDOM_SEED)
    oof = cross_val_predict(meta, x_cv, y, cv=cv, method="predict_proba")[:, 1]
    meta.fit(x_cv, y)
    all_p = meta.predict_proba(x_all)[:, 1]
    all_p[labeled_pos] = oof
    return oof, all_p, {n: float(c) for n, c in zip(names, meta.coef_[0])}


def _priority_labels(probabilities: np.ndarray, threshold: float, unanimous_exclude: np.ndarray) -> np.ndarray:
    """스태킹 확률을 3단계로 구분한다.

    - 우선 검토: 확률이 임계값 이상 (흰색)
    - 안전 제외 후보: 확률이 안전 컷오프 미만 (진한 회색)
    - 경계 문헌: 그 사이, 사람이 반드시 확인 (중간 회색)
    """
    probabilities = np.asarray(probabilities, dtype=float)
    unanimous_exclude = np.asarray(unanimous_exclude, dtype=bool)
    return np.where(
        probabilities >= threshold, "우선 검토",
        np.where(unanimous_exclude, "안전 제외 후보", "경계 문헌"),
    )


def _sort_by_priority(pred_df: pd.DataFrame) -> pd.DataFrame:
    order = pd.Categorical(pred_df["AI_Recommendation"], PRIORITY_ORDER, ordered=True)
    return pred_df.assign(_priority_order=order).sort_values(
        ["_priority_order", "AI_Probability"], ascending=[True, False]
    ).drop(columns="_priority_order").reset_index(drop=True)


def _labeled_view(pred_df: pd.DataFrame):
    """저장된 예측 테이블에서 재타이어링에 필요한 라벨 뷰(라벨·CV확률·가중치·게이트)를 꺼낸다."""
    mask = pred_df["CV_Probability"].notna() & pred_df["Human_Label_Normalized"].isin([0, 1])
    y_labeled = pred_df.loc[mask, "Human_Label_Normalized"].astype(int).to_numpy()
    cv_probs = pd.to_numeric(pred_df.loc[mask, "CV_Probability"], errors="coerce").to_numpy()
    w_labeled = (pd.to_numeric(pred_df.loc[mask, "Sampling_Weight"], errors="coerce").to_numpy()
                 if "Sampling_Weight" in pred_df.columns else None)
    # 규칙 게이트 결과(Gate_Pass)는 재학습 없이도 그대로 유지한다.
    if "Gate_Pass" in pred_df.columns:
        gate_all = pred_df["Gate_Pass"].fillna(True).astype(bool).to_numpy()
    else:
        gate_all = np.ones(len(pred_df), dtype=bool)
    if "No_Abstract" in pred_df.columns:
        na_all = pred_df["No_Abstract"].fillna(False).astype(bool).to_numpy()
    else:
        na_all = _missing_abstract_mask(pred_df.get("Abstract", pd.Series([""] * len(pred_df))).to_numpy())
    return mask, y_labeled, cv_probs, w_labeled, gate_all, gate_all[mask.to_numpy()], na_all, na_all[mask.to_numpy()]


def _retier(result: ScreeningResult, threshold: float, strategy: str) -> ScreeningResult:
    """재학습 없이 임계값만 바꿔 3단계 분류와 성능 지표를 다시 계산한다.
    게이트에서 떨어진 문헌은 임계값과 무관하게 항상 '안전 제외 후보'로 남는다."""
    updated = deepcopy(result)
    pred_df = updated.predictions.copy()
    mask, y_labeled, cv_probs_main, w_labeled, gate_all, gate_lab, na_all, na_lab = _labeled_view(pred_df)
    updated.threshold = float(threshold)

    probs = pd.to_numeric(pred_df["AI_Probability"], errors="coerce").fillna(0).to_numpy()
    tune = gate_lab & ~na_lab
    if int(((y_labeled == 1) & tune).sum()) < 2:
        tune = gate_lab
    safe_cut = min(_safe_exclude_cutoff(cv_probs_main[tune], y_labeled[tune], SAFE_RECALL_TARGET), threshold)
    safe_all = ((probs < safe_cut) | ~gate_all) & ~na_all
    safe_cv = ((cv_probs_main < safe_cut) | ~gate_lab) & ~na_lab
    pred_df["Unanimous_Exclude"] = safe_all
    pred_df["AI_Recommendation"] = np.where(
        na_all, MANUAL_REVIEW_TIER,
        _priority_labels(np.where(gate_all, probs, -1.0), threshold, safe_all))
    updated.metrics["safe_cutoff"] = float(safe_cut)
    updated.metrics["safe_exclude_cv_n"] = int(safe_cv.sum())
    updated.metrics["safe_exclude_cv_false_negatives"] = int(((y_labeled == 1) & safe_cv).sum())

    cv_pred = (((cv_probs_main >= threshold) & gate_lab) | na_lab).astype(int)
    pred_df.loc[:, "CV_Prediction"] = np.nan
    pred_df.loc[mask, "CV_Prediction"] = cv_pred
    pred_df.loc[:, "False_Negative"] = False
    pred_df.loc[mask, "False_Negative"] = (y_labeled == 1) & (cv_pred == 0)
    tn, fp, fn, tp = confusion_matrix(y_labeled, cv_pred, labels=[0, 1]).ravel()
    rec = float(recall_score(y_labeled, cv_pred, zero_division=0))
    pre = float(precision_score(y_labeled, cv_pred, zero_division=0))
    updated.confusion = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    updated.metrics.update({
        "recall": rec, "precision": pre,
        "accuracy": float(accuracy_score(y_labeled, cv_pred)),
        "f1": float(2 * pre * rec / (pre + rec)) if pre + rec > 0 else 0.0,
        "measured_fn": int(fn),
        "wss": work_saved_over_sampling(tn, fn, tp, fp),
        "threshold": float(threshold),
        "threshold_strategy": strategy,
    })
    updated.metrics.update(_weighted_screening_metrics(y_labeled, cv_pred.astype(bool), w_labeled))
    updated.predictions = _sort_by_priority(pred_df)
    return updated


def apply_fn_budget(result: ScreeningResult, allowed_fn: int) -> ScreeningResult:
    """저장된 교차검증 확률을 사용해 재학습 없이 '허용 False Negative 개수'만 바꿔
    임계값과 화면 분류(우선 검토 / 경계 문헌 / 안전 제외 후보)를 다시 계산한다."""
    pred_df = result.predictions
    if not {"CV_Probability", "Human_Label_Normalized"}.issubset(pred_df.columns):
        return deepcopy(result)
    _, y_labeled, cv_probs_main, w_labeled, _, gate_lab, _na_all, na_lab = _labeled_view(pred_df)
    tune = gate_lab & ~na_lab
    if int(((y_labeled == 1) & tune).sum()) < 2:
        tune = gate_lab
    threshold = _fn_budget_cutoff(cv_probs_main[tune], y_labeled[tune], allowed_fn,
                                  None if w_labeled is None else w_labeled[tune])
    updated = _retier(result, threshold, "FN-budget cutoff (manual)")
    updated.metrics["allowed_fn"] = int(allowed_fn)
    return updated


def apply_recall_target(result: ScreeningResult, recall_target: float, confidence: float = 0.95) -> ScreeningResult:
    """저장된 교차검증 확률을 재사용해(재학습 없이), '목표 재현율' 정책만 바꿔
    화면 분류를 다시 계산한다. UI에는 99% / 95% / 90% 같은 고정된 정책 선택지만
    노출하고, allowed_fn(절대 개수)은 항상 이 함수 안에서 라벨 수로부터 자동 계산되므로
    사용자가 임의의 정수를 직접 입력할 일이 없다."""
    n_include = int(result.metrics.get("include_n", 0))
    allowed_fn = allowed_fn_from_recall_target(n_include, recall_target)
    pred_df = result.predictions
    if not {"CV_Probability", "Human_Label_Normalized"}.issubset(pred_df.columns):
        return deepcopy(result)
    _, y_labeled, cv_probs_main, w_labeled, _, gate_lab, _na_all, na_lab = _labeled_view(pred_df)
    # train_and_predict와 동일한 규칙으로 임계값을 잡는다: 게이트 통과 문헌 안에서
    # (가중) Recall ≥ 목표를 만족하는 후보 중 WSS가 최대인 값. 정책만 바꿔도 학습 직후와
    # 같은 임계값이 나오도록 두 경로를 일치시킨다.
    tune = gate_lab & ~na_lab
    if int(((y_labeled == 1) & tune).sum()) < 2:
        tune = gate_lab
    threshold, _info = _optimize_threshold_wss(
        cv_probs_main[tune], y_labeled[tune], recall_target,
        None if w_labeled is None else w_labeled[tune])
    updated = _retier(result, threshold, "Recall-constrained WSS optimization")
    updated.metrics["recall_target"] = float(recall_target)
    updated.metrics["allowed_fn"] = int(allowed_fn)
    # V36: 하한은 계획값(allowed_fn)이 아니라 실제 관측 FN으로 계산한다.
    updated.metrics["recall_lower_ci"] = recall_lower_confidence_bound(
        n_include, int(updated.metrics.get("measured_fn", allowed_fn)), confidence)
    return updated


def train_and_predict(
    df: pd.DataFrame,
    recall_target: float = DEFAULT_RECALL_TARGET,
    allowed_fn: int | None = None,
    criteria_text: str = "",
    gate_rules: list[tuple[str, str]] | None = None,
    validation_expected_n: int | None = None,
) -> ScreeningResult:
    """recall_target: 목표 재현율(예: 0.95 = 95%). 라벨 Include 중 이 비율 이상을
    반드시 '우선 검토' 또는 '경계 문헌'에 남기도록 allowed_fn을 자동으로 계산한다.
    allowed_fn을 직접 넘기면(고급 사용/하위 호환) recall_target 대신 그 값을 그대로 쓴다.
    """
    data, _ = prepare_screening_data(df)
    labeled = data[data["Human_Label"].isin([0, 1])].copy()
    # V36 재현성: 폴드 분할·학습 순서를 입력 행 순서가 아니라 레코드 내용 해시 순서로 고정한다.
    # (같은 코퍼스를 다른 순서로 export해도 결과가 같아야 한다.)
    labeled = labeled.assign(_order_key=_content_keys(labeled["Title"], labeled["Abstract"]))
    labeled = labeled.sort_values(["_order_key", "Human_Label"], kind="stable").drop(columns="_order_key")
    if len(labeled) < MIN_LABELS_FOR_SUPERVISED or labeled["Human_Label"].nunique() < 2:
        raise ValueError(f"학습을 위해 Include와 Exclude가 모두 포함된 최소 {MIN_LABELS_FOR_SUPERVISED}개 라벨이 필요합니다.")

    y = labeled["Human_Label"].astype(int).to_numpy()
    texts = labeled["StructuredText"].to_numpy()
    all_texts = data["StructuredText"].to_numpy()
    all_plain_texts = data["Text"].to_numpy()
    n_include = int(y.sum())

    if allowed_fn is None:
        allowed_fn = allowed_fn_from_recall_target(n_include, recall_target)

    min_class = int(labeled["Human_Label"].value_counts().min())
    folds = max(2, min(5, min_class))
    # 외부 fold의 학습 부분에 남는 소수 클래스 수에 맞춰 내부 보정(calibration) fold를 줄인다.
    # (Include가 적을 때 "less than 3 examples" 오류로 학습이 중단되던 문제)
    calib_cv = min(3, min_class - int(np.ceil(min_class / folds)))
    if calib_cv < 2:
        raise ValueError(f"Include 라벨이 {min_class}편뿐이라 교차검증을 할 수 없습니다. Include가 최소 4편 이상 필요합니다.")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_SEED)
    cv_fold_id = np.zeros(len(y), dtype=int)
    for fold_no, (_tr, va) in enumerate(cv.split(texts, y), start=1):
        cv_fold_id[va] = fold_no

    # 의미 임베딩은 전체 문헌 텍스트에 대해 한 번만 계산한다 (고정 가중치 인코더라
    # fold별 재계산이 필요 없고, 데이터로 다시 학습되지 않으므로 fold 밖에서 계산해도
    # 검증 누수가 생기지 않는다). sentence-transformers가 없는 환경에서는 None이 되어
    # 자동으로 이 신호 없이 나머지 모델들로만 동작한다 (기능 저하 없이 안전하게 폴백).
    # Embedding vector는 plain Title+Abstract에서 계산하되 StructuredText를 key로 사용한다.
    embedding_lookup = None
    embedding_error = ""
    if embeddings_available():
        try:
            model = _get_embed_model()
            vecs = model.encode(all_plain_texts.tolist(), batch_size=32, show_progress_bar=False, normalize_embeddings=True)
            embedding_lookup = {str(k): v for k, v in zip(all_texts, vecs)}
        except Exception as exc:
            # sentence-transformers 패키지는 있으나 모델 다운로드/로드가 실패하는 배포 환경을 안전하게 폴백.
            embedding_lookup = None
            embedding_error = f"{type(exc).__name__}: {exc}"[:300]
    sentence_pico_lookup = build_sentence_pico_lookup(all_texts, data["Abstract"].to_numpy(), criteria_text)

    raw = df.reset_index(drop=True)
    weights = (pd.to_numeric(raw.loc[labeled.index, "Sampling_Weight"], errors="coerce").to_numpy()
               if "Sampling_Weight" in raw.columns else None)
    labeled_pos = labeled.index.to_numpy()

    # 모든 신호를 cross-fit으로 계산한다: 라벨 문헌은 OOF 확률, 비라벨 문헌은 fold 모델 평균.
    main_pipeline = _build_pipeline(criteria_text, embedding_lookup, sentence_pico_lookup, calib_cv)
    svm_cv, svm_all = _crossfit(main_pipeline, texts, y, cv, all_texts)
    svm_all[labeled_pos] = svm_cv
    signals = _compute_safety_signals(texts, y, cv, all_texts, criteria_text, embedding_lookup, sentence_pico_lookup)
    signals["linear_svm"] = {"cv": svm_cv, "all": svm_all}
    for sig in signals.values():
        sig["all"][labeled_pos] = sig["cv"]

    probs, all_probs, stack_coef = _stack_signals(signals, y, cv, labeled_pos)

    precision, recall, pr_thresholds = precision_recall_curve(y, probs)
    fpr, tpr, _ = roc_curve(y, probs)

    # 규칙 게이트: 적격 문헌이라면 제목·초록에 반드시 나타나는 용어군을 AND로 걸어
    # ML 앞에서 corpus를 줄인다. 라벨 Include를 한 편이라도 떨어뜨리는 규칙은 자동 폐기된다.
    # 초록이 없는 레코드: 제목만으로는 PECO 판정이 불가능하므로 자동 제외 대상에서 빼고,
    # 임계값·안전컷오프·정책검증 어디에도 넣지 않는다. 별도 수기 확인 더미로 보낸다.
    no_abstract_all = _missing_abstract_mask(data["Abstract"].to_numpy())
    no_abstract_lab = no_abstract_all[labeled_pos]

    gate = build_gate(all_plain_texts, labeled_pos, y, weights, gate_rules, exempt=no_abstract_all)
    gate_pass_all = gate["pass_mask"]
    gate_pass_lab = gate_pass_all[labeled_pos]

    # 품질 게이트용 cross-fold policy evaluation:
    # 각 fold는 나머지 fold가 정한 threshold/cutoff만 적용받는다.
    gate_masks_lab = ({name: np.asarray(m)[labeled_pos] for name, m in gate.get("masks", {}).items()}
                      if gate.get("masks") else None)
    policy_eval = _crossfold_policy_evaluation(
        probs, y, cv_fold_id, recall_target, weights=weights, gate_pass=gate_pass_lab,
        manual_review=no_abstract_lab, gate_rule_masks=gate_masks_lab,
    )

    # 우선 검토 임계값: (가중) Recall ≥ 목표를 만족하면서 WSS가 최대인 값.
    # 게이트가 켜져 있으면 게이트 통과 문헌만으로 임계값을 잡는다 (게이트 밖은 이미 제외이므로
    # 그 문헌들까지 넣고 최적화하면 임계값이 불필요하게 낮아진다).
    if gate["active"]:
        tuning_lab = gate_pass_lab & ~no_abstract_lab
    else:
        tuning_lab = ~no_abstract_lab
    if int(((y == 1) & tuning_lab).sum()) < 2:   # 추정 불가 시 전체로 되돌림
        tuning_lab = np.ones(len(y), dtype=bool)
    thr_probs, thr_y = probs[tuning_lab], y[tuning_lab]
    thr_w = None if weights is None else np.asarray(weights)[tuning_lab]
    threshold, threshold_info = _optimize_threshold_wss(thr_probs, thr_y, recall_target, thr_w)
    pred = (((probs >= threshold) & gate_pass_lab) | no_abstract_lab).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    recall_v = float(recall_score(y, pred, zero_division=0))
    precision_v = float(precision_score(y, pred, zero_division=0))
    metrics = {
        "recall_target": float(recall_target),
        "allowed_fn": int(allowed_fn),
        "measured_fn": int(fn),
        "recall_lower_ci": recall_lower_confidence_bound(n_include, int(fn)),
        "recall": recall_v,
        "precision": precision_v,
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(2 * precision_v * recall_v / (precision_v + recall_v)) if (precision_v + recall_v) > 0 else 0.0,
        "roc_auc": float(roc_auc_score(y, probs)),
        "average_precision": float(average_precision_score(y, probs)),
        "labeled_n": int(len(labeled)),
        "validation_expected_n": int(validation_expected_n) if validation_expected_n is not None else int(len(labeled)),
        "validation_complete": bool(validation_expected_n is None or len(labeled) == int(validation_expected_n)),
        "include_n": n_include,
        "embedding_signal_used": embedding_lookup is not None,
        "embedding_fallback_reason": embedding_error,
        "wss": work_saved_over_sampling(tn, fn, tp, fp),
        "recall_weighted": float(threshold_info.get("recall", 1.0)),
        "wss_weighted": float(threshold_info.get("wss", 0.0)),
        "gate_active": bool(gate["active"]),
        "gate_kept_rules": list(gate["kept_rules"]),
        "gate_dropped_rules": dict(gate["dropped_rules"]),
        "gate_stats": dict(gate["stats"]),
        "sampling_weighted": weights is not None,
        "threshold": float(threshold),
        "threshold_strategy": "Recall-constrained WSS optimization (sampling-weighted)" if weights is not None else "Recall-constrained WSS optimization",
        "tier_method": "stacking",
        "algorithm_version": ALGORITHM_VERSION,
        "stack_coefficients": stack_coef,
    }

    metrics.update(_weighted_screening_metrics(y, pred.astype(bool), weights))
    metrics.update({
        "policy_priority_recall_weighted": policy_eval["priority_recall_weighted"],
        "policy_priority_recall_unweighted": policy_eval["priority_recall_unweighted"],
        "policy_priority_wss_weighted": policy_eval["priority_wss_weighted"],
        "policy_priority_burden_weighted": policy_eval["priority_burden_weighted"],
        "policy_priority_fn": policy_eval["priority_fn"],
        "policy_safe_recall_weighted": policy_eval["safe_recall_weighted"],
        "policy_safe_recall_unweighted": policy_eval["safe_recall_unweighted"],
        "policy_safe_wss_weighted": policy_eval["safe_wss_weighted"],
        "policy_safe_burden_weighted": policy_eval["safe_burden_weighted"],
        "policy_safe_fn": policy_eval["safe_fn"],
        "policy_safe_excluded_n": policy_eval["safe_excluded_n"],
        "policy_min_fold_safe_recall": policy_eval["min_fold_safe_recall"],
        "policy_min_fold_priority_recall": policy_eval["min_fold_priority_recall"],
        "policy_fold_details": policy_eval["details"],
        "policy_priority_recall_lower_ci": recall_lower_confidence_bound(n_include, int(policy_eval["priority_fn"]), VALIDATION_CONFIDENCE),
        "policy_safe_recall_lower_ci": recall_lower_confidence_bound(n_include, int(policy_eval["safe_fn"]), VALIDATION_CONFIDENCE),
    })

    threshold_100, info_100 = _optimize_threshold_wss(thr_probs, thr_y, 1.0, thr_w)
    metrics["threshold_100"] = float(threshold_100)
    metrics["wss_100"] = float(info_100.get("wss", 0.0))
    metrics["burden_100"] = float(info_100.get("burden", 1.0))
    metrics["fn_100"] = int(info_100.get("fn", 0))

    result_df = df.copy().reset_index(drop=True)
    result_df["AI_Probability"] = all_probs
    result_df["AI_Probability_%"] = (all_probs * 100).round(2)
    result_df["Human_Label_Normalized"] = data["Human_Label"].to_numpy()
    result_df["CV_Probability"] = np.nan
    result_df.loc[labeled.index, "CV_Probability"] = probs
    result_df["CV_Prediction"] = np.nan
    result_df.loc[labeled.index, "CV_Prediction"] = pred
    result_df["CV_Fold"] = np.nan
    result_df.loc[labeled.index, "CV_Fold"] = cv_fold_id
    result_df["Policy_Priority_Prediction"] = np.nan
    result_df.loc[labeled.index, "Policy_Priority_Prediction"] = policy_eval["priority_pred"].astype(int)
    result_df["Policy_Safe_Excluded"] = np.nan
    result_df.loc[labeled.index, "Policy_Safe_Excluded"] = policy_eval["safe_excluded"].astype(int)
    fold_recalls = []
    for fold_no in range(1, folds + 1):
        fm = cv_fold_id == fold_no
        if int((y[fm] == 1).sum()) == 0:
            continue
        fold_recalls.append(float(recall_score(y[fm], pred[fm], zero_division=0)))
    metrics["cv_fold_recalls"] = fold_recalls
    metrics["min_fold_recall"] = float(min(fold_recalls)) if fold_recalls else 0.0
    result_df["False_Negative"] = False
    result_df.loc[labeled.index, "False_Negative"] = (y == 1) & (pred == 0)

    for name, sig in signals.items():
        result_df[f"Prob_{name}"] = sig["all"]
        result_df[f"CV_Prob_{name}"] = np.nan
        result_df.loc[labeled.index, f"CV_Prob_{name}"] = sig["cv"]

    # 안전 제외: 라벨 Include 점수 분포의 99% 단측 예측구간 하한 아래(_safe_exclude_cutoff).
    # 게이트가 켜져 있으면 게이트 통과 라벨만으로 분포를 추정한다.
    safe_cut = min(_safe_exclude_cutoff(thr_probs, thr_y, SAFE_RECALL_TARGET), threshold)
    safe_all = ((all_probs < safe_cut) | ~gate_pass_all) & ~no_abstract_all
    safe_cv = ((probs < safe_cut) | ~gate_pass_lab) & ~no_abstract_lab
    metrics["safe_cutoff"] = float(safe_cut)
    metrics["safe_recall_target"] = float(SAFE_RECALL_TARGET)
    metrics["safe_exclude_cv_n"] = int(safe_cv.sum())
    safe_fn = int(((y == 1) & safe_cv).sum())
    metrics["safe_exclude_cv_false_negatives"] = safe_fn

    # 최종 운영정책의 안전성: 경계 문헌도 사람이 읽으므로 '사람 검토 유지'를 positive로 본다.
    # 이것이 실제 자동제외 때문에 relevant record를 놓치는지를 직접 측정하는 지표다.
    human_review_cv = ~safe_cv
    safe_unweighted = _weighted_screening_metrics(y, human_review_cv, None)
    safe_weighted = _weighted_screening_metrics(y, human_review_cv, weights)
    metrics["safe_recall"] = float(safe_unweighted["recall_weighted"])
    metrics["safe_wss"] = float(safe_unweighted["wss_weighted"])
    metrics["safe_burden"] = float(safe_unweighted["burden_weighted"])
    metrics["safe_recall_weighted"] = float(safe_weighted["recall_weighted"])
    metrics["safe_wss_weighted"] = float(safe_weighted["wss_weighted"])
    metrics["safe_burden_weighted"] = float(safe_weighted["burden_weighted"])
    metrics["safe_recall_lower_ci"] = recall_lower_confidence_bound(n_include, safe_fn, VALIDATION_CONFIDENCE)

    gate_fn = int(gate.get("stats", {}).get("labeled_include_removed_n", 0)) if gate.get("active") else 0
    complete = bool(metrics.get("validation_complete", False))
    enough_include = n_include >= MIN_INCLUDE_FOR_SUPERVISED
    ranking_ok = (
        float(metrics.get("policy_priority_recall_weighted", 0.0)) + 1e-12 >= float(recall_target)
        and float(metrics.get("policy_priority_recall_unweighted", 0.0)) + 1e-12 >= float(recall_target)
    )
    policy_safe_fn = int(metrics.get("policy_safe_fn", 0))
    safe_ok = (
        float(metrics.get("policy_safe_recall_weighted", 0.0)) + 1e-12 >= float(recall_target)
        and float(metrics.get("policy_safe_recall_unweighted", 0.0)) + 1e-12 >= float(recall_target)
    )
    gate_ok = gate_fn == 0

    quality_reasons = []
    if not complete:
        quality_reasons.append("human validation 표본이 완전히 라벨링되지 않음")
    if not enough_include:
        quality_reasons.append(f"Include가 {n_include}편으로 내부 검증에 부족함(권장 최소 {MIN_INCLUDE_FOR_SUPERVISED}편)")
    if not ranking_ok:
        quality_reasons.append(
            f"fold-held-out priority Recall(가중/비가중)이 목표 {recall_target*100:.0f}% 미만 "
            "— 우선 검토/경계 문헌 순위 품질 문제이며, 경계 문헌도 사람이 읽으므로 자동 제외 잠금 사유는 아님")
    if not safe_ok:
        quality_reasons.append(
            f"fold-held-out safe-exclude Recall(가중/비가중)이 목표 {recall_target*100:.0f}% 미만 "
            f"(관찰 FN {policy_safe_fn}편)"
        )
    if not gate_ok:
        quality_reasons.append(f"규칙 게이트가 human Include {gate_fn}편을 제외함")
    # AUC가 우연 수준이면 모델 문제가 아니라 Include 라벨이 서로 다른 성격의 문헌을 섞고 있을
    # 가능성이 크다. 같은 PECO로 라벨했다면 포함문헌끼리 어휘가 겹쳐야 한다.
    auc_now = float(metrics.get("roc_auc", 0.0) or 0.0)
    if n_include > 0 and auc_now < 0.60:
        quality_reasons.append(
            f"ROC-AUC {auc_now:.2f} — 우연 수준. 모델 성능 문제이기 이전에 Include 라벨이 "
            "서로 성격이 다른 문헌을 섞고 있는지(적격 기준 해석 일관성) 먼저 확인할 것"
        )
        metrics["label_consistency_warning"] = True

    # 자동 제외의 안전성은 safe-exclude Recall과 게이트 FN으로만 결정된다.
    # priority/경계 분할(ranking_ok)은 '우선 검토'와 '경계 문헌' 사이의 순서 문제이며,
    # 경계 문헌도 사람이 모두 읽으므로 자동 제외의 안전성과는 무관하다.
    # V30까지는 ranking_ok를 자동 제외 잠금 조건에 함께 넣어, 순위 품질이 낮다는 이유만으로
    # 안전성이 입증된 자동 제외까지 잠겼다.
    metrics["ranking_quality_ok"] = bool(ranking_ok)
    metrics["auto_exclusion_enabled"] = bool(complete and enough_include and safe_ok and gate_ok)
    metrics["quality_gate_status"] = "PASS" if (metrics["auto_exclusion_enabled"] and ranking_ok) else "REVIEW"
    metrics["quality_gate_reasons"] = quality_reasons
    # 예측 테이블은 우선순위대로 정렬되므로, 입력 코퍼스의 행 순서를 복원할 키를 남긴다.
    # (validation 확장에서 '코퍼스 행 ↔ 확률'을 정렬하는 데 필요하다.)
    result_df["_Corpus_Row"] = np.arange(len(result_df), dtype=int)
    rules_used = list(gate_rules if gate_rules is not None else GATE_RULES_DEFAULT)
    metrics["gate_rules_input"] = [list(r) for r in rules_used]
    metrics["run_fingerprint"] = run_fingerprint(
        data, extra=_fingerprint_extra("supervised", criteria_text, "", rules_used, recall_target))
    metrics["random_seed"] = int(RANDOM_SEED)
    metrics["software"] = software_versions()
    metrics["safe_metrics_in_sample_note"] = (
        "safe_recall / safe_exclude_cv_false_negatives는 컷오프를 정한 같은 라벨에서 계산되므로 "
        "정의상 FN이 거의 0이다. 성능 근거는 policy_* (fold-held-out, nested gate) 값을 쓴다.")
    result_df["Safety_Score"] = 1.0 - all_probs
    metrics["safety_signal_count"] = len(signals)
    result_df["Unanimous_Exclude"] = safe_all
    result_df["Gate_Pass"] = gate_pass_all
    result_df["No_Abstract"] = no_abstract_all
    result_df["Gate_Fail_Reason"] = gate["reasons"]
    result_df["AI_Recommendation"] = np.where(
        no_abstract_all,
        MANUAL_REVIEW_TIER,
        _priority_labels(np.where(gate_pass_all, all_probs, -1.0), threshold, safe_all),
    )
    metrics["no_abstract_n"] = int(no_abstract_all.sum())
    metrics["no_abstract_labeled_n"] = int(no_abstract_lab.sum())
    metrics["no_abstract_include_n"] = int(((y == 1) & no_abstract_lab).sum())
    result_df["Operational_Action"] = np.where(
        bool(metrics.get("auto_exclusion_enabled", False)) & safe_all,
        "AUTO_EXCLUDE",
        "HUMAN_REVIEW",
    )
    result_df["AI_Exclusion_Signal"] = [_obvious_exclusion_reason(t, a) for t, a in zip(data["Title"], data["Abstract"])]

    result_df = _sort_by_priority(result_df)

    return ScreeningResult(
        predictions=result_df,
        metrics=metrics,
        threshold=threshold,
        pr_curve={"precision": precision.tolist(), "recall": recall.tolist(), "thresholds": pr_thresholds.tolist()},
        roc_curve={"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        confusion={"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    )


def _export_group_labels(predictions: pd.DataFrame) -> pd.Series:
    """다운로드용 4구간 라벨을 만든다: 우선 검토 -> 경계 문헌 -> False Negative ->
    안전 제외 후보. False Negative(실제 Include인데 컷오프 아래로 예측된 문헌)는
    원래 속했던 버킷(보통 경계 문헌 또는 드물게 안전 제외 후보)에서 분리해
    독립된 구간으로 모아, 다운로드했을 때 놓치면 안 되는 문헌이 눈에 띄도록 한다.
    한 문헌은 정확히 한 구간에만 속한다 (중복 없음).
    """
    rec = predictions.get("AI_Recommendation", pd.Series("", index=predictions.index)).fillna("")
    is_fn = predictions.get("False_Negative", pd.Series(False, index=predictions.index)).fillna(False).astype(bool)
    group = np.select(
        [is_fn, rec.eq("우선 검토"), rec.eq("안전 제외 후보")],
        ["False Negative", "우선 검토", "안전 제외 후보"],
        default="경계 문헌",
    )
    return pd.Series(group, index=predictions.index, name="_export_group")


def build_grouped_excel_bytes(predictions: pd.DataFrame) -> bytes:
    """AI 스크리닝 결과를 우선 검토 -> 경계 문헌 -> False Negative -> 안전 제외 후보
    순서로 정렬하고, 구간별로 배경색을 입힌 엑셀 파일 바이트를 만든다.
    """
    df = predictions.copy()
    df["_export_group"] = _export_group_labels(df)
    df["_export_group"] = pd.Categorical(df["_export_group"], EXPORT_GROUP_ORDER, ordered=True)
    sort_cols = ["_export_group"] + (["AI_Probability"] if "AI_Probability" in df.columns else [])
    ascending = [True] + ([False] * (len(sort_cols) - 1))
    df = df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    group_labels = df["_export_group"].astype(str).tolist()
    export_df = df.drop(columns=["_export_group"])

    wb = Workbook()
    ws = wb.active
    ws.title = "AI_Screening_Ranked"

    for row in dataframe_to_rows(export_df, index=False, header=True):
        ws.append(row)

    header_fill = PatternFill(start_color="1F2937", end_color="1F2937", fill_type="solid")
    header_font = Font(color="FFFFFF", bold=True)
    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = header_font

    bands = (export_df["검토_우선도"].astype(str).tolist()
             if "검토_우선도" in export_df.columns else [""] * len(export_df))
    for i, grp in enumerate(group_labels, start=2):  # 1행은 헤더
        band = bands[i - 2]
        # 사람이 읽을 문헌은 그룹색 대신 '읽는 순서' 밴드색으로 칠한다(제외 결정과 무관).
        color = REVIEW_BAND_COLORS.get(band, EXPORT_GROUP_COLORS.get(grp, "FFFFFF"))
        fill = PatternFill(start_color=color, end_color=color, fill_type="solid")
        for cell in ws[i]:
            cell.fill = fill

    for col_idx, col_name in enumerate(export_df.columns, start=1):
        sample = export_df[col_name].astype(str).head(200).tolist()
        max_len = max([len(str(col_name))] + [len(str(v)) for v in sample]) if sample else len(str(col_name))
        ws.column_dimensions[get_column_letter(col_idx)].width = min(60, max(10, max_len + 2))

    ws.freeze_panes = "A2"

    legend_ws = wb.create_sheet("안내")
    legend_ws.append(["구간", "설명"])
    legend_ws["A1"].font = Font(bold=True)
    legend_ws["B1"].font = Font(bold=True)
    legend_rows = [
        ("우선 검토", "Include 확률이 임계값 이상인 문헌. 사람이 우선적으로 확인해야 합니다."),
        ("경계 문헌", "스태킹 확률이 우선 검토 임계값과 안전 제외 컷오프 사이인 문헌. 반드시 사람이 확인해야 합니다."),
        ("False Negative", "실제 라벨은 Include였지만 교차검증에서 임계값 아래로 예측된 문헌. 모델 개선 및 재확인이 필요합니다."),
        ("안전 제외 후보", "Word/Char TF-IDF, 로지스틱 회귀, 선형 SVM, PICO 유사도, (가능한 경우) 의미 임베딩 신호를 스태킹한 확률이, 라벨 Include 점수 분포로부터 새 Include가 이보다 낮을 확률이 1%가 되도록 정한 컷오프 미만인 문헌. 사람이 읽지 않아도 되는 문헌으로 제안되지만, 200편 교차검증에 근거한 추정입니다."),
    ]
    for i, (name, desc) in enumerate(legend_rows, start=2):
        legend_ws.append([name, desc])
        color = EXPORT_GROUP_COLORS.get(name, "FFFFFF")
        legend_ws.cell(row=i, column=1).fill = PatternFill(start_color=color, end_color=color, fill_type="solid")
    legend_ws.column_dimensions["A"].width = 16
    legend_ws.column_dimensions["B"].width = 90

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()




def _pct(x) -> str:
    """비율을 '93.8%' 문자열로 쓴다. 1.0이 엑셀에서 TRUE로 보이는 혼동을 막는다."""
    try:
        return f"{float(x) * 100:.1f}%"
    except Exception:
        return str(x)


def validation_methods_text(result: ScreeningResult) -> str:
    """현재 validation 결과를 Methods에 옮길 수 있는 보수적 문구를 만든다."""
    m = result.metrics
    expected = int(m.get("validation_expected_n", m.get("labeled_n", 0)))
    labeled = int(m.get("labeled_n", 0))
    inc = int(m.get("include_n", 0))
    target = float(m.get("recall_target", DEFAULT_RECALL_TARGET)) * 100
    safe_fn = int(m.get("policy_safe_fn", 0))
    safe_rec = float(m.get("policy_safe_recall_weighted", 0.0)) * 100
    status = str(m.get("quality_gate_status", "REVIEW"))
    return (
        f"A fixed human-validation sample of {expected} records was selected using PICO/PECO-enriched "
        f"stratified sampling (high-, mid-, and low-relevance strata). All {labeled} records were manually "
        f"labelled as potentially eligible or excluded ({inc} potentially eligible). Inverse-probability "
        f"sampling weights were retained for performance estimation. Model predictions for labelled records "
        f"were generated out-of-fold. For policy validation, each fold was classified using review-priority and "
        f"safe-exclusion cutoffs derived only from the remaining folds, targeting at least {target:.0f}% recall. "
        f"The fold-held-out auto-exclusion policy retained a sampling-weighted recall of {safe_rec:.1f}% in the "
        f"human-validation sample, with {safe_fn} potentially eligible records assigned to the auto-exclusion region. "
        f"The operational quality-gate status was {status}. "
        "These internal validation results were used as a quality-control safeguard and were not interpreted as a "
        "guarantee that no eligible records remained among unlabelled records. "
        + (
            f"Because the number of eligible records in the validation sample was finite ({inc}), recall is reported "
            f"with its one-sided 95% lower confidence bound ({float(m.get('policy_safe_recall_lower_ci', 0.0)) * 100:.1f}%) "
            "rather than as a point estimate of 100%. "
        )
        + (
            f"Applying Clopper-Pearson one-sided 95% upper bounds within risk strata of the automatically excluded set, "
            f"at most {float(m.get('audit_max_missed_current', 0.0)):.0f} eligible records could have been missed. "
            if m.get("audit_max_missed_current") else ""
        )
        + "Reference lists and forward citations of all included studies were additionally screened to detect records "
          "potentially missed at the title/abstract stage."
    )


def build_validation_report_excel_bytes(result: ScreeningResult) -> bytes:
    """Human-validation/AI screening 품질관리 결과를 감사 가능한 Excel report로 내보낸다."""
    m = result.metrics
    rule_only = str(m.get("mode", "")) == "rule_only"

    def _mv(key, fmt=None):
        """규칙 모드에는 존재하지 않는 모델 지표를 0.0%로 찍지 않는다."""
        if rule_only:
            return "N/A (규칙 기반 단독 모드 — 확률 모델 없음)"
        return (fmt or _pct)(m.get(key, 0.0))

    wb = Workbook()
    ws = wb.active
    ws.title = "Validation_Summary"
    header_fill = PatternFill(start_color="1F2937", end_color="1F2937", fill_type="solid")
    header_font = Font(color="FFFFFF", bold=True)
    ws.append(["Metric", "Value"])
    for c in ws[1]:
        c.fill = header_fill; c.font = header_font

    rows = [
        ("Algorithm version", m.get("algorithm_version", ALGORITHM_VERSION)),
        ("Generated UTC", datetime.now(timezone.utc).isoformat(timespec="seconds")),
        ("Quality gate", m.get("quality_gate_status", "REVIEW")),
        ("Auto-exclusion enabled", bool(m.get("auto_exclusion_enabled", False))),
        ("Validation expected n", int(m.get("validation_expected_n", m.get("labeled_n", 0)))),
        ("Validation labelled n", int(m.get("labeled_n", 0))),
        ("Human Include n", int(m.get("include_n", 0))),
        ("Human Exclude n", int(m.get("labeled_n", 0)) - int(m.get("include_n", 0))),
        ("Target Recall", float(m.get("recall_target", DEFAULT_RECALL_TARGET))),
        ("OOF Recall (unweighted)", _mv("recall")),
        ("OOF Recall (sampling-weighted, globally tuned)", _mv("recall_weighted")),
        ("Fold-held-out priority Recall (sampling-weighted)", _mv("policy_priority_recall_weighted")),
        ("Fold-held-out priority Recall (unweighted)", _mv("policy_priority_recall_unweighted")),
        ("OOF Recall one-sided 95% lower bound", _mv("recall_lower_ci")),
        ("OOF min-fold Recall", _mv("min_fold_recall")),
        ("Safe-exclude Recall (in-sample: cutoff set on these labels — not evidence)", _mv("safe_recall")),
        ("Safe-exclude Recall (in-sample, sampling-weighted — not evidence)", _mv("safe_recall_weighted")),
        ("Safe-exclude Recall lower bound (in-sample — not evidence)", _mv("safe_recall_lower_ci")),
        ("Fold-held-out safe Recall one-sided 95% lower bound", _mv("policy_safe_recall_lower_ci")),
        ("Fold-held-out safe Recall (sampling-weighted)", _mv("policy_safe_recall_weighted")),
        ("Fold-held-out safe Recall (unweighted)", _mv("policy_safe_recall_unweighted")),
        ("Fold-held-out safe FN", int(m.get("policy_safe_fn", 0))),
        ("Fold-held-out safe-excluded n", int(m.get("policy_safe_excluded_n", 0))),
        ("Fold-held-out final WSS (sampling-weighted)", float(m.get("policy_safe_wss_weighted", 0.0))),
        ("Fold-held-out final human-review burden (sampling-weighted)", float(m.get("policy_safe_burden_weighted", 1.0))),
        ("F1 (OOF, unweighted)", _mv("f1", lambda v: f"{float(v):.3f}")),
        ("Precision (OOF, unweighted)", _mv("precision")),
        ("ROC-AUC", _mv("roc_auc", lambda v: f"{float(v):.3f}")),
        ("Average precision", float(m.get("average_precision", 0.0))),
        ("Run fingerprint (input hash)", str(m.get("run_fingerprint", ""))),
        ("Random seed", int(m.get("random_seed", RANDOM_SEED))),
        ("Quality-gate reasons", "; ".join(map(str, m.get("quality_gate_reasons", []))) or "None"),
    ]
    for r in rows:
        # 값은 전부 문자열로 적는다. 숫자 0이 엑셀에서 FALSE로 보이거나 1.0이 TRUE로 보이는
        # 혼동을 없애기 위해서다(심사자가 보는 표라 모호하면 안 된다).
        v = r[1]
        if isinstance(v, bool):          # bool은 int의 하위형이라 먼저 걸러야 한다
            txt = "Yes" if v else "No"
        elif isinstance(v, str):
            txt = v
        elif isinstance(v, (int, np.integer)):
            txt = f"{int(v):,}"
        else:
            txt = str(v)
        ws.append([str(r[0]), txt])
    ws.column_dimensions["A"].width = 46
    ws.column_dimensions["B"].width = 80
    ws.freeze_panes = "A2"

    # Human validation rows only
    pred = result.predictions.copy()
    val = pred[pred.get("Human_Label_Normalized", pd.Series(np.nan, index=pred.index)).isin([0, 1])].copy()
    preferred = [
        "Training_No", "Validation_Record_ID", "Training_Stratum", "Sampling_Weight",
        "Title", "제목", "Abstract", "초록", "Human_Label_Normalized", "CV_Fold",
        "CV_Probability", "CV_Prediction", "Policy_Priority_Prediction", "Policy_Safe_Excluded",
        "False_Negative", "AI_Recommendation",
        "AI_Probability", "Gate_Pass", "Gate_Fail_Reason",
    ]
    cols = [c for c in preferred if c in val.columns]
    val = val[cols]
    vws = wb.create_sheet("Human_Validation")
    for row in dataframe_to_rows(val, index=False, header=True):
        vws.append(row)
    if vws.max_row >= 1:
        for c in vws[1]: c.fill = header_fill; c.font = header_font
        vws.freeze_panes = "A2"
        vws.auto_filter.ref = vws.dimensions
    for j, name in enumerate(val.columns, start=1):
        vws.column_dimensions[get_column_letter(j)].width = 42 if name in {"Title", "제목"} else (70 if name in {"Abstract", "초록"} else 20)

    # 실제 자동 제외 영역에서 발견된 human Include를 별도 표시
    _policy_safe = pd.to_numeric(val.get("Policy_Safe_Excluded", pd.Series(0, index=val.index)), errors="coerce").fillna(0).astype(int)
    safe_err = val[(val.get("Human_Label_Normalized", 0) == 1) & (_policy_safe == 1)].copy()
    ews = wb.create_sheet("Safe_Exclude_Errors")
    for row in dataframe_to_rows(safe_err, index=False, header=True):
        ews.append(row)
    if ews.max_row >= 1:
        for c in ews[1]: c.fill = header_fill; c.font = header_font
        ews.freeze_panes = "A2"

    # --- PRISMA 흐름과 감사 설계: 동료심사에서 반드시 요구되는 두 가지 -------------
    pred = result.predictions
    counts = pred["AI_Recommendation"].value_counts()
    auto_on = bool(result.metrics.get("auto_exclusion_enabled", False))
    auto_n = int(counts.get("안전 제외 후보", 0)) if auto_on else 0
    flow = [
        ["Stage", "n"],
        ["Records screened by AI-assisted workflow", int(len(pred))],
        ["Human-validation sample labelled by reviewers", int(result.metrics.get("labeled_n", 0))],
        ["  of which judged potentially eligible", int(result.metrics.get("include_n", 0))],
        ["Records assigned to automatic exclusion", auto_n],
        ["Records retained for human title/abstract screening", int(len(pred)) - auto_n],
        ["  priority tier", int(counts.get("우선 검토", 0))],
        ["  borderline tier", int(counts.get("경계 문헌", 0))],
        ["  human-review tier (rule-only mode)", int(counts.get("사람 검토", 0))],
        ["  no-abstract tier (title-only, manual)", int(counts.get(MANUAL_REVIEW_TIER, 0))],
    ]
    if str(result.metrics.get("mode", "")) == "rule_only" and "검토_우선도" in pred.columns:
        # 규칙 모드에는 우선/경계 구분이 없다. 대신 읽는 순서 밴드별 편수를 적는다.
        for b in REVIEW_BAND_ORDER:
            flow.append([f"    reading band {b}", int((pred["검토_우선도"] == b).sum())])
    fws = wb.create_sheet("PRISMA_Flow")
    for row in flow: fws.append(row)
    for c in fws[1]: c.fill = header_fill; c.font = header_font
    fws.column_dimensions["A"].width = 56; fws.column_dimensions["B"].width = 14

    try:
        strata = audit_risk_strata(pred)
        opts = audit_size_options(strata)
        aws = wb.create_sheet("Audit_Design")
        aws.append(["자동 제외 집합의 위험층별 누락 상한 (Clopper-Pearson 95% 단측)"])
        aws["A1"].font = Font(bold=True)
        aws.append([])
        for row in dataframe_to_rows(strata, index=False, header=True): aws.append(row)
        start = aws.max_row - len(strata)
        for c in aws[start]: c.fill = header_fill; c.font = header_font
        aws.append([])
        aws.append(["목표 상한별 추가 감사 분량"]); aws.cell(row=aws.max_row, column=1).font = Font(bold=True)
        for row in dataframe_to_rows(opts, index=False, header=True): aws.append(row)
        for j, wdt in enumerate([46, 14, 12, 14, 18, 18], start=1):
            aws.column_dimensions[get_column_letter(j)].width = wdt
        result.metrics["audit_max_missed_current"] = float(strata["최대_누락_추정"].sum())
    except Exception:
        pass

    try:
        lc, lstats = label_consistency_check(result.predictions)
        if len(lc):
            lws = wb.create_sheet("Label_Consistency")
            lws.append(["Include 라벨끼리의 어휘 유사도 점검 — '재확인_권고'가 True면 적격 기준을 다시 적용해 볼 것"])
            lws["A1"].font = Font(bold=True)
            lws.append([])
            for row in dataframe_to_rows(lc, index=False, header=True): lws.append(row)
            hdr_row = lws.max_row - len(lc)
            for c in lws[hdr_row]: c.fill = header_fill; c.font = header_font
            lws.column_dimensions["A"].width = 80
            for j in range(2, 6): lws.column_dimensions[get_column_letter(j)].width = 24
            result.metrics["label_flagged_n"] = int(lstats.get("flagged", 0))
    except Exception:
        pass

    mws = wb.create_sheet("Methods_Text")
    mws["A1"] = "Suggested Methods wording"
    mws["A1"].font = Font(bold=True)
    mws["A2"] = validation_methods_text(result)
    mws["A2"].alignment = __import__('openpyxl').styles.Alignment(wrap_text=True, vertical="top")
    mws.column_dimensions["A"].width = 120
    mws.row_dimensions[2].height = 160

    nws = wb.create_sheet("Interpretation")
    notes = [
        ["Item", "Interpretation"],
        ["PASS", "Human validation이 완전하며 최소 Include 수를 충족하고, fold-held-out priority 및 safe-exclude Recall이 weighted·unweighted 모두 목표 이상이며, 활성 custom gate가 human Include를 제거하지 않는 경우."],
        ["REVIEW", "위 조건 중 하나라도 충족하지 못한 경우. 모델 순위는 참고할 수 있으나 자동 제외는 잠금 상태로 취급해야 함."],
        ["Confidence bound", "표본의 유한한 Include 수 때문에 생기는 불확실성을 보여주는 보조 지표. PASS를 통계적 무누락 보장으로 해석하지 않음."],
        ["Scope", "현재 결과는 해당 review의 200편 내부 human-validation 및 OOF 예측에 대한 품질관리 결과이며, 미라벨 전체 코퍼스에 대한 절대적 보장이 아님."],
        ["운영 규칙(고정)", "초록 있음 + safe-exclusion 규칙 충족 → 자동 제외 / 규칙 미충족 → 사람 검토 / 초록 없음 → 무조건 사람 검토. 나중에 초록을 확보하더라도 이미 사람 검토로 분류된 문헌은 규칙에 다시 넣지 않는다(Human_Review_Locked)."],
        ["초록 없음 티어의 의미", "이 문헌들은 Include가 아니라 'Retain for human screening'이다. 적격 여부 판정은 사람이 제목·초록 또는 원문을 확인한 뒤 내린다."],
        ["Figure 구성", "A 선별 효율(사람이 읽는 비율 대비 적격 보존율), B 집합별 보존율과 95% CI, C 작업량 감소와 잘못된 자동 제외 수. ROC/PR/AUC/F1은 결정론적 규칙에 정의되지 않으므로 넣지 않는다."],
        ["규칙 기반 단독 모드", "확률 모델을 쓰지 않으므로 Recall/ROC-AUC/F1은 정의되지 않는다(N/A). 보고할 수치는 (1) 적용된 규칙과 제외 편수, (2) human Include 탈락 수, (3) 위험층별 누락 상한, (4) known-item 복구 결과다."],
        ["노출어 셀 vs 결과어 셀", "노출어가 없어 제외된 셀은 논리로 방어된다(PECO상 노출이 필수이므로 제목·초록에 노출어가 없으면 적격 판정 자체가 불가능). 표본 감사가 필요한 것은 노출어는 있는데 결과어가 없어 제외된 셀뿐이다. 전체 셀을 한꺼번에 목표 상한에 맞추려 하면 불필요하게 많이 읽게 된다."],
        ["보고 원칙", "Recall은 점추정 100%가 아니라 단측 95% 하한과 함께 보고한다. Include 수가 적을수록 하한은 낮아지며, 이것이 실제 불확실성이다."],
        ["Audit_Design", "자동 제외 집합을 '제외 사유 × 노출어 포함 여부'로 나눈 뒤 셀별 누락 상한을 계산한 표. 노출어가 있는데 제외된 셀이 가장 위험하다(주제는 맞는데 결과어 규칙이 못 잡은 경우). 목표 상한을 정하고 그만큼 추가로 읽는 것이 동료심사 대응의 핵심이다."],
        ["논리적 근거 vs 통계적 근거", "노출어가 아예 없어 제외된 문헌은 PECO상 노출이 필수이므로 제목·초록 단계에서 적격 판정 자체가 불가능하다(논리적 근거). 반면 노출어는 있으나 결과어가 없어 제외된 문헌은 결과어 사전의 누락 가능성이 있으므로 표본 감사로 뒷받침해야 한다(통계적 근거)."],
        ["Known-item recovery", "이미 적격임을 아는 문헌(연구계획서 인용문헌, 선행 리뷰 포함문헌 등)이 자동 제외되지 않았는지 확인하는 검사. 통계적 상한보다 심사자 설득력이 크다."],
    ]
    for row in notes: nws.append(row)
    for c in nws[1]: c.fill = header_fill; c.font = header_font
    nws.column_dimensions["A"].width = 24; nws.column_dimensions["B"].width = 120
    for row in nws.iter_rows(min_row=2): row[1].alignment = __import__('openpyxl').styles.Alignment(wrap_text=True, vertical="top")

    buf = io.BytesIO(); wb.save(buf); return buf.getvalue()


# ---------------------------------------------------------------------------
# Validation 표본 확장 (post-stratified extension)
# ---------------------------------------------------------------------------
# Include가 적어 safe-exclude Recall의 신뢰구간이 넓을 때, 정확도를 올리는 올바른 방법은
# 학습에 쓴 라벨을 validation에 합치는 것이 아니다(독립성이 깨져 추정이 낙관적으로 편향된다).
# 대신 같은 코퍼스에서 validation 표본을 '추가로' 뽑아야 한다.
#
# 단순 무작위 추가는 유병률이 1% 수준이라 비효율적이므로, PICO 층 × AI 확률구간으로
# 사후층화(post-stratification)한 뒤 Include가 실제로 있는 셀에 표본을 집중한다.
# AI 확률은 독립된 training 표본으로 적합된 모델의 출력이고 validation 라벨과 무관한
# 공변량이므로, 이 사후층화는 설계상 유효하다. 각 셀 안에서는 기존 라벨도 추가 표본도
# 그 셀의 단순무작위표본이므로, 가중치는 셀별 N/n으로 다시 계산하면 불편추정이 유지된다.
# ---------------------------------------------------------------------------

PROB_BAND_EDGES = (0.30, 0.60)


def _prob_band(p: np.ndarray, edges: tuple[float, float] = PROB_BAND_EDGES) -> np.ndarray:
    lo, hi = edges
    p = np.asarray(p, dtype=float)
    return np.where(p >= hi, f"P>={hi:.2f}", np.where(p >= lo, f"P {lo:.2f}-{hi:.2f}", f"P<{lo:.2f}"))


def pico_rank_strata(
    df: pd.DataFrame,
    criteria_text: str,
    exclusion_text: str = "",
    sample_size: int = TRAINING_SAMPLE_SIZE,
) -> pd.Series:
    """build_training_sample과 동일한 점수·경계로 코퍼스 전체의 PICO 층을 복원한다."""
    base = df.copy().reset_index(drop=True)
    title_col = _find_col(base, ["title", "제목"])
    abstract_col = _find_col(base, ["abstract", "초록"])
    titles = base[title_col].fillna("").astype(str)
    abstracts = base[abstract_col].fillna("").astype(str) if abstract_col else pd.Series([""] * len(base))
    docs = (titles + " " + abstracts).str.strip().tolist()

    sections = _parse_pico_sections(criteria_text)
    queries = [q.strip() for q in sections.values() if q and q.strip()] or [criteria_text.strip()]
    exclusion_items = _split_bullet_items(exclusion_text)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=60000,
                          sublinear_tf=True, stop_words="english")
    mat = vec.fit_transform(docs + queries + exclusion_items)
    doc_mat = mat[:len(docs)]
    pico_sim = cosine_similarity(doc_mat, mat[len(docs):len(docs) + len(queries)])
    scores = 0.65 * pico_sim.mean(axis=1) + 0.35 * pico_sim.min(axis=1)
    if exclusion_items:
        scores = scores - 0.75 * cosine_similarity(doc_mat, mat[len(docs) + len(queries):]).max(axis=1)

    n_total = len(base)
    n = min(int(sample_size), n_total)
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    high_n = min(int(round(n * STRATUM_ALLOCATION[0])), n_total)
    cut_mid = max(high_n, min(n_total, int(np.ceil(n_total * STRATUM_BOUNDS[1]))))
    out = np.array(["Low PICO relevance"] * n_total, dtype=object)
    out[order[:high_n]] = "High PICO relevance"
    out[order[high_n:cut_mid]] = "Mid PICO relevance"
    return pd.Series(out, index=base.index)


def plan_validation_extension(
    corpus: pd.DataFrame,
    labeled_validation: pd.DataFrame,
    probabilities: np.ndarray,
    criteria_text: str,
    exclusion_text: str = "",
    n_add: int = 200,
) -> pd.DataFrame:
    """셀(PICO 층 × 확률구간)별 추가 표본 배분 계획과 기대 Include 수를 돌려준다."""
    if len(probabilities) != len(corpus):
        raise ValueError(
            f"확률 배열 길이({len(probabilities)})가 코퍼스 행 수({len(corpus)})와 다릅니다. "
            "예측 테이블을 _Corpus_Row 순으로 정렬해 전달하세요."
        )
    strata = pico_rank_strata(corpus, criteria_text, exclusion_text).to_numpy()
    bands = _prob_band(probabilities)
    cell_all = pd.Series([f"{s} | {b}" for s, b in zip(strata, bands)])

    lab = labeled_validation.copy()
    key = _find_col(lab, ["_source_index", "_Source_Index"])
    lab_idx = lab[key].astype(int).to_numpy() if key is not None else np.array([], dtype=int)
    y = lab.get("Human_Label_Normalized")
    if y is None:
        y = lab["Human_Label"].map(_normalize_label_value)
    y = pd.to_numeric(y, errors="coerce").fillna(-1).to_numpy()

    rows = []
    for cell in sorted(cell_all.unique()):
        in_cell = (cell_all == cell).to_numpy()
        N = int(in_cell.sum())
        lab_in = np.isin(lab_idx, np.flatnonzero(in_cell))
        n_lab = int(lab_in.sum())
        inc = int((y[lab_in] == 1).sum())
        # Include 수가 적으므로 Jeffreys 사전(0.5)으로 평활한 유병률을 쓴다.
        p_hat = (inc + 0.5) / (n_lab + 1.0) if n_lab else 0.5 / 1.0
        rows.append({"Cell": cell, "N_corpus": N, "n_labeled": n_lab,
                     "include_labeled": inc, "prevalence_est": p_hat,
                     "expected_includes_per_label": p_hat})
    plan = pd.DataFrame(rows)
    # 추가 라벨 1편당 기대 Include 수가 큰 셀부터 채운다(유병률 내림차순, 남은 미라벨 수로 상한).
    # 유병률에 비례 배분하면 Include가 한 편도 없는 큰 셀에 표본이 몰린다 — 목적은
    # '코퍼스를 대표하는 추가 표본'이 아니라 '가장 적은 노동으로 Include를 확보하는 것'이고,
    # 대표성은 셀별 N/n 가중치가 담당한다.
    plan["remaining"] = (plan["N_corpus"] - plan["n_labeled"]).clip(lower=0)
    plan["allocate"] = 0
    budget = int(n_add)
    for i in plan.sort_values(["prevalence_est", "remaining"], ascending=[False, False]).index:
        if budget <= 0:
            break
        take = int(min(budget, plan.at[i, "remaining"]))
        plan.at[i, "allocate"] = take
        budget -= take
    plan["expected_new_includes"] = (plan["allocate"] * plan["prevalence_est"]).round(2)
    return plan.sort_values("expected_new_includes", ascending=False).reset_index(drop=True)


def build_validation_extension(
    corpus: pd.DataFrame,
    labeled_validation: pd.DataFrame,
    probabilities: np.ndarray,
    criteria_text: str,
    exclusion_text: str = "",
    n_add: int = 200,
    random_state: int = 20260101,
) -> pd.DataFrame:
    """추가로 라벨링할 validation 표본을 뽑는다 (셀 내 단순무작위, 기존 라벨과 중복 없음)."""
    if len(probabilities) != len(corpus):
        raise ValueError("확률 배열 길이가 코퍼스 행 수와 다릅니다.")
    plan = plan_validation_extension(corpus, labeled_validation, probabilities,
                                     criteria_text, exclusion_text, n_add)
    strata = pico_rank_strata(corpus, criteria_text, exclusion_text).to_numpy()
    bands = _prob_band(probabilities)
    cell_all = np.array([f"{s} | {b}" for s, b in zip(strata, bands)], dtype=object)

    key = _find_col(labeled_validation, ["_source_index", "_Source_Index"])
    already = set(labeled_validation[key].astype(int).tolist()) if key is not None else set()

    base = corpus.copy().reset_index(drop=True)
    base["_Source_Index"] = np.arange(len(base), dtype=int)
    title_col = _find_col(base, ["title", "제목"])
    rng = np.random.default_rng(random_state)

    picks = []
    for _, r in plan.iterrows():
        take = int(r["allocate"])
        if take <= 0:
            continue
        pool = [i for i in np.flatnonzero(cell_all == r["Cell"]) if int(i) not in already]
        if not pool:
            continue
        take = min(take, len(pool))
        chosen = rng.choice(np.array(pool), size=take, replace=False)
        for i in chosen:
            row = base.iloc[int(i)].copy()
            row["Validation_Cell"] = r["Cell"]
            row["Validation_Phase"] = "extension"
            row["Validation_Record_ID"] = _stable_record_id(int(i), str(base[title_col].iloc[int(i)]))
            picks.append(row)

    out = pd.DataFrame(picks).reset_index(drop=True)
    if out.empty:
        raise ValueError("추가로 뽑을 수 있는 미라벨 문헌이 없습니다.")
    out.insert(0, "Extension_No", np.arange(1, len(out) + 1))
    out["Human_Label"] = ""
    return out


def merge_validation_extension(
    corpus: pd.DataFrame,
    labeled_validation: pd.DataFrame,
    labeled_extension: pd.DataFrame,
    probabilities: np.ndarray,
    criteria_text: str,
    exclusion_text: str = "",
) -> tuple[pd.DataFrame, dict]:
    """기존 validation + 확장 표본을 합치고, 셀별 N/n으로 Sampling_Weight를 다시 계산한다."""
    if len(probabilities) != len(corpus):
        raise ValueError("확률 배열 길이가 코퍼스 행 수와 다릅니다.")
    strata = pico_rank_strata(corpus, criteria_text, exclusion_text).to_numpy()
    bands = _prob_band(probabilities)
    cell_all = np.array([f"{s} | {b}" for s, b in zip(strata, bands)], dtype=object)
    N_cell = pd.Series(cell_all).value_counts().to_dict()

    frames = []
    for part, phase in ((labeled_validation, "initial"), (labeled_extension, "extension")):
        if part is None or len(part) == 0:
            continue
        d = part.copy()
        d["Validation_Phase"] = phase
        frames.append(d)
    merged = pd.concat(frames, ignore_index=True)

    key = _find_col(merged, ["_source_index", "_Source_Index"])
    if key is None:
        raise ValueError("_Source_Index 열이 없어 표본을 합칠 수 없습니다.")
    merged = merged.drop_duplicates(subset=[key], keep="first").reset_index(drop=True)
    idx = merged[key].astype(int).to_numpy()
    merged["Validation_Cell"] = cell_all[idx]

    n_cell = merged["Validation_Cell"].value_counts().to_dict()
    merged["Sampling_Stratum_N"] = merged["Validation_Cell"].map(N_cell).astype(int)
    merged["Sampling_Stratum_n"] = merged["Validation_Cell"].map(n_cell).astype(int)
    merged["Sampling_Weight"] = (merged["Sampling_Stratum_N"] / merged["Sampling_Stratum_n"]).round(8)
    merged["Sampling_Probability"] = (1.0 / merged["Sampling_Weight"]).round(8)

    lab = merged.get("Human_Label_Normalized")
    if lab is None:
        lab = merged["Human_Label"].map(_normalize_label_value)
    lab = pd.to_numeric(lab, errors="coerce")
    stats = {
        "n_total": int(len(merged)),
        "n_initial": int((merged["Validation_Phase"] == "initial").sum()),
        "n_extension": int((merged["Validation_Phase"] == "extension").sum()),
        "include_n": int((lab == 1).sum()),
        "cells_used": int(merged["Validation_Cell"].nunique()),
        "max_weight": float(merged["Sampling_Weight"].max()),
    }
    return merged, stats


# ---------------------------------------------------------------------------
# 위험층화 감사 (risk-stratified audit of the auto-excluded set)
# ---------------------------------------------------------------------------
# 동료심사에서 가장 먼저 지적되는 지점은 "AI가 제외한 수천 편을 아무도 읽지 않았다"이다.
# 내부 validation(라벨 200편)에서 safe-exclude FN=0이라는 사실만으로는 부족하다.
# 라벨 수가 작으면 '놓쳤을 수 있는 최대 편수'의 상한이 전체 적격 문헌 수보다 커질 수 있고,
# 그러면 "누락이 없다"는 주장은 통계적으로 공허하다.
#
# 그래서 자동 제외 집합을 위험 셀(제외 사유 × 노출어 포함 여부)로 나누고,
#   (1) 셀별로 현재 상한(Clopper-Pearson 95% 단측 상한 × 셀 크기)을 계산하고
#   (2) 목표 상한을 만족하려면 셀별로 몇 편을 더 읽어야 하는지 역산하고
#   (3) 그만큼 무작위로 뽑아 사람이 읽은 뒤 상한을 다시 계산한다.
# 보고에는 recall 100%가 아니라 이 상한을 쓴다.
# ---------------------------------------------------------------------------

AUDIT_CONFIDENCE = 0.95
# V36: 감사 층화에 쓰는 '노출어' 패턴은 프로젝트 규칙의 첫 줄에서 가져온다(없으면 층화 안 함).
AUDIT_EXPOSURE_PATTERN = ""


def _cp_upper(n: int, k: int = 0, confidence: float = AUDIT_CONFIDENCE) -> float:
    """Clopper-Pearson 단측 상한. k=0이면 1-(1-c)^(1/n)과 같다."""
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(_beta_dist.ppf(confidence, k + 1, n - k))


def _audit_cells(pool: pd.DataFrame, exposure_pattern: str | None) -> list[str]:
    reason = pool.get("Gate_Fail_Reason", pd.Series([""] * len(pool), index=pool.index)).fillna("")
    reason = reason.where(reason.astype(str).str.len() > 0, "확률 기준 제외")
    if not exposure_pattern:
        return [str(r) for r in reason]
    text = (pool.get("Title", pool.get("제목", pd.Series([""] * len(pool), index=pool.index))).fillna("").astype(str) + " "
            + pool.get("Abstract", pool.get("초록", pd.Series([""] * len(pool), index=pool.index))).fillna("").astype(str)).str.lower()
    has_exp = text.str.contains(exposure_pattern, regex=True, na=False)
    return [f"{r} | 노출어 {'있음' if e else '없음'}" for r, e in zip(reason, has_exp)]


def audit_risk_strata(
    predictions: pd.DataFrame,
    exposure_pattern: str | None = AUDIT_EXPOSURE_PATTERN,
    audit_labels: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """자동 제외 집합을 위험 셀로 나누고 셀별 누락 상한을 계산한다.

    셀 = 제외 사유(Gate_Fail_Reason, 없으면 '확률 기준 제외') × 노출어 포함 여부.
    노출어가 있는데 제외된 문헌이 가장 위험하다 — 주제는 맞는데 규칙이 결과어를 못 찾은 경우다.
    """
    pool = predictions[predictions["AI_Recommendation"] == "안전 제외 후보"].copy()
    if pool.empty:
        raise ValueError("자동 제외 후보가 없어 감사 설계를 만들 수 없습니다.")

    pool["_audit_cell"] = _audit_cells(pool, exposure_pattern)

    # 셀 크기는 '실제 문헌 수'다. 표본가중치는 라벨 표본을 코퍼스로 환산할 때만 쓰는 값이라,
    # 이미 코퍼스 전체 행을 들고 있는 예측 테이블에 다시 곱하면 셀 합이 코퍼스를 초과한다.
    lab = pd.to_numeric(pool.get("Human_Label_Normalized", pd.Series([np.nan] * len(pool))), errors="coerce")

    extra_n: dict[str, int] = {}
    extra_k: dict[str, int] = {}
    if audit_labels is not None and len(audit_labels):
        a = audit_labels.copy()
        acol = next((c for c in a.columns if str(c).strip().lower() in
                     {"audit_label", "human_label", "판정", "라벨"}), None)
        if acol is None:
            raise ValueError("감사 파일에서 판정 열(Audit_Label)을 찾지 못했습니다.")
        av = a[acol].map(_normalize_label_value)
        cell_col = next((c for c in a.columns if str(c) in {"감사_셀", "_audit_cell", "Audit_Cell"}), None)
        if cell_col is None:
            raise ValueError("감사 파일에서 감사_셀 열을 찾지 못했습니다.")
        for cell, grp in a.assign(_v=av).groupby(a[cell_col].astype(str)):
            judged = grp["_v"].isin([0, 1])
            extra_n[str(cell)] = int(judged.sum())
            extra_k[str(cell)] = int((grp["_v"] == 1).sum())

    rows = []
    for cell, grp in pool.groupby("_audit_cell"):
        idx = grp.index
        N = float(len(idx))
        # V36: 개발용(validation) 라벨은 규칙·컷오프를 고르는 데 쓰였고 무작위 표본도 아니므로
        # 감사 근거에서 뺀다. 참고용으로만 별도 열에 남긴다.
        dev_n = int(lab.loc[idx].isin([0, 1]).sum())
        n_lab = extra_n.get(str(cell), 0)
        k = extra_k.get(str(cell), 0)
        ub = _cp_upper(n_lab, k)
        rows.append({
            "감사_셀": cell,
            "N_corpus": int(round(N)),
            "읽은_편수": n_lab,
            "발견_Include": k,
            "개발라벨_편수(참고·제외)": dev_n,
            "누락률_95%상한": ub,
            "최대_누락_추정": ub * N,
        })
    out = pd.DataFrame(rows).sort_values("최대_누락_추정", ascending=False).reset_index(drop=True)
    return out


def recommend_audit_sizes(strata: pd.DataFrame, target_max_missed: float = 10.0) -> pd.DataFrame:
    """전체 누락 상한을 target_max_missed 이하로 만들기 위해 셀별로 몇 편을 더 읽어야 하는지.

    셀별 목표는 코퍼스 크기에 비례 배분한다(큰 셀일수록 더 읽어야 함).
    """
    out = strata.copy()
    total_N = float(out["N_corpus"].sum()) or 1.0
    needs = []
    for _, r in out.iterrows():
        N = float(r["N_corpus"])
        cell_target = max(target_max_missed * N / total_N, 1.0)
        n_needed = int(r["읽은_편수"])
        k = int(r["발견_Include"])
        # 추가로 읽을 때 Include가 더 나오지 않는다는 가정 하의 최소 n
        while n_needed < int(N) and _cp_upper(n_needed, k) * N > cell_target:
            n_needed += max(1, int(n_needed * 0.1) or 1)
        needs.append(max(0, min(int(N), n_needed) - int(r["읽은_편수"])))
    out["추가_필요_편수"] = needs
    out["목표_달성시_최대누락"] = [
        _cp_upper(int(r["읽은_편수"]) + n, int(r["발견_Include"])) * float(r["N_corpus"])
        for n, (_, r) in zip(needs, out.iterrows())
    ]
    return out


def build_risk_audit_sample(
    predictions: pd.DataFrame,
    sizes: pd.DataFrame,
    exposure_pattern: str | None = AUDIT_EXPOSURE_PATTERN,
    seed: int = 20260101,
) -> pd.DataFrame:
    """셀별 '추가_필요_편수'만큼 자동 제외 집합에서 무작위로 뽑는다(개발 라벨 문헌은 제외)."""
    pool = predictions[predictions["AI_Recommendation"] == "안전 제외 후보"].copy()
    pool["감사_셀"] = _audit_cells(pool, exposure_pattern)
    already = pd.to_numeric(pool.get("Human_Label_Normalized", pd.Series([np.nan] * len(pool))),
                            errors="coerce").isin([0, 1])

    rng = np.random.default_rng(seed)
    picks = []
    for _, r in sizes.iterrows():
        take = int(r.get("추가_필요_편수", 0))
        if take <= 0:
            continue
        grp = pool[(pool["감사_셀"] == r["감사_셀"]) & (~already)]
        if grp.empty:
            continue
        take = int(min(take, len(grp)))
        picks.append(grp.iloc[np.sort(rng.choice(len(grp), size=take, replace=False))])
    if not picks:
        raise ValueError("추가로 읽어야 할 문헌이 없습니다(이미 목표 상한을 만족했거나, 남은 자동 제외 문헌이 모두 validation 라벨 문헌입니다).")

    out = pd.concat(picks).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    out.insert(0, "Audit_No", np.arange(1, len(out) + 1))
    out.insert(1, "Audit_Label", "")
    front = [c for c in ["Audit_No", "Audit_Label", "감사_셀", "제목", "Title", "초록", "Abstract",
                         "Gate_Fail_Reason", "AI_Probability_%"] if c in out.columns]
    return out[front + [c for c in out.columns if c not in front]]


def summarize_audit(strata_after: pd.DataFrame, total_include_est: float | None = None) -> dict:
    """감사 후 전체 누락 상한과 해석 문구를 만든다."""
    max_missed = float(strata_after["최대_누락_추정"].sum())
    read = int(strata_after["읽은_편수"].sum())
    found = int(strata_after["발견_Include"].sum())
    out = {
        "audited_n": read,
        "found_include": found,
        "max_missed_upper": max_missed,
        "pool_n": int(strata_after["N_corpus"].sum()),
        "confidence": AUDIT_CONFIDENCE,
    }
    if total_include_est:
        out["total_include_est"] = float(total_include_est)
        out["max_missed_share"] = max_missed / float(total_include_est)
    return out


def audit_size_options(strata: pd.DataFrame, targets=(100.0, 50.0, 25.0, 10.0)) -> pd.DataFrame:
    """목표 상한별로 '몇 편을 더 읽어야 하는지'를 표로 만든다.

    목표를 하나로 강제하지 않는 이유: 셀이 크면 상한을 낮추는 비용이 급격히 커진다.
    연구자가 비용과 보고 가능한 상한을 보고 직접 고르는 편이 정직하다.
    """
    rows = []
    for t in targets:
        rec = recommend_audit_sizes(strata, target_max_missed=float(t))
        rows.append({
            "목표_최대누락": float(t),
            "추가_읽을_편수": int(rec["추가_필요_편수"].sum()),
            "달성시_최대누락": float(rec["목표_달성시_최대누락"].sum()),
        })
    return pd.DataFrame(rows)


def known_item_recovery(
    predictions: pd.DataFrame,
    known_titles: list[str],
    threshold: float = 0.72,
) -> tuple[pd.DataFrame, dict]:
    """알려진 적격 문헌(seed study)이 자동 제외되지 않았는지 확인한다.

    동료심사에서 통계적 상한보다 설득력이 큰 증거다. 연구계획서 인용문헌, 선행 리뷰의
    포함문헌, 다른 경로로 이미 찾은 적격 문헌의 제목을 넣으면, 각각이 어느 티어에
    배치되었는지와 코퍼스에서 찾았는지를 돌려준다.
    자동 제외(안전 제외 후보)로 간 문헌이 한 편이라도 있으면 그 자체가 반증이다.
    """
    titles = predictions.get("Title", predictions.get("제목", pd.Series([""] * len(predictions))))
    titles = titles.fillna("").astype(str)
    norm = titles.str.lower().str.replace(r"[^a-z0-9가-힣 ]", " ", regex=True).str.split().str.join(" ")

    rows = []
    for q in known_titles:
        qn = re.sub(r"[^a-z0-9가-힣 ]", " ", str(q).lower())
        qn = " ".join(qn.split())
        if not qn:
            continue
        qset = set(qn.split())
        best_i, best_s = -1, 0.0
        for i, t in enumerate(norm):
            tset = set(t.split())
            if not tset:
                continue
            s = len(qset & tset) / max(len(qset), 1)
            if s > best_s:
                best_i, best_s = i, s
        if best_i >= 0 and best_s >= threshold:
            rows.append({"입력_제목": q, "매칭_제목": titles.iloc[best_i], "유사도": round(best_s, 3),
                         "배치": predictions["AI_Recommendation"].iloc[best_i],
                         "AI_확률_%": predictions.get("AI_Probability_%", pd.Series([np.nan] * len(predictions))).iloc[best_i]})
        else:
            rows.append({"입력_제목": q, "매칭_제목": "(코퍼스에서 찾지 못함)", "유사도": round(best_s, 3),
                         "배치": "미검색", "AI_확률_%": np.nan})

    out = pd.DataFrame(rows)
    found = out[out["배치"] != "미검색"]
    auto_excluded = out[out["배치"] == "안전 제외 후보"]
    stats = {
        "n_known": int(len(out)),
        "n_found_in_corpus": int(len(found)),
        "n_auto_excluded": int(len(auto_excluded)),
        "recovery_rate": float(len(found[found["배치"] != "안전 제외 후보"]) / len(out)) if len(out) else 0.0,
        "passed": bool(len(out) > 0 and len(auto_excluded) == 0),
    }
    return out, stats


# ---------------------------------------------------------------------------
# 라벨 일관성 점검
# ---------------------------------------------------------------------------
# 같은 PECO로 판정했다면 Include끼리는 어휘가 겹쳐야 한다. 한 편만 성격이 전혀 다르면
# 기준 해석이 흔들렸다는 신호이고, 그 한 편이 임계값과 자동 제외 정책 전체를 좌우한다.
# (실제로 간문맥 혈역학 논문 몇 편이 Include로 들어오자 ROC-AUC가 0.85에서 0.53으로 떨어졌다.)
# ---------------------------------------------------------------------------

def label_consistency_check(predictions: pd.DataFrame, flag_quantile: float = 0.34) -> tuple[pd.DataFrame, dict]:
    """Include 라벨끼리의 어휘 유사도를 보고, 성격이 동떨어진 Include를 표시한다.

    반환 표의 '재확인_권고'가 True인 문헌은 적격 기준을 다시 적용해 볼 대상이다.
    자동으로 라벨을 바꾸지는 않는다 — 판단은 연구자 몫이다.
    """
    lab = pd.to_numeric(predictions.get("Human_Label_Normalized", pd.Series([np.nan] * len(predictions))),
                        errors="coerce")
    inc = predictions[lab == 1].copy()
    exc = predictions[lab == 0].copy()
    if len(inc) < 3:
        return pd.DataFrame(), {"n_include": int(len(inc)), "flagged": 0, "note": "Include가 3편 미만이라 점검 불가"}

    def _txt(df):
        t = df.get("Title", df.get("제목", pd.Series([""] * len(df)))).fillna("").astype(str)
        a = df.get("Abstract", df.get("초록", pd.Series([""] * len(df)))).fillna("").astype(str)
        return (t + " " + a).str.strip().tolist()

    inc_txt, exc_txt = _txt(inc), _txt(exc)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, stop_words="english")
    mat = vec.fit_transform(inc_txt + exc_txt)
    M_inc = mat[:len(inc_txt)]
    M_exc = mat[len(inc_txt):]

    sim_inc = cosine_similarity(M_inc, M_inc)
    np.fill_diagonal(sim_inc, 0.0)
    best_inc = sim_inc.max(axis=1)
    best_exc = cosine_similarity(M_inc, M_exc).max(axis=1) if M_exc.shape[0] else np.zeros(len(inc_txt))

    cut = float(np.quantile(best_inc, flag_quantile))
    out = pd.DataFrame({
        "제목": inc.get("Title", inc.get("제목")).to_numpy(),
        "다른_Include와_최대유사도": np.round(best_inc, 3),
        "Exclude와_최대유사도": np.round(best_exc, 3),
        "AI_확률": np.round(pd.to_numeric(inc.get("AI_Probability", pd.Series([np.nan] * len(inc))),
                                        errors="coerce").to_numpy(), 3),
        "재확인_권고": (best_inc <= cut) | (best_exc > best_inc),
    }).sort_values("다른_Include와_최대유사도").reset_index(drop=True)

    stats = {
        "n_include": int(len(inc)),
        "flagged": int(out["재확인_권고"].sum()),
        "median_inc_similarity": float(np.median(best_inc)),
    }
    return out, stats


def _content_keys(titles, abstracts) -> np.ndarray:
    """레코드 내용(정규화 제목+초록)의 SHA-256. 행 순서와 무관한 정렬 키로 쓴다."""
    t = pd.Series(titles).fillna("").astype(str).str.strip().str.casefold()
    a = pd.Series(abstracts).fillna("").astype(str).str.strip().str.casefold()
    joined = (t + "\u001f" + a).tolist()
    return np.array([hashlib.sha256(x.encode("utf-8", errors="ignore")).hexdigest() for x in joined], dtype=object)


def _fingerprint_extra(mode: str, criteria_text: str = "", exclusion_text: str = "",
                       gate_rules=None, recall_target=None) -> str:
    rules_txt = "\u0003".join(f"{n}\u0004{p}" for n, p in (gate_rules or []))
    crit = hashlib.sha256(f"{criteria_text}\u0005{exclusion_text}".encode("utf-8")).hexdigest()[:16]
    rt = "" if recall_target is None else f"{float(recall_target):.4f}"
    emb = f"emb={EMBEDDING_MODEL_NAME if embeddings_available() else 'none'}"
    return f"{mode}|pico={crit}|rules={hashlib.sha256(rules_txt.encode('utf-8')).hexdigest()[:16]}|rt={rt}|{emb}"


def run_fingerprint(df: pd.DataFrame, extra: str = "") -> str:
    """입력(제목·초록·라벨) + 알고리즘 버전 + 실행 설정(extra)의 해시.

    V36: 레코드별 해시를 정렬한 뒤 합치므로 행 순서가 달라도 같은 코퍼스면 같은 지문이 나온다.
    extra에는 PICO 해시, 규칙 해시, 목표 재현율, 임베딩 사용 여부가 들어간다(_fingerprint_extra).
    """
    title_col = _find_col(df, ["title", "제목"]) or ""
    abs_col = _find_col(df, ["abstract", "초록"]) or ""
    lab_col = _find_col(df, ["human_label", "human_label_normalized", "label"]) or ""
    n = len(df)
    cols = []
    for col in (title_col, abs_col, lab_col):
        cols.append(df[col].fillna("").astype(str).tolist() if col and col in df.columns else [""] * n)
    rows = sorted(hashlib.sha256("\u0001".join(v).encode("utf-8", errors="ignore")).hexdigest()
                  for v in zip(*cols)) if n else []
    payload = ("\u0002".join(rows) + f"|{ALGORITHM_VERSION}|seed={RANDOM_SEED}|{extra}").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16].upper()


def software_versions() -> dict:
    """재현성 기록용 실행 환경 버전."""
    import platform
    import sys
    import scipy
    import sklearn
    out = {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__, "algorithm_version": ALGORITHM_VERSION, "random_seed": RANDOM_SEED,
        "embedding_model": EMBEDDING_MODEL_NAME if embeddings_available() else "not installed (TF-IDF fallback)",
    }
    try:
        import sentence_transformers as _st_mod
        out["sentence_transformers"] = _st_mod.__version__
    except Exception:
        out["sentence_transformers"] = "not installed"
    return out


def build_run_manifest(result: "ScreeningResult", criteria_text: str = "", exclusion_text: str = "",
                       gate_rules=None, corpus: pd.DataFrame | None = None) -> bytes:
    """실행 1회의 재현성 기록(JSON). 입력 해시·PICO 원문·규칙·버전·출력 해시를 남긴다."""
    import json
    m = result.metrics or {}
    pred = result.predictions
    out_cols = [c for c in ("Title", "AI_Recommendation", "Operational_Action") if c in pred.columns]
    out_rows = sorted(hashlib.sha256("\u0001".join(map(str, r)).encode("utf-8", errors="ignore")).hexdigest()
                      for r in pred[out_cols].itertuples(index=False)) if out_cols else []
    corpus_hash = ""
    if corpus is not None and len(corpus):
        corpus_hash = hashlib.sha256(corpus.to_csv(index=False).encode("utf-8", errors="ignore")).hexdigest()
    manifest = {
        "sr_studio_version": ALGORITHM_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": m.get("mode", "supervised"),
        "run_fingerprint": m.get("run_fingerprint", ""),
        "input_corpus_sha256": corpus_hash,
        "pico_text": criteria_text,
        "pico_sha256": hashlib.sha256(criteria_text.encode("utf-8")).hexdigest(),
        "exclusion_text": exclusion_text,
        "gate_rules": [list(r) for r in (gate_rules if gate_rules is not None else m.get("gate_rules_input", []))],
        "policy": {k: m.get(k) for k in ("recall_target", "threshold", "safe_cutoff", "auto_exclusion_enabled",
                                         "quality_gate_status") if k in m},
        "software": m.get("software") or software_versions(),
        "output_sha256": hashlib.sha256("|".join(out_rows).encode("utf-8")).hexdigest(),
        "n_records": int(len(pred)),
    }
    return json.dumps(manifest, ensure_ascii=False, indent=2, default=str).encode("utf-8")


def bootstrap_extension_probabilities(
    corpus: pd.DataFrame,
    criteria_text: str,
    exclusion_text: str = "",
) -> np.ndarray:
    """학습된 모델이 없을 때(Include가 너무 적어 교차검증 자체가 불가능할 때) 쓰는 확률.

    적격 문헌이 2~3편뿐이면 모델을 만들 수 없고, 그러면 validation 표본을 늘릴 방법도
    없어지는 교착이 생긴다. 이때는 라벨이 필요 없는 PICO 유사도(zero-shot) 점수로
    확률구간을 나눠 추가 표본을 뽑는다. 사후층화에 쓰는 공변량은 라벨과 무관하기만 하면
    되므로, 모델 확률 대신 zero-shot 점수를 써도 설계의 불편성은 유지된다.
    """
    zs = zero_shot_screen(corpus, criteria_text=criteria_text, exclusion_text=exclusion_text)
    pred = zs.predictions
    col = "AI_Probability" if "AI_Probability" in pred.columns else "Combined_Score"
    score = pd.to_numeric(pred[col], errors="coerce").fillna(0.0).to_numpy()
    # Combined_Score는 0~1 척도가 아니므로 min-max로 맞춘다(확률구간 분할에만 쓰인다).
    lo, hi = float(np.nanmin(score)), float(np.nanmax(score))
    score = (score - lo) / (hi - lo) if hi > lo else np.zeros_like(score)
    out = np.zeros(len(corpus), dtype=float)
    out[pred["_Corpus_Row"].to_numpy()] = score
    return out


def estimate_include_yield(labeled: pd.DataFrame) -> pd.DataFrame:
    """라벨된 validation 표본에서 층별 Include 수확량과 코퍼스 투영치를 계산한다.

    '200편을 다 읽었는데 Include가 2편'인 상황을 라벨링 직후 바로 알 수 있게 한다.
    추가로 몇 편을 어디서 읽어야 하는지는 plan_validation_extension이 계산한다.
    """
    d = labeled.copy()
    lab = d.get("Human_Label_Normalized")
    if lab is None:
        lab = d["Human_Label"].map(_normalize_label_value)
    lab = pd.to_numeric(lab, errors="coerce")
    strat = d.get("Training_Stratum", pd.Series(["(층 정보 없음)"] * len(d)))
    w = pd.to_numeric(d.get("Sampling_Weight", pd.Series([1.0] * len(d))), errors="coerce").fillna(1.0)

    rows = []
    for st_name, grp in d.assign(_l=lab, _w=w, _s=strat).groupby("_s"):
        n = int(grp["_l"].isin([0, 1]).sum())
        k = int((grp["_l"] == 1).sum())
        rows.append({
            "층": st_name,
            "라벨_편수": n,
            "Include": k,
            "층내_유병률": (k / n) if n else 0.0,
            "코퍼스_투영_Include": float(grp.loc[grp["_l"] == 1, "_w"].sum()),
        })
    out = pd.DataFrame(rows).sort_values("Include", ascending=False).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# 규칙 기반 단독 모드 (rule-only screening)
# ---------------------------------------------------------------------------
# 적격 문헌이 몇 편 없으면 지도학습 모델은 만들 수 없다. 그런데 실제로 자동 제외를 만들어낸
# 것은 모델이 아니라 규칙 게이트였다(확률로 제외된 문헌은 0편이었다). 그렇다면 모델 없이
# 규칙만으로 선별하고, 라벨 200편은 '학습 데이터'가 아니라 '규칙의 안전성 점검'으로 쓰는 편이
# 정직하고 안정적이다.
#
# 이 모드의 성질:
#   - 완전 결정론. 폴드 분할도, 확률 추정도 없으므로 실행 간 편차가 원리적으로 0이다.
#   - Include 수에 흔들리지 않는다. 라벨은 "규칙이 적격 문헌을 떨어뜨리지 않는가"만 확인한다.
#   - 제외 근거가 전부 문장으로 설명된다(어떤 용어군이 없어서 빠졌는지).
#   - 대신 '우선 검토' 순위의 정밀도는 포기한다. 읽는 순서는 PICO 유사도로만 정한다.
# ---------------------------------------------------------------------------

RULE_ONLY_MIN_LABELED_PASS = 20

# 사람 검토 대상 안에서의 '읽는 순서' 표시. 제외 결정과 무관한 참고용 색 구분이다.
# 임계값을 만들지 않으므로 밴드 경계가 바뀌어도 무엇을 읽을지는 달라지지 않는다.
REVIEW_BAND_ORDER = ["상", "중", "하"]
REVIEW_BAND_COLORS = {"상": "D3E8D3", "중": "FBF0C4", "하": "EEF0F3"}
REVIEW_BAND_CUTS = (0.34, 0.67)   # 사람 검토 집합 내부의 분위수


def rule_only_screen(
    df: pd.DataFrame,
    criteria_text: str,
    exclusion_text: str = "",
    gate_rules: list[tuple[str, str]] | None = None,
) -> ScreeningResult:
    """모델 없이 규칙 게이트만으로 선별하고, 라벨은 규칙 점검에만 쓴다."""
    data, _ = prepare_screening_data(df)
    texts = (data["Title"].fillna("") + " " + data["Abstract"].fillna("")).to_numpy()
    # prepare_screening_data는 정규화된 라벨을 Human_Label(0/1/NaN)로 돌려준다.
    y_all = pd.to_numeric(data.get("Human_Label", pd.Series([np.nan] * len(data), index=data.index)),
                          errors="coerce")
    data["Human_Label_Normalized"] = y_all
    # 표본가중치 등 원본의 부가 열을 그대로 가져온다(코퍼스 투영·보고서에 필요).
    src = df.reset_index(drop=True)
    for col in ("Sampling_Weight", "Training_Stratum", "Validation_Record_ID", "_Source_Index"):
        if col in src.columns and col not in data.columns:
            data[col] = src[col].to_numpy()
    labeled_pos = np.flatnonzero(y_all.isin([0, 1]).to_numpy())
    y = y_all.iloc[labeled_pos].astype(int).to_numpy() if len(labeled_pos) else np.array([], dtype=int)
    weights = (pd.to_numeric(data["Sampling_Weight"], errors="coerce").fillna(1.0).to_numpy()[labeled_pos]
               if "Sampling_Weight" in data.columns and len(labeled_pos) else None)


    no_abs = _missing_abstract_mask(data["Abstract"].to_numpy())
    # 이전 단계에서 이미 사람 검토로 확정된 문헌은 규칙 대상에서 제외한다.
    # (초록을 나중에 확보해도 자동 제외로 되돌리지 않기 위한 장치)
    locked = _human_review_lock_mask(df.reset_index(drop=True))
    exempt = no_abs | locked
    gate = build_gate(texts, labeled_pos, y, weights, gate_rules, exempt=exempt)

    # 지도학습 모드와 달리 Include 수 하한을 두지 않는다. 게이트의 근거는 "적격이려면 반드시
    # 등장하는 용어"라는 논리이고, 라벨은 그 논리가 깨지지 않았는지(FN=0) 확인할 뿐이다.
    rules_in = list(gate_rules if gate_rules is not None else GATE_RULES_DEFAULT)
    kept = gate["kept_rules"]
    pass_mask = gate["pass_mask"]
    reasons = gate["reasons"]
    if not gate["active"] and kept:
        rules_kept = [(n, p) for n, p in rules_in if n in kept]
        masks = _gate_rule_masks(texts, rules_kept)
        pass_mask = np.ones(len(texts), dtype=bool)
        why = [[] for _ in range(len(texts))]
        for name, _p in rules_kept:
            m = masks[name] | exempt
            pass_mask &= m
            for i in np.flatnonzero(~m):
                why[i].append(name)
        reasons = np.array(["; ".join(r) for r in why], dtype=object)
    lab_pass = pass_mask[labeled_pos] if len(labeled_pos) else np.array([], dtype=bool)
    gate_fn = int(((y == 1) & ~lab_pass).sum()) if len(labeled_pos) else 0
    inc_pass = int(((y == 1) & lab_pass).sum()) if len(labeled_pos) else 0
    # V36: 라벨이 없거나 Include가 너무 적으면 'FN=0'은 아무것도 증명하지 못한다.
    # (V35는 라벨 0개일 때 min(20, 0)=0이 되어 규칙이 무조건 적용되는 구멍이 있었다.)
    usable = (bool(kept) and len(labeled_pos) > 0 and gate_fn == 0
              and inc_pass >= GATE_MIN_INCLUDE_AFTER
              and int(lab_pass.sum()) >= min(RULE_ONLY_MIN_LABELED_PASS, len(labeled_pos)))

    # Jackknife 점검: Include 한 편을 빼고 규칙을 다시 고르면, 그 Include가 떨어지는가.
    # 규칙 선택에 쓰지 않은 Include에 대한 정직한 누락 추정이다.
    jk_fn = 0
    if len(labeled_pos) and rules_in and gate.get("masks"):
        masks_lab = {n: np.asarray(m)[labeled_pos] for n, m in gate["masks"].items()}
        inc_idx = np.flatnonzero(y == 1)
        for i in inc_idx:
            others = np.ones(len(y), dtype=bool)
            others[i] = False
            sel = [n for n, m in masks_lab.items() if int(((y == 1) & others & ~m).sum()) == 0]
            if sel and not all(masks_lab[n][i] for n in sel):
                jk_fn += 1
        if jk_fn > 0:
            usable = False
    n_inc_lab = int((y == 1).sum()) if len(labeled_pos) else 0

    # 읽는 순서: PICO 유사도(zero-shot). 제외 여부에는 쓰지 않는다.
    score = bootstrap_extension_probabilities(data, criteria_text, exclusion_text)

    rec = np.where(no_abs, MANUAL_REVIEW_TIER,
                   np.where((pass_mask | locked) if usable else True, "사람 검토", "안전 제외 후보"))
    out = data.copy()
    out["_Corpus_Row"] = np.arange(len(out), dtype=int)
    out["Gate_Pass"] = pass_mask
    out["Gate_Fail_Reason"] = reasons
    out["No_Abstract"] = no_abs
    # 이번 실행에서 사람 검토로 분류된 문헌은 다음 실행에서도 그대로 유지되도록 표시한다.
    out[HUMAN_REVIEW_LOCK_COL] = (rec != "안전 제외 후보")
    out["PICO_Similarity"] = np.round(score, 4)
    out["AI_Probability"] = score            # 순서 표시용. 확률 해석을 하지 않는다.
    out["AI_Probability_%"] = np.round(score * 100, 1)
    out["AI_Recommendation"] = rec
    out["Unanimous_Exclude"] = (rec == "안전 제외 후보")

    # 사람 검토 대상 안에서만 PICO 유사도 분위수로 상/중/하를 매긴다.
    # 전부 읽는다는 사실은 바뀌지 않고, 어느 쪽부터 읽을지 눈으로 구분만 해 준다.
    band = np.array([""] * len(out), dtype=object)
    review = (rec == "사람 검토")
    if review.sum() >= 3:
        pos = np.flatnonzero(review)
        v = out.loc[review, "PICO_Similarity"].to_numpy()
        # 값 분위수는 동점이 많으면 한 밴드로 몰린다. 순위로 3등분한다(동점은 안정 정렬).
        order = np.argsort(-v, kind="stable")
        rank = np.empty(len(v), dtype=int)
        rank[order] = np.arange(len(v))
        frac = rank / max(len(v) - 1, 1)
        band[pos] = np.where(frac <= 1 - REVIEW_BAND_CUTS[1], "상",
                             np.where(frac <= 1 - REVIEW_BAND_CUTS[0], "중", "하"))
    elif review.sum():
        band[np.flatnonzero(review)] = "중"
    band[rec == MANUAL_REVIEW_TIER] = "초록없음"
    out["검토_우선도"] = band
    out = out.sort_values(["AI_Recommendation", "PICO_Similarity"], ascending=[True, False]).reset_index(drop=True)

    w_all = pd.to_numeric(data.get("Sampling_Weight", pd.Series([1.0] * len(data))), errors="coerce").fillna(1.0)
    metrics = {
        "mode": "rule_only",
        "algorithm_version": ALGORITHM_VERSION,
        "random_seed": int(RANDOM_SEED),
        "run_fingerprint": run_fingerprint(
            data, extra=_fingerprint_extra("rule_only", criteria_text, exclusion_text, rules_in)),
        "gate_rules_input": [list(r) for r in rules_in],
        "gate_jackknife_fn": int(jk_fn),
        "rule_retention_lower_ci": (recall_lower_confidence_bound(n_inc_lab, gate_fn, 0.95) if n_inc_lab else 0.0),
        "software": software_versions(),
        "n_total": int(len(data)),
        "labeled_n": int(len(labeled_pos)),
        "include_n": int((y == 1).sum()) if len(labeled_pos) else 0,
        "gate_active": bool(usable),
        "gate_kept_rules": list(kept),
        "gate_dropped_rules": dict(gate["dropped_rules"]),
        "gate_fn_on_labels": gate_fn,
        "auto_exclusion_enabled": bool(usable),
        "auto_excluded_n": int((rec == "안전 제외 후보").sum()),
        "human_review_n": int((rec != "안전 제외 후보").sum()),
        "no_abstract_n": int(no_abs.sum()),
        "quality_gate_status": "PASS" if usable else "REVIEW",
        "quality_gate_reasons": ([] if usable else [r for r in [
            "자동 제외 규칙이 입력되지 않음(PICO 화면의 「자동 제외 규칙」)" if not rules_in else "",
            "라벨이 없음" if not len(labeled_pos) else "",
            f"규칙이 human Include {gate_fn}편을 제외함" if gate_fn else "",
            (f"규칙 통과 Include {inc_pass}편 < {GATE_MIN_INCLUDE_AFTER}편 — FN=0이 근거가 되지 못함"
             if rules_in and len(labeled_pos) and inc_pass < GATE_MIN_INCLUDE_AFTER else ""),
            f"Jackknife 점검에서 Include {jk_fn}편이 규칙에 의해 떨어짐" if jk_fn else "",
            (f"규칙을 통과한 라벨 문헌 {int(lab_pass.sum())}편 < {min(RULE_ONLY_MIN_LABELED_PASS, len(labeled_pos))}편 — 검증 표본 부족"
             if rules_in and len(labeled_pos) and int(lab_pass.sum()) < min(RULE_ONLY_MIN_LABELED_PASS, len(labeled_pos)) else ""),
            "적용 가능한 규칙이 없음" if rules_in and not kept else "",
        ] if r]),
        "deterministic": True,
    }
    if len(labeled_pos):
        metrics["labeled_include_retained"] = int(((y == 1) & lab_pass).sum())
        metrics["weighted_excluded_share"] = float(
            w_all.to_numpy()[labeled_pos][~lab_pass].sum() / max(w_all.to_numpy()[labeled_pos].sum(), 1e-9))
    return ScreeningResult(predictions=out, metrics=metrics, threshold=float("nan"), confusion={})


# ---------------------------------------------------------------------------
# 사람 검토 고정(lock) — 한 번 사람 검토로 간 문헌은 다시 자동 제외하지 않는다
# ---------------------------------------------------------------------------
# 초록이 없어 "정보 부족 → 사람 검토"로 분류된 문헌은, 나중에 PubMed나 출판사에서 초록을
# 확보하더라도 규칙에 다시 넣지 않는다. 다시 넣으면 "기계가 초록 없는 문헌을 버리지 않는다"는
# 보수성 주장이 깨지고, 같은 문헌의 운명이 초록 확보 시점에 따라 달라진다.
HUMAN_REVIEW_LOCK_COL = "Human_Review_Locked"


def _human_review_lock_mask(df: pd.DataFrame) -> np.ndarray:
    """입력에 Human_Review_Locked 열이 있으면 그 문헌은 규칙과 무관하게 사람 검토로 보낸다."""
    if HUMAN_REVIEW_LOCK_COL not in df.columns:
        return np.zeros(len(df), dtype=bool)
    truthy = {"1", "true", "t", "y", "yes", "o", "lock", "locked", "사람검토", "사람 검토"}
    return df[HUMAN_REVIEW_LOCK_COL].fillna("").astype(str).str.strip().str.lower().isin(truthy).to_numpy()


# ---------------------------------------------------------------------------
# 논문용 성능 Figure (규칙 기반 safe-exclusion 전용)
# ---------------------------------------------------------------------------
# ROC/PR/AUC/F1은 확률 분류기의 지표이고 결정론적 규칙에는 정의되지 않는다. 대신
#   A. 사람이 읽는 비율 대비 적격 문헌 보존율 (screening efficiency)
#   B. 검증 집합별 적격 문헌 보존율과 95% CI (safety) — 표준 forest plot 형식
#   C. 작업량 감소와 잘못된 자동 제외 건수 (utility)
# 를 그린다. 3패널 합본과 패널별 개별 파일을 모두 만든다.
# 주석은 전부 축 바깥(아래)에 두어 데이터와 겹치지 않게 한다.
# ---------------------------------------------------------------------------

RETENTION_KIND_DEV = "development"
RETENTION_KIND_IND = "independent"

_FIG_RC = {
    "font.size": 9, "axes.titlesize": 10.5, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8,
    "font.family": "DejaVu Sans", "axes.linewidth": 0.8,
}
_C_BLUE, _C_GREY, _C_DARK, _C_LIGHT = "#1A56DB", "#9AA0A6", "#111827", "#C9D2E4"


def retention_ci(retained: int, total: int, confidence: float = 0.95):
    """보존율과 Clopper-Pearson 양측 신뢰구간. total=0이면 (nan, 0, 1)."""
    if total <= 0:
        return (float("nan"), 0.0, 1.0)
    p = retained / total
    alpha = 1.0 - confidence
    lo = 0.0 if retained == 0 else float(_beta_dist.ppf(alpha / 2, retained, total - retained + 1))
    hi = 1.0 if retained == total else float(_beta_dist.ppf(1 - alpha / 2, retained + 1, total - retained))
    return (p, lo, hi)


def _fig_inputs(result: ScreeningResult, retention_sets):
    m = result.metrics
    pred = result.predictions
    total_n = int(m.get("n_total", len(pred)))
    auto_n = int(m.get("auto_excluded_n", int((pred["AI_Recommendation"] == "안전 제외 후보").sum())))
    human_n = max(total_n - auto_n, 0)
    if not retention_sets:
        inc = int(m.get("include_n", 0))
        retention_sets = [{"label": "Internal QC (rule development)",
                           "retained": int(m.get("labeled_include_retained", inc)),
                           "total": inc, "kind": RETENTION_KIND_DEV}]
    rows = [r for r in retention_sets if int(r.get("total", 0)) > 0]
    obs_ret = sum(int(r["retained"]) for r in rows)
    obs_tot = sum(int(r["total"]) for r in rows)
    if len(rows) > 1 and obs_tot:
        rows = rows + [{"label": "All sets combined", "retained": obs_ret,
                        "total": obs_tot, "kind": "overall"}]
    return dict(total_n=total_n, auto_n=auto_n, human_n=human_n,
                human_pct=100.0 * human_n / total_n if total_n else 0.0,
                rows=rows, obs_ret=obs_ret, obs_tot=obs_tot)


def _draw_panel_a(ax, d, title="A  Screening efficiency", note_ax=None):
    """사람이 읽는 비율 대비 적격 보존율. 주석은 축 아래 바깥에 둔다."""
    from matplotlib.lines import Line2D
    hp = d["human_pct"]
    obs = 100.0 * d["obs_ret"] / d["obs_tot"] if d["obs_tot"] else float("nan")
    ax.plot([0, 100], [0, 100], ls="--", lw=1.0, color=_C_GREY)
    ax.plot([hp], [obs], marker="o", ms=9, color=_C_BLUE, zorder=5)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Records requiring human screening (%)")
    ax.set_ylabel("Eligible records retained (%)")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(handles=[
        Line2D([], [], ls="--", color=_C_GREY, label="Random / unaided screening"),
        Line2D([], [], marker="o", ls="none", color=_C_BLUE, ms=7,
               label="Rule-based safe exclusion"),
    ], loc="lower right", frameon=False, handlelength=1.8)
    (note_ax or ax).text(
        0, -0.235,
        "Operating point: {:.1f}% of records screened by humans; "
        "{:.0f}% of eligible records retained ({}/{}).".format(hp, obs, d["obs_ret"], d["obs_tot"]),
        transform=ax.transAxes, ha="left", va="top", fontsize=8.2, color="#374151")


def _draw_panel_b(fig, gs_cell, d, title="B  Safety of automated exclusion", audit=None):
    """표준 forest plot: 왼쪽 라벨 열 · 가운데 CI · 오른쪽 수치 열 (서로 겹치지 않음)."""
    from matplotlib.lines import Line2D
    rows = d["rows"]
    inner = gs_cell.subgridspec(1, 3, width_ratios=[1.30, 1.55, 0.95], wspace=0.0)
    axL = fig.add_subplot(inner[0, 0])
    axM = fig.add_subplot(inner[0, 1])
    axR = fig.add_subplot(inner[0, 2])
    for a in (axL, axR):
        a.axis("off")
        a.set_ylim(-1.0, len(rows) - 0.3)
        a.set_xlim(0, 1)

    ypos = np.arange(len(rows))[::-1]
    style = {RETENTION_KIND_DEV: ("o", _C_GREY, "white"),
             RETENTION_KIND_IND: ("o", _C_BLUE, _C_BLUE),
             "overall": ("D", _C_DARK, _C_DARK)}
    axL.text(1.0, len(rows) - 0.55, "Validation set", ha="right", va="center",
             fontsize=8.6, fontweight="bold")
    axR.text(0.02, len(rows) - 0.55, "Retention (95% CI)", ha="left", va="center",
             fontsize=8.6, fontweight="bold")
    for yv, r in zip(ypos, rows):
        p, lo, hi = retention_ci(int(r["retained"]), int(r["total"]))
        mk, ec, fc = style.get(str(r.get("kind", RETENTION_KIND_IND)), style[RETENTION_KIND_IND])
        axL.text(1.0, yv + 0.13, str(r["label"]), ha="right", va="center", fontsize=8.4)
        axL.text(1.0, yv - 0.24, "{}/{}".format(int(r["retained"]), int(r["total"])),
                 ha="right", va="center", fontsize=7.8, color="#6B7280")
        axM.hlines(yv, lo * 100, hi * 100, color=ec, lw=1.8)
        for xe in (lo * 100, hi * 100):
            axM.vlines(xe, yv - 0.13, yv + 0.13, color=ec, lw=1.2)
        axM.plot([p * 100], [yv], marker=mk, ms=7.5, mec=ec, mfc=fc, zorder=5)
        axR.text(0.02, yv, "{:.0f}%  ({:.0f}–{:.0f})".format(p * 100, lo * 100, hi * 100),
                 ha="left", va="center", fontsize=8.3, color="#1F2937")

    axM.axvline(100, color="#C9CDD4", lw=0.9, ls=":")
    axM.set_xlim(0, 104)
    axM.set_xticks([0, 25, 50, 75, 100])
    axM.set_ylim(-1.0, len(rows) - 0.3)
    axM.set_yticks([])
    axM.spines["left"].set_visible(False)
    axM.set_xlabel("Eligible-record retention (%)")
    axM.set_title(title, loc="left", fontweight="bold")
    axM.legend(handles=[
        Line2D([], [], marker="o", ls="none", mec=_C_GREY, mfc="white", ms=6.5,
               label="Development set (not independent)"),
        Line2D([], [], marker="o", ls="none", mec=_C_BLUE, mfc=_C_BLUE, ms=6.5,
               label="Independent set"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1,
        frameon=False, handlelength=1.2, borderpad=0.1)
    # 무작위 감사는 '보존율'이 아니라 '놓친 문헌 수의 상한'으로 말해야 한다.
    # 적격 문헌을 한 편도 찾지 못했다면 분모가 0이라 비율 자체가 정의되지 않기 때문이다.
    if audit and int(audit.get("n_read", 0)) > 0:
        n_read = int(audit["n_read"]); n_found = int(audit.get("n_found", 0))
        pool = int(audit.get("pool_n", 0))
        ub = _cp_upper(n_read, n_found)
        txt = ("Independent random audit of the automatically excluded set: "
               "{} records re-screened, {} eligible found"
               .format(n_read, n_found))
        if pool:
            txt += "; \u2264{:.0f} eligible records missed (95% upper bound)".format(ub * pool)
        axM.text(0.0, -0.33, txt, transform=axM.transAxes, ha="left", va="top",
                 fontsize=8.0, color="#374151")
    return axM


def _draw_panel_c(ax, d, title="C  Human workload"):
    """작업량 막대. 요약 주석은 축 아래 바깥에 둔다."""
    from matplotlib.lines import Line2D
    hp, ap = d["human_pct"], 100.0 - d["human_pct"]
    ax.barh([1], [100], color="#D5D9E0", height=0.4)
    ax.barh([0], [hp], color=_C_BLUE, height=0.4)
    ax.barh([0], [ap], left=hp, color=_C_LIGHT, height=0.4)
    ax.text(hp + ap / 2, 0, "{:.1f}%".format(ap), va="center", ha="center",
            fontsize=8.4, color="#1F2937")
    ax.text(hp / 2, 0.33, "{:.1f}%".format(hp), va="bottom", ha="center",
            fontsize=8.4, color=_C_BLUE, fontweight="bold")
    ax.text(50, 1, "{:,} records".format(d["total_n"]), va="center", ha="center",
            fontsize=8.4, color="#1F2937")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Rule-based", "Manual"], fontsize=8.8)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_ylim(-0.55, 1.6)
    ax.set_xlabel("Proportion of records (%)")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(handles=[
        Line2D([], [], color=_C_BLUE, lw=7, label="Human screening"),
        Line2D([], [], color=_C_LIGHT, lw=7, label="Automatically excluded"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2,
        frameon=False, handlelength=1.1, borderpad=0.1)
    ax.text(0, -0.52,
            "Human screening {:,} · automatically excluded {:,} · "
            "observed false exclusions {}".format(
                d["human_n"], d["auto_n"], d["obs_tot"] - d["obs_ret"]),
            transform=ax.transAxes, ha="left", va="top", fontsize=8.2, color="#374151")


def _save_fig(fig, formats, dpi):
    out = {}
    for fmt in formats:
        buf = io.BytesIO()
        kw = {"dpi": dpi} if fmt in ("png", "tiff") else {}
        if fmt == "tiff":
            kw["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(buf, format=fmt, bbox_inches="tight", facecolor="white", **kw)
        out[fmt] = buf.getvalue()
    return out


def build_rule_performance_figure(
    result: ScreeningResult,
    retention_sets: list | None = None,
    formats: tuple = ("pdf", "svg", "png", "tiff"),
    dpi: int = 600,
    audit: dict | None = None,
) -> dict:
    """3패널 합본과 패널별 개별 파일을 만든다.

    audit: {"n_read", "n_found", "pool_n"} — 무작위 감사 결과를 B 패널 각주로 적는다.

    반환 키: 합본은 'pdf'/'svg'/'png'/'tiff', 개별은 'A_pdf', 'B_png' 같은 형식.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = _fig_inputs(result, retention_sets)
    n_rows = len(d["rows"])
    plt.rcParams.update(_FIG_RC)

    # --- 합본 ---------------------------------------------------------------
    fig = plt.figure(figsize=(14.0, max(4.2, 1.5 + 0.75 * n_rows)))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.65, 1.0], wspace=0.30,
                          left=0.05, right=0.99, top=0.84,
                          bottom=0.34 if audit else 0.26)
    _draw_panel_a(fig.add_subplot(gs[0, 0]), d)
    # 합본에서는 감사 각주를 패널 안에 두면 C 패널 범례와 겹친다. 그림 맨 아래에 따로 적는다.
    _draw_panel_b(fig, gs[0, 1], d)
    _draw_panel_c(fig.add_subplot(gs[0, 2]), d)
    if audit and int(audit.get("n_read", 0)) > 0:
        n_read = int(audit["n_read"]); n_found = int(audit.get("n_found", 0))
        pool = int(audit.get("pool_n", 0))
        note = ("Independent random audit of the automatically excluded set: "
                "{} records re-screened, {} eligible found".format(n_read, n_found))
        if pool:
            note += "; \u2264{:.0f} eligible records missed (95% upper bound)".format(
                _cp_upper(n_read, n_found) * pool)
        fig.text(0.05, 0.035, note, ha="left", va="bottom", fontsize=8.2, color="#374151")
    out = _save_fig(fig, formats, dpi)
    plt.close(fig)

    # --- 개별 패널 ----------------------------------------------------------
    figA = plt.figure(figsize=(5.0, 4.4))
    gsA = figA.add_gridspec(1, 1, left=0.15, right=0.97, top=0.88, bottom=0.26)
    _draw_panel_a(figA.add_subplot(gsA[0, 0]), d, title="Screening efficiency")
    for k, v in _save_fig(figA, formats, dpi).items():
        out["A_" + k] = v
    plt.close(figA)

    figB = plt.figure(figsize=(7.6, max(3.0, 1.4 + 0.78 * n_rows)))
    gsB = figB.add_gridspec(1, 1, left=0.02, right=0.99, top=0.86,
                            bottom=(0.26 if audit else 0.20) + 0.03 * max(0, 3 - n_rows))
    _draw_panel_b(figB, gsB[0, 0], d, title="Safety of automated exclusion", audit=audit)
    for k, v in _save_fig(figB, formats, dpi).items():
        out["B_" + k] = v
    plt.close(figB)

    figC = plt.figure(figsize=(5.4, 3.5))
    gsC = figC.add_gridspec(1, 1, left=0.17, right=0.97, top=0.86, bottom=0.34)
    _draw_panel_c(figC.add_subplot(gsC[0, 0]), d, title="Human workload")
    for k, v in _save_fig(figC, formats, dpi).items():
        out["C_" + k] = v
    plt.close(figC)
    return out


def build_original_order_excel_bytes(
    predictions: pd.DataFrame,
    original_df: pd.DataFrame | None = None,
) -> bytes:
    """업로드한 원본 파일의 행 순서를 그대로 유지한 채 행 색만 입혀 돌려준다.

    PICO 적합도 순으로 정렬된 파일과 달리, 원본에서 몇 번째 문헌인지 그대로 보면서
    색으로 분류만 확인하고 싶을 때 쓴다. 판정 열은 맨 앞에 붙인다.
      회색  = 자동 제외(사람이 읽지 않음)
      초록/노랑/연회색 = 사람 검토 대상의 읽는 순서 밴드(상/중/하)
      노랑  = 초록 없음(무조건 사람 검토)
    """
    df = predictions.copy()
    if "_Corpus_Row" in df.columns:
        df = df.sort_values("_Corpus_Row").reset_index(drop=True)

    if original_df is not None and len(original_df) == len(df):
        base = original_df.reset_index(drop=True).copy()
    else:
        drop = {"Text", "StructuredText", "_export_group"}
        base = df[[c for c in df.columns if c not in drop
                   and not str(c).startswith(("Prob_", "CV_Prob_"))]].copy()

    tier = df.get("AI_Recommendation", pd.Series([""] * len(df))).astype(str)
    band = df.get("검토_우선도", pd.Series([""] * len(df))).astype(str)
    reason = df.get("Gate_Fail_Reason", pd.Series([""] * len(df))).astype(str)
    score = df.get("AI_Probability_%", df.get("PICO_Similarity", pd.Series([np.nan] * len(df))))

    front = pd.DataFrame({
        "AI_판정": tier.to_numpy(),
        "검토_우선도": band.to_numpy(),
        "제외_사유": reason.to_numpy(),
        "PICO_적합도": pd.to_numeric(score, errors="coerce").to_numpy(),
    })
    out = pd.concat([front, base.drop(columns=[c for c in front.columns if c in base.columns],
                                      errors="ignore")], axis=1)

    wb = Workbook()
    ws = wb.active
    ws.title = "원본순서_색표시"
    for row in dataframe_to_rows(out, index=False, header=True):
        ws.append(row)
    header_fill = PatternFill("solid", fgColor="1A56DB")
    header_font = Font(bold=True, color="FFFFFF")
    for c in ws[1]:
        c.fill = header_fill
        c.font = header_font
    ws.freeze_panes = "A2"

    for i in range(len(out)):
        b = str(band.iloc[i])
        t = str(tier.iloc[i])
        color = REVIEW_BAND_COLORS.get(b)
        if color is None:
            color = "B8BDC6" if t == "안전 제외 후보" else ("FFF2CC" if b == "초록없음" else "FFFFFF")
        fill = PatternFill("solid", fgColor=color)
        for j in range(1, ws.max_column + 1):
            ws.cell(row=i + 2, column=j).fill = fill

    widths = {"AI_판정": 20, "검토_우선도": 12, "제외_사유": 34, "PICO_적합도": 13}
    for j, name in enumerate(out.columns, start=1):
        letter = get_column_letter(j)
        if name in widths:
            ws.column_dimensions[letter].width = widths[name]
        elif str(name) in ("제목", "Title"):
            ws.column_dimensions[letter].width = 60
        elif str(name) in ("초록", "Abstract"):
            ws.column_dimensions[letter].width = 80
        else:
            ws.column_dimensions[letter].width = 18

    legend = wb.create_sheet("색_범례")
    legend.append(["색", "의미"])
    for c in legend[1]:
        c.fill = header_fill
        c.font = header_font
    rows = [("상", "사람 검토 — PICO와 가장 가까움"),
            ("중", "사람 검토 — 중간"),
            ("하", "사람 검토 — 먼 쪽"),
            ("초록없음", "초록 없음 — 무조건 사람 검토"),
            ("", "자동 제외(규칙) — 사람이 읽지 않음")]
    for band_name, desc in rows:
        legend.append(["", desc])
        fill_color = REVIEW_BAND_COLORS.get(band_name, "B8BDC6")
        legend.cell(row=legend.max_row, column=1).fill = PatternFill("solid", fgColor=fill_color)
    legend.append([])
    legend.append(["", "색은 읽는 순서를 돕는 표시이며 제외 결정과 무관합니다."])
    legend.append(["", "'사람 검토'로 분류된 문헌은 색과 상관없이 전부 읽어야 합니다."])
    legend.column_dimensions["A"].width = 8
    legend.column_dimensions["B"].width = 70

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
