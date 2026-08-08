from __future__ import annotations

from collections import defaultdict
import re

import pandas as pd
from rapidfuzz.fuzz import token_set_ratio

from utils import normalize_doi, normalize_title, normalize_text, safe_year

FUZZY_TITLE_THRESHOLD = 97.4


def _title_words(value: object) -> str:
    text = normalize_text(value)
    return " ".join(re.findall(r"[0-9a-z가-힣]+", text))


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def deduplicate_records(df: pd.DataFrame, progress_callback=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df.copy(), df.copy()

    work = df.copy().reset_index(drop=True)
    if progress_callback:
        progress_callback("준비", 0, max(len(work), 1))
    for col in ("doi", "title", "abstract", "year"):
        if col not in work.columns:
            work[col] = ""

    work["_doi_key"] = work["doi"].map(normalize_doi)
    work["_title_key"] = work["title"].map(normalize_title)
    work["_title_words"] = work["title"].map(_title_words)
    work["_year_key"] = work["year"].map(safe_year)
    work["_abstract_len"] = work["abstract"].fillna("").astype(str).str.len()

    uf = _UnionFind(len(work))

    # Exact DOI and exact normalized-title matching.
    for key_col in ("_doi_key", "_title_key"):
        seen: dict[str, int] = {}
        for idx, key in enumerate(work[key_col].tolist()):
            if not key:
                continue
            if key in seen:
                uf.union(idx, seen[key])
            else:
                seen[key] = idx

    # One representative from each exact cluster is used for strict fuzzy
    # matching of title variants from different databases.
    exact_groups: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(work)):
        exact_groups[uf.find(idx)].append(idx)
    representatives = [
        max(indices, key=lambda i: int(work.at[i, "_abstract_len"]))
        for indices in exact_groups.values()
    ]

    by_year: dict[str, list[int]] = defaultdict(list)
    for idx in representatives:
        year = work.at[idx, "_year_key"]
        words = work.at[idx, "_title_words"]
        if year and len(words) >= 20:
            by_year[year].append(idx)

    # Strict fuzzy title matching. Sorting by title length lets us stop scanning
    # as soon as a pair can no longer satisfy the existing 0.65 length-ratio
    # rule. This preserves the original duplicate rule while avoiding many
    # unnecessary comparisons on large imports.
    year_groups = list(by_year.values())
    total_groups = max(len(year_groups), 1)
    comparisons = 0
    for group_no, indices in enumerate(year_groups, start=1):
        indices = sorted(indices, key=lambda i: len(work.at[i, "_title_words"]))
        for pos, left in enumerate(indices):
            left_title = work.at[left, "_title_words"]
            left_len = len(left_title)
            for right in indices[pos + 1:]:
                right_title = work.at[right, "_title_words"]
                right_len = len(right_title)
                # Because the list is length-sorted, all later titles are at
                # least this long; once the ratio drops below 0.65 we can break.
                if left_len / max(right_len, 1) < 0.65:
                    break
                comparisons += 1
                if token_set_ratio(left_title, right_title) >= FUZZY_TITLE_THRESHOLD:
                    uf.union(left, right)
        if progress_callback:
            progress_callback("유사 제목 확인", group_no, total_groups)

    if progress_callback:
        progress_callback("중복 그룹 정리", 1, 1)

    clusters: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(work)):
        clusters[uf.find(idx)].append(idx)

    keep_indices: list[int] = []
    removed_indices: list[int] = []
    for indices in clusters.values():
        winner = max(
            indices,
            key=lambda i: (
                int(work.at[i, "_abstract_len"]),
                bool(work.at[i, "_doi_key"]),
                len(str(work.at[i, "title"])),
                -i,
            ),
        )
        keep_indices.append(winner)
        removed_indices.extend(i for i in indices if i != winner)

    clean_cols = [c for c in work.columns if not c.startswith("_")]
    kept = work.loc[sorted(keep_indices), clean_cols].reset_index(drop=True)
    removed = work.loc[sorted(removed_indices), clean_cols].reset_index(drop=True)

    if "year" in kept.columns:
        kept["_year_sort"] = pd.to_numeric(kept["year"], errors="coerce")
        kept = kept.sort_values("_year_sort", na_position="last", kind="stable").drop(columns="_year_sort").reset_index(drop=True)
    if progress_callback:
        progress_callback("완료", 1, 1)
    return kept, removed


def screening_export(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "순번": range(1, len(df) + 1),
        "연도": df.get("year", ""),
        "제목": df.get("title", ""),
        "초록": df.get("abstract", ""),
    })
