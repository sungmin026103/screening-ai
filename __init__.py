from __future__ import annotations

import pandas as pd

from .utils import normalize_doi, normalize_title


def deduplicate_records(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df.copy(), df.copy()
    work = df.copy().reset_index(drop=True)
    work["_doi_key"] = work["doi"].map(normalize_doi)
    work["_title_key"] = work["title"].map(normalize_title)
    work["_abstract_len"] = work["abstract"].fillna("").astype(str).str.len()

    keep_set: set[int] = set()
    removed_indices: list[int] = []
    seen_doi: dict[str, int] = {}
    seen_title: dict[str, int] = {}

    for idx, row in work.iterrows():
        doi_key, title_key = row["_doi_key"], row["_title_key"]
        matched = seen_doi.get(doi_key) if doi_key else None
        if matched is None and title_key:
            matched = seen_title.get(title_key)
        if matched is None:
            keep_set.add(idx)
            if doi_key:
                seen_doi[doi_key] = idx
            if title_key:
                seen_title[title_key] = idx
            continue

        matched_row = work.loc[matched]
        if row["_abstract_len"] > matched_row["_abstract_len"]:
            keep_set.discard(matched)
            keep_set.add(idx)
            removed_indices.append(matched)
            winner = idx
        else:
            removed_indices.append(idx)
            winner = matched
        # Repoint every key involved in this match (idx's own keys AND the
        # matched row's own keys) to whichever row actually survives. If we
        # only updated idx's keys, a later record that matches `matched` via
        # its OTHER key (the one idx didn't share) would still be pointed at
        # an already-removed row -- and a second replacement attempt on that
        # stale row would call keep_set.discard on an index no longer kept,
        # or worse, look confusing further down. Repointing both sides keeps
        # every key consistently referencing a currently-kept row.
        if doi_key:
            seen_doi[doi_key] = winner
        if title_key:
            seen_title[title_key] = winner
        if matched_row["_doi_key"]:
            seen_doi[matched_row["_doi_key"]] = winner
        if matched_row["_title_key"]:
            seen_title[matched_row["_title_key"]] = winner

    clean_cols = [c for c in work.columns if not c.startswith("_")]
    kept = work.loc[sorted(keep_set), clean_cols].reset_index(drop=True)
    removed = work.loc[sorted(set(removed_indices)), clean_cols].reset_index(drop=True)
    return kept, removed


def screening_export(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "순번": range(1, len(df) + 1),
        "연도": df.get("year", ""),
        "제목": df.get("title", ""),
        "초록": df.get("abstract", ""),
    })
    return out
