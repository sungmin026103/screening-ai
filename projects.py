from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from utils import load_json, save_json

ROOT = Path(__file__).resolve().parents[0]
PROJECTS_DIR = ROOT / "data" / "projects"


def slugify(name: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z가-힣_-]+", "-", name.strip()).strip("-")
    return slug or "project"


def list_projects() -> list[dict]:
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for path in PROJECTS_DIR.iterdir():
        if path.is_dir():
            meta = load_json(path / "project.json", {"name": path.name})
            meta["slug"] = path.name
            items.append(meta)
    return sorted(items, key=lambda x: x.get("updated_at", ""), reverse=True)


def create_project(name: str) -> dict:
    slug = slugify(name)
    path = PROJECTS_DIR / slug
    path.mkdir(parents=True, exist_ok=True)
    now = datetime.now().isoformat(timespec="seconds")
    meta = {"name": name.strip() or slug, "slug": slug, "created_at": now, "updated_at": now}
    save_json(path / "project.json", meta)
    return meta


def project_path(slug: str) -> Path:
    path = PROJECTS_DIR / slug
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_records(slug: str, df: pd.DataFrame) -> None:
    path = project_path(slug)
    df.to_pickle(path / "records.pkl")
    meta = load_json(path / "project.json", {"name": slug, "slug": slug})
    meta["updated_at"] = datetime.now().isoformat(timespec="seconds")
    meta["records"] = int(len(df))
    save_json(path / "project.json", meta)


def load_records(slug: str) -> pd.DataFrame:
    file = project_path(slug) / "records.pkl"
    return pd.read_pickle(file) if file.exists() else pd.DataFrame()


def save_pico(slug: str, pico: dict) -> None:
    save_json(project_path(slug) / "pico.json", pico)


def load_pico(slug: str) -> dict:
    return load_json(project_path(slug) / "pico.json", {
        "population": "", "intervention": "", "comparator": "", "outcome": "", "exclusion_criteria": "",
    })
