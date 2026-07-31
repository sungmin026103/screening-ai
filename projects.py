from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from utils import load_json, save_json

ROOT = Path(__file__).resolve().parents[0]
PROJECTS_DIR = ROOT / "data" / "projects"


def slugify(name: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z가-힣_-]+", "-", name.strip()).strip("-")
    return slug or "project"


def project_exists(slug: str) -> bool:
    return (PROJECTS_DIR / slug).is_dir()


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
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("프로젝트 이름을 입력해주세요.")

    slug = slugify(clean_name)
    path = PROJECTS_DIR / slug
    if path.exists():
        raise ValueError("같은 이름의 프로젝트가 이미 있습니다. 다른 이름을 사용해주세요.")

    path.mkdir(parents=True, exist_ok=False)
    now = datetime.now().isoformat(timespec="seconds")
    meta = {"name": clean_name, "slug": slug, "created_at": now, "updated_at": now}
    save_json(path / "project.json", meta)
    return meta


def rename_project(slug: str, new_name: str) -> dict:
    clean_name = new_name.strip()
    if not clean_name:
        raise ValueError("새 프로젝트 이름을 입력해주세요.")

    old_path = PROJECTS_DIR / slug
    if not old_path.exists():
        raise FileNotFoundError("프로젝트를 찾을 수 없습니다.")

    new_slug = slugify(clean_name)
    new_path = PROJECTS_DIR / new_slug
    if new_slug != slug and new_path.exists():
        raise ValueError("같은 이름의 프로젝트가 이미 있습니다. 다른 이름을 사용해주세요.")

    meta = load_json(old_path / "project.json", {"name": slug, "slug": slug})
    if new_slug != slug:
        old_path.rename(new_path)
    else:
        new_path = old_path

    meta["name"] = clean_name
    meta["slug"] = new_slug
    meta["updated_at"] = datetime.now().isoformat(timespec="seconds")
    save_json(new_path / "project.json", meta)
    return meta


def delete_project(slug: str) -> None:
    path = PROJECTS_DIR / slug
    if not path.exists():
        raise FileNotFoundError("프로젝트를 찾을 수 없습니다.")
    shutil.rmtree(path)


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
    touch_project(slug)


def load_pico(slug: str) -> dict:
    return load_json(project_path(slug) / "pico.json", {
        "population": "", "intervention": "", "comparator": "", "outcome": "", "exclusion_criteria": "",
    })


def touch_project(slug: str, **fields) -> None:
    """Update project metadata and modified time."""
    path = project_path(slug)
    meta = load_json(path / "project.json", {"name": slug, "slug": slug})
    meta.update(fields)
    meta["updated_at"] = datetime.now().isoformat(timespec="seconds")
    save_json(path / "project.json", meta)


def save_project_state(slug: str, key: str, value) -> None:
    """Persist a Python object that belongs to one project."""
    import pickle
    path = project_path(slug) / "state"
    path.mkdir(parents=True, exist_ok=True)
    with open(path / f"{key}.pkl", "wb") as fh:
        pickle.dump(value, fh, protocol=pickle.HIGHEST_PROTOCOL)
    touch_project(slug)


def load_project_state(slug: str, key: str, default=None):
    import pickle
    file = project_path(slug) / "state" / f"{key}.pkl"
    if not file.exists():
        return default
    try:
        with open(file, "rb") as fh:
            return pickle.load(fh)
    except Exception:
        return default


def project_progress(slug: str) -> dict:
    """Return simple, evidence-based workflow completion flags."""
    path = project_path(slug)
    records = load_records(slug)
    pico = load_pico(slug)
    screening = load_project_state(slug, "screening_result")
    meta_done = bool(load_project_state(slug, "meta_done", False))
    flags = {
        "import": not records.empty,
        "pico": any(str(v).strip() for v in pico.values()),
        "screen": screening is not None,
        "meta": meta_done,
    }
    completed = sum(flags.values())
    return {"flags": flags, "percent": int(round(completed / len(flags) * 100))}
