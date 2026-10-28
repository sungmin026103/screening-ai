"""저장소 규칙을 기계적으로 강제한다(CLAUDE.md '하드 규칙'). 규칙을 바꾸려면 CLAUDE.md와 이 파일을 함께 고친다."""
import ast
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 규칙 1 — 기존 파일명 유지(사용자가 GitHub에 파일 단위로 붙여넣어 관리). 새 파일 추가는 자유, 이름 변경·삭제는 금지.
PROTECTED = """
AI_screening_figures.py advanced.py app.py auto_figures.py dedup.py extraction.py feasibility_page.py figure_auto.py
figure_digitizer.py figure_evidence.py forest_styles.py importers.py jobs.py keepalive.py ma_core.py ma_feasibility.py
ma_feasibility_pro.py ma_methods.py meta_page.py meta_sections.py metaanalysis.py pdf_analyzer.py projects.py rmeta.py
screening.py styles.py supplementary.py upload_safety.py utils.py requirements.txt packages.txt .streamlit/config.toml
feasibility/__init__.py feasibility/estimator.py feasibility/fulltext.py feasibility/llm_local.py feasibility/population.py
feasibility/priors.py feasibility/replay.py feasibility/sampling.py feasibility/search_apis.py feasibility/session.py
feasibility/priors.json feasibility/design_default.json feasibility/library/__init__.py feasibility/library/enrich.py
feasibility/library/etl.py feasibility/library/manifest.py fonts/Carlito-Regular.ttf fonts/Carlito-Bold.ttf
""".split()

PAID_SDKS = {"openai", "anthropic", "cohere", "mistralai", "google.generativeai", "google.genai", "groq", "together"}
AGPL_LIBS = {"fitz", "pymupdf", "ghostscript"}
SKIP_DIRS = {".git", ".venv", "venv", "data", "__pycache__", "node_modules"}


def _py_files():
    return [p for p in ROOT.rglob("*.py") if not SKIP_DIRS & set(p.relative_to(ROOT).parts)]


def _imports(p: Path) -> set[str]:
    tree = ast.parse(p.read_text(encoding="utf-8"))
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            out.add(n.module)
    return out


@pytest.mark.parametrize("path", PROTECTED)
def test_protected_file_names_kept(path):
    assert (ROOT / path).exists(), f"{path}: 기존 파일명은 바꾸거나 지우지 않는다(CLAUDE.md 규칙 1)"


def test_free_only_no_paid_llm_sdk():
    bad = {str(p.relative_to(ROOT)): sorted(m for m in _imports(p) if any(m == s or m.startswith(s + ".") for s in PAID_SDKS))
           for p in _py_files()}
    bad = {k: v for k, v in bad.items() if v}
    assert not bad, f"유료 API SDK 금지(CLAUDE.md 규칙 3): {bad}"


def test_no_agpl_dependency():
    bad = {str(p.relative_to(ROOT)): sorted(_imports(p) & AGPL_LIBS) for p in _py_files()}
    bad = {k: v for k, v in bad.items() if v}
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    assert not bad and not any(re.search(rf"^{lib}\b", req, re.M) for lib in AGPL_LIBS), \
        f"AGPL 라이브러리 금지(상용화 목표, CLAUDE.md 규칙 4): {bad}"


def test_requirements_pinned():
    lines = [ln.split("#")[0].strip() for ln in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()]
    loose = [ln for ln in lines if ln and "==" not in ln]
    assert not loose, f"requirements.txt는 정확한 버전(==)으로 고정한다(재현성): {loose}"


def test_forest_default_is_v1():
    import forest_styles as F
    import meta_sections as M
    assert F.ForestOptions().style == 1 and M.default_settings("X")["style"] == 1, "forest plot 기본 디자인은 V1(CLAUDE.md 규칙 5)"


def test_codemap_lists_every_module():
    codemap = (ROOT / "CODEMAP.md").read_text(encoding="utf-8")
    missing = [str(p.relative_to(ROOT)) for p in _py_files()
               if p.name != "__init__.py" and not p.relative_to(ROOT).parts[0] == "tests" and p.name not in codemap]
    assert not missing, f"새 모듈은 CODEMAP.md에 한 줄 추가한다(AI가 지도만 보고 찾아가도록): {missing}"


def test_claude_md_points_to_verify():
    txt = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    assert "python tools/verify.py" in txt and (ROOT / "tools" / "verify.py").exists()
    assert (ROOT / "AGENTS.md").exists()
