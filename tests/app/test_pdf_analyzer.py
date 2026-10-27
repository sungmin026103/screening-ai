"""PDF 분석: 두 단 편집 + 서론의 인용 문장(Wistar rats, female, 4 weeks)에 속지 않고 Methods 값을 읽는지, 스캔 PDF는 OCR로."""
import io
import shutil
from pathlib import Path

import pytest

from pdf_analyzer import analyze_pdf_bytes

PDF = (Path(__file__).parent / "fixtures" / "two_col.pdf").read_bytes()
TRUTH = {"study": "Park (2021)", "species": "C57BL/6J mice", "sex": "Male", "age": "8-week-old",
         "dose": "50 mg/kg/day", "duration": "14 days", "route": "Oral gavage", "sample_size": "n=8"}


def _check(data: bytes):
    f = analyze_pdf_bytes(data, "x.pdf")["fields"]
    assert {k: f[k]["value"] for k in TRUTH} == TRUTH


def test_two_column_methods_first():
    _check(PDF)


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract 없음")
def test_scanned_pdf_ocr():
    pdfium = pytest.importorskip("pypdfium2")
    imgs = [p.render(scale=200 / 72).to_pil().convert("L") for p in pdfium.PdfDocument(PDF)]
    buf = io.BytesIO()
    imgs[0].save(buf, format="PDF", save_all=True, append_images=imgs[1:], resolution=200)
    _check(buf.getvalue())
