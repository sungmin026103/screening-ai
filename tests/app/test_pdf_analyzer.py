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


# 표현이 다른 시험 논문: 반복 머리글, 위첨자/문자 소속, 체중(g)·적응 기간·서론 인용 함정, 여러 용량
CASES = {'lycopene': {'study': 'Zhang (2019)',
              'species': 'SD rats',
              'sex': 'Male',
              'age': '6–8 weeks old',
              'model': 'High-fat diet',
              'intervention': 'lycopene',
              'dose': '10 mg/kg; 40 mg/kg',
              'duration': '8 weeks',
              'route': 'Oral',
              'control_groups': 'ND',
              'treat_groups': 'HFD; HFD+LL; HFD+HL',
              'sample_size': 'n=12',
              'dispersion': 'SD'},
 'quercetin': {'study': 'Rossi (2023)',
               'species': 'C57BL/6N mice',
               'sex': 'Female',
               'age': '10 weeks of age',
               'intervention': 'quercetin',
               'dose': '25 mg/kg',
               'duration': '6 weeks',
               'route': 'Intraperitoneal',
               'control_groups': 'VEH',
               'sample_size': 'n=10',
               'dispersion': 'SEM'},
 'resveratrol': {'study': 'Garcia-Lopez (2021)',
                 'species': 'Wistar rats',
                 'sex': 'Female',
                 'age': '12 weeks',
                 'model': 'Ovariectomy',
                 'intervention': 'Resveratrol',
                 'dose': '20 mg/kg',
                 'duration': '12 weeks',
                 'route': 'Oral gavage',
                 'control_groups': 'SHAM',
                 'treat_groups': 'OVX+RES',
                 'sample_size': 'n=8',
                 'dispersion': 'SD'}}


@pytest.mark.parametrize("name", list(CASES))
def test_varied_papers(name):
    f = analyze_pdf_bytes((Path(__file__).parent / "fixtures" / f"{name}.pdf").read_bytes(), name)["fields"]
    for k, t in CASES[name].items():
        v = f[k]["value"].lower()
        assert (v == t.lower()) if k == "sex" else all(p.strip().lower() in v for p in t.split(";")), (k, v, t)
