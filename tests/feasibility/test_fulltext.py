import io
import json
import urllib.error

import pytest

from feasibility import fulltext as ft
from feasibility import llm_local as ll


# ---------------- detect_quant ----------------
def test_likely_pm_with_n():
    r = ft.detect_quant("TG (mg/dL) 120.5 ± 10.2, n = 8", "triglyceride", ["TG"])
    assert r["status"] == "likely"
    assert r["evidence"] == ["TG (mg/dL) 120.5 ± 10.2, n = 8"]
    assert r["pages"] == []


def test_figure_only():
    r = ft.detect_quant("Serum TG decreased in the treated group (Fig. 2).", "triglyceride", ["TG"])
    assert r["status"] == "figure_only"
    assert r["evidence"] and r["pages"] == []


def test_not_found():
    r = ft.detect_quant("Body weight was 25 ± 2 g.", "triglyceride", ["TG"])
    assert r == {"status": "not_found", "evidence": [], "pages": []}


def test_word_boundary_no_false_hit():
    r = ft.detect_quant("The ATGC panel gave 3.1 ± 0.2 (n = 5).", "triglyceride", ["TG"])
    assert r["status"] == "not_found"


def test_korean_synonym():
    r = ft.detect_quant("혈중 중성지방 수치는 120.5 ± 10.2 mg/dL 이었다.", "triglyceride", ["중성지방"])
    assert r["status"] == "likely"


def test_paren_sd():
    r = ft.detect_quant("LDL cholesterol, mean (SD): 130.2 (12.4)", "LDL", [])
    assert r["status"] == "likely"


def test_paren_without_sd_context_is_not_likely():
    r = ft.detect_quant("LDL was measured at week 4 (12)", "LDL", [])
    assert r["status"] == "not_found"


def test_mean_with_n():
    r = ft.detect_quant("Mean LDL was 130 mg/dL in the control group (n = 10).", "LDL", [])
    assert r["status"] == "likely"


def test_table_row_next_line():
    text = "Table 2\nLDL\n130.2 ± 12.4 | 110.1 ± 9.8"
    r = ft.detect_quant(text, "LDL", [])
    assert r["status"] == "likely"
    assert "130.2 ± 12.4 | 110.1 ± 9.8" in r["evidence"]


def test_caption_rule():
    text = ("Table 1. Values are expressed as mean ± SEM.\n"
            "Liver weight changed after treatment.\n")
    r = ft.detect_quant(text, "liver weight", [])
    assert r["status"] == "likely"
    assert any("mean ± SEM" in e for e in r["evidence"])


def test_caption_without_outcome_not_likely():
    r = ft.detect_quant("Table 1. Values are mean ± SD.\nNothing else.", "liver weight", [])
    assert r["status"] == "not_found"


def test_pages_and_evidence_verbatim_capped():
    lines = "\n".join(f"TG at day {i}: {100+i}.0 ± 5.0, n = 6" for i in range(8))
    text = "[PDF page 1]\nIntro text.\n[PDF page 3]\n" + lines
    r = ft.detect_quant(text, "TG", [])
    assert r["status"] == "likely"
    assert len(r["evidence"]) == 5
    assert all(e in text for e in r["evidence"])
    assert r["pages"] == [3]


@pytest.mark.parametrize("text,status", [
    ("TG was administered at 2.5 ± 0.3 mg/kg.", "not_found"),
    ("Mice received 5 +/- 1 g of diet; TG fell (Fig. 2).", "figure_only"),
    ("TG was dosed at 10 ± 2 mg/kg and TG decreased (Fig. 3).", "figure_only"),
    ("Mean age was 54 ± 8 years; TG was measured.", "not_found"),
    ("Median TG levels were assessed (n = 40 patients).", "not_found"),
])
def test_false_likely_negatives(text, status):
    assert ft.detect_quant(text, "triglyceride", ["TG"])["status"] == status


# ---------------- fetch_oa_fulltext ----------------
XML = b"""<article><body><sec><title>Results</title><p>TG fell to 100 &amp; below.</p>
<table-wrap><table><tbody><tr><td>TG</td><td>120.5 &#177; 10.2</td></tr>
<tr><td>LDL</td><td>130</td></tr></tbody></table></table-wrap></sec></body></article>"""


def test_xml_to_text_table():
    t = ft.xml_to_text(XML.decode())
    assert "TG fell to 100 & below." in t
    assert "[Table]" in t
    assert "TG | 120.5 ± 10.2" in t.splitlines()
    assert "<" not in t


def test_fetch_europepmc():
    calls = []

    def get(url):
        calls.append(url)
        return XML
    t = ft.fetch_oa_fulltext(pmcid="PMC123", _get=get)
    assert "europepmc/webservices/rest/PMC123/fullTextXML" in calls[0]
    assert "TG | 120.5" in t


def test_fetch_pmcid_normalized():
    calls = []
    ft.fetch_oa_fulltext(pmcid="123", _get=lambda u: calls.append(u) or XML)
    assert "/PMC123/" in calls[0]


def test_fetch_unpaywall_path(monkeypatch):
    seen = []

    def get(url):
        seen.append(url)
        if "unpaywall" in url:
            return json.dumps({"best_oa_location": {"url_for_pdf": "https://x.org/a.pdf"}}).encode()
        return b"%PDF-fake"
    monkeypatch.setattr(ft, "segment_pages", lambda b, *a, **k: (["[PDF page 1]\nTG 1 ± 2 n = 3"], 1, []))
    t = ft.fetch_oa_fulltext(doi="10.1/abc", email="a@b.org", _get=get)
    assert "TG 1 ± 2" in t
    assert "email=a%40b.org" in seen[0] or "email=a@b.org" in seen[0]
    assert seen[1] == "https://x.org/a.pdf"


def test_fetch_unpaywall_needs_email():
    assert ft.fetch_oa_fulltext(doi="10.1/abc", _get=lambda u: b"{}") is None
    t, reason = ft.fetch_oa_fulltext_ex(doi="10.1/abc", _get=lambda u: b"{}")
    assert t is None and "email" in reason


def test_fetch_failure_returns_none_with_reason():
    def boom(url):
        raise urllib.error.URLError("offline")
    assert ft.fetch_oa_fulltext(pmcid="PMC1", doi="10.1/x", email="a@b.org", _get=boom) is None
    t, reason = ft.fetch_oa_fulltext_ex(pmcid="PMC1", _get=boom)
    assert t is None and "offline" in reason


def test_fetch_falls_through_to_unpaywall(monkeypatch):
    def get(url):
        if "europepmc" in url:
            raise urllib.error.HTTPError(url, 404, "nf", {}, None)
        if "unpaywall" in url:
            return json.dumps({"best_oa_location": {"url_for_pdf": "https://x.org/a.pdf"}}).encode()
        return b"%PDF-1.4 fake"
    monkeypatch.setattr(ft, "segment_pages", lambda b, *a, **k: (["[PDF page 1]\nhello"], 1, []))
    assert "hello" in ft.fetch_oa_fulltext(pmcid="PMC1", doi="10.1/x", email="a@b.org", _get=get)


# ---------------- llm_local ----------------
class FakeResp:
    def __init__(self, payload):
        self._b = payload if isinstance(payload, bytes) else json.dumps(payload).encode()

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_ollama_unavailable_on_refused(monkeypatch):
    def boom(*a, **k):
        raise ConnectionRefusedError()
    monkeypatch.setattr(ll.urllib.request, "urlopen", boom)
    assert ll.ollama_available() is False
    assert ll.ollama_models() == []


def test_ollama_available_and_models(monkeypatch):
    monkeypatch.setattr(ll.urllib.request, "urlopen",
                        lambda req, timeout=None: FakeResp({"models": [{"name": "qwen2.5:3b"}]}))
    assert ll.ollama_available() is True
    assert ll.ollama_models() == ["qwen2.5:3b"]


def test_ollama_host_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "10.0.0.5:11434")
    assert ll.ollama_url() == "http://10.0.0.5:11434"
    monkeypatch.delenv("OLLAMA_HOST")
    assert ll.ollama_url() == "http://localhost:11434"


def _capture(monkeypatch, content):
    cap = {}

    def fake(req, timeout=None):
        cap["url"] = req.full_url
        cap["body"] = json.loads(req.data)
        return FakeResp({"message": {"content": content}})
    monkeypatch.setattr(ll.urllib.request, "urlopen", fake)
    return cap


def test_judge_parses_json(monkeypatch):
    cap = _capture(monkeypatch, '{"a": 1}')
    assert ll.ollama_judge("p", "qwen2.5:3b", system="s") == {"a": 1}
    b = cap["body"]
    assert cap["url"].endswith("/api/chat")
    assert b["format"] == "json" and b["stream"] is False and b["options"]["temperature"] == 0
    assert [m["role"] for m in b["messages"]] == ["system", "user"]


def test_judge_none_on_bad_json_and_error(monkeypatch):
    _capture(monkeypatch, "not json")
    assert ll.ollama_judge("p", "m") is None
    _capture(monkeypatch, "[1,2]")
    assert ll.ollama_judge("p", "m") is None

    def boom(*a, **k):
        raise OSError("x")
    monkeypatch.setattr(ll.urllib.request, "urlopen", boom)
    assert ll.ollama_judge("p", "m") is None


def test_defaults():
    assert ll.CPU_DEFAULT == "qwen2.5:3b" and ll.GPU_DEFAULT == "qwen2.5:7b"


def test_screen_verbatim_evidence(monkeypatch):
    _capture(monkeypatch, '{"decision": "X", "reason": "rat", "evidence": "in rats"}')
    r = ll.llm_screen_abstract("Effect in rats", "A study in rats.", "humans", "animal", "m")
    assert r["decision"] == "X" and r["evidence"] == "in rats"


def test_screen_non_verbatim_downgraded(monkeypatch):
    _capture(monkeypatch, '{"decision": "O", "reason": "r", "evidence": "randomized trial in humans"}')
    r = ll.llm_screen_abstract("T", "An open-label study.", "p", "e", "m")
    assert r["decision"] == "?"


def test_screen_bad_decision_and_none(monkeypatch):
    _capture(monkeypatch, '{"decision": "maybe", "reason": "r", "evidence": ""}')
    assert ll.llm_screen_abstract("T", "A", "p", "e", "m")["decision"] == "?"
    _capture(monkeypatch, "garbage")
    assert ll.llm_screen_abstract("T", "A", "p", "e", "m") is None


def test_assess_fulltext_validates(monkeypatch):
    text = "[PDF page 1]\nTG was 120.5 ± 10.2 mg/dL in controls (n = 8)."
    monkeypatch.setattr(ll, "segment_text", lambda t: [t])
    good = {"eligibility": "include", "outcomes": [{"name": "TG", "status": "confirmed",
            "evidence": "TG was 120.5 ± 10.2 mg/dL", "source": "p1"}]}
    monkeypatch.setattr(ll, "ollama_judge", lambda *a, **k: good)
    r = ll.llm_assess_fulltext(text, ["TG"], "pico", "m")
    assert r["outcomes"][0]["status"] == "confirmed"
    bad = {"eligibility": "include", "outcomes": [{"name": "TG", "status": "confirmed",
           "evidence": "fabricated sentence here", "source": "p1"}]}
    monkeypatch.setattr(ll, "ollama_judge", lambda *a, **k: bad)
    r = ll.llm_assess_fulltext(text, ["TG"], "pico", "m")
    assert r["outcomes"][0]["status"] == "uncertain"


def test_assess_fulltext_none_when_llm_down(monkeypatch):
    monkeypatch.setattr(ll, "ollama_judge", lambda *a, **k: None)
    assert ll.llm_assess_fulltext("some text", ["TG"], "p", "m") is None


def _two_chunks(monkeypatch, replies):
    monkeypatch.setattr(ll, "segment_text", lambda t: ["alpha TG was 120.5 ± 10.2 mg/dL in controls", "beta excluded because rats only"])
    it = iter(replies)
    monkeypatch.setattr(ll, "ollama_judge", lambda *a, **k: next(it))


INC = {"eligibility": "include", "outcomes": [{"name": "TG", "status": "confirmed",
       "evidence": "TG was 120.5 ± 10.2 mg/dL", "source": "p1"}]}
EXC = {"eligibility": "exclude", "reason": "animal", "evidence": "excluded because rats only", "outcomes": []}


def test_assess_disagreement_uncertain(monkeypatch):
    _two_chunks(monkeypatch, [INC, EXC])
    r = ll.llm_assess_fulltext("x", ["TG"], "p", "m")
    assert r["eligibility"] == "uncertain" and r["failed_chunks"] == 0
    assert all(o["status"] != "confirmed" for o in r["outcomes"])


def test_assess_exclude_copies_reason(monkeypatch):
    _two_chunks(monkeypatch, [EXC, EXC])
    r = ll.llm_assess_fulltext("x", ["TG"], "p", "m")
    assert r["eligibility"] == "exclude" and r["reason"] == "animal"
    assert r["evidence"] == "excluded because rats only"


def test_assess_failed_chunk_blocks_exclude(monkeypatch):
    _two_chunks(monkeypatch, [None, EXC])
    r = ll.llm_assess_fulltext("x", ["TG"], "p", "m")
    assert r["eligibility"] == "uncertain" and r["failed_chunks"] == 1


def test_ollama_request_has_num_ctx(monkeypatch):
    seen = {}

    def fake(req, timeout=None):
        seen["body"] = json.loads(req.data)
        seen["timeout"] = timeout
        return FakeResp({"message": {"content": '{"decision": "?"}'}})
    monkeypatch.setattr(ll.urllib.request, "urlopen", fake)
    monkeypatch.delenv("OLLAMA_NUM_CTX", raising=False)
    ll.ollama_judge("p", "m")
    assert seen["body"]["options"]["num_ctx"] == 8192 and seen["body"]["options"]["temperature"] == 0
    monkeypatch.setenv("OLLAMA_NUM_CTX", "16384")
    ll.ollama_judge("p", "m")
    assert seen["body"]["options"]["num_ctx"] == 16384
    monkeypatch.setenv("OLLAMA_NUM_CTX", "junk")
    ll.ollama_judge("p", "m")
    assert seen["body"]["options"]["num_ctx"] == 8192
    ll.llm_screen_abstract("t", "a", "pico", "", "m")
    assert seen["timeout"] == ll.SCREEN_TIMEOUT == 120


def test_segment_text_default_and_long_line():
    assert ll.CHUNK_CHARS == 6000
    text = "x" * 15000 + "\nshort line\n" + "y" * 100
    chunks = ll.segment_text(text)
    assert all(len(c) <= 6000 for c in chunks)
    assert "".join(c.replace("\n", "") for c in chunks) == text.replace("\n", "")
    assert ll.segment_text("") == [""]
    assert ll.segment_text("a\nb", 10) == ["a\nb"]


def _unpaywall_get(pdf_bytes):
    def get(url):
        if "unpaywall" in url:
            return json.dumps({"best_oa_location": {"url_for_pdf": "https://x.org/a.pdf"}}).encode()
        return pdf_bytes
    return get


def test_unpaywall_rejects_non_pdf_and_oversize(monkeypatch):
    parsed = []
    monkeypatch.setattr(ft, "segment_pages", lambda b, *a, **k: parsed.append(b) or (["text"], 1, []))
    t, reason = ft.fetch_oa_fulltext_ex(doi="10.1/a", email="a@b.org", _get=_unpaywall_get(b"<html>paywall</html>"))
    assert t is None and "PDF header" in reason and not parsed
    monkeypatch.setattr(ft, "PDF_MAX_BYTES", 10)
    t, reason = ft.fetch_oa_fulltext_ex(doi="10.1/a", email="a@b.org", _get=_unpaywall_get(b"%PDF-" + b"x" * 20))
    assert t is None and "size" in reason and not parsed


def test_default_get_caps_bytes(monkeypatch):
    class R:
        def __init__(self):
            self.asked = None

        def read(self, n=-1):
            self.asked = n
            return b"x" * (n if n and n > 0 else 50)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    r = R()
    monkeypatch.setattr(ft.urllib.request, "urlopen", lambda req, timeout=None: r)
    with pytest.raises(ValueError):
        ft._default_get("https://x.org/a.pdf", max_bytes=10)
    assert r.asked == 11
