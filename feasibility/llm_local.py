"""Optional local LLM (Ollama) helpers. Every failure degrades to None -> manual mode."""
from __future__ import annotations

import json
import os
import re
import urllib.request

from ma_core import normalize, validate_assessment

CPU_DEFAULT = "qwen2.5:3b"
GPU_DEFAULT = "qwen2.5:7b"
OLLAMA_URL = "http://localhost:11434"
NUM_CTX = 8192          # Ollama defaults to a 2k-4k context, which silently truncates full-text chunks
CHUNK_CHARS = 6000      # ~1.5-2k tokens of text + prompt/schema fits comfortably in NUM_CTX
SCREEN_TIMEOUT = 120    # seconds per abstract-screening call


def num_ctx() -> int:
    """OLLAMA_NUM_CTX overrides the default context window."""
    try:
        v = int(os.environ.get("OLLAMA_NUM_CTX", "").strip())
        return v if v > 0 else NUM_CTX
    except ValueError:
        return NUM_CTX


def ollama_url() -> str:
    h = os.environ.get("OLLAMA_HOST", "").strip()
    if not h:
        return OLLAMA_URL
    if not re.match(r"^https?://", h):
        h = "http://" + h
    return h.rstrip("/")


def _tags(timeout):
    with urllib.request.urlopen(ollama_url() + "/api/tags", timeout=timeout) as r:
        return json.loads(r.read())


def ollama_available(timeout: float = 1.5) -> bool:
    try:
        return isinstance(_tags(timeout), dict)
    except Exception:
        return False


def ollama_models(timeout: float = 1.5) -> list[str]:
    try:
        return [m["name"] for m in _tags(timeout).get("models", []) if isinstance(m, dict) and m.get("name")]
    except Exception:
        return []


def ollama_judge(prompt: str, model: str, system: str | None = None, timeout: int = 300):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    body = json.dumps({"model": model, "messages": msgs, "stream": False, "format": "json",
                       "options": {"temperature": 0, "num_ctx": num_ctx()}}).encode()
    try:
        req = urllib.request.Request(ollama_url() + "/api/chat", data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            content = json.loads(r.read())["message"]["content"]
        out = json.loads(content)
        return out if isinstance(out, dict) else None
    except Exception:
        return None


def llm_screen_abstract(title, abstract, pico, exclusion, model, timeout: int = SCREEN_TIMEOUT):
    """Suggest O/X/? for a human. Evidence must be verbatim in title+abstract, else '?'."""
    source = f"{title or ''}\n{abstract or ''}"
    prompt = (
        "Screen this record for a systematic review. Reply as JSON: "
        '{"decision": "O"|"X"|"?", "reason": str, "evidence": str}. '
        "'evidence' must be an exact quote from the record. Use '?' if unsure.\n"
        f"PICO: {pico}\nExclusion criteria: {exclusion}\n\nTitle: {title}\nAbstract: {abstract}"
    )
    raw = ollama_judge(prompt, model, system="You are a careful systematic-review screener.", timeout=timeout)
    if raw is None:
        return None
    decision = str(raw.get("decision", "?")).strip()
    reason = str(raw.get("reason") or "")
    evidence = str(raw.get("evidence") or "").strip()
    if decision not in ("O", "X", "?"):
        decision = "?"
    if decision != "?" and (len(evidence) < 4 or normalize(evidence) not in normalize(source)):
        decision, reason = "?", "Unverified evidence: " + reason
    return {"decision": decision, "reason": reason, "evidence": evidence}


def segment_text(text: str, chunk_chars: int = CHUNK_CHARS) -> list[str]:
    """Split full text into chunks of <= chunk_chars on line boundaries; a single line longer than
    chunk_chars (PDF extraction often yields one) is cut into chunk-sized pieces."""
    chunks, cur, size = [], [], 0
    lines = []
    for line in text.splitlines():
        lines.extend([line[i:i + chunk_chars] for i in range(0, len(line), chunk_chars)] or [""])
    for line in lines:
        if cur and size + len(line) > chunk_chars:
            chunks.append("\n".join(cur))
            cur, size = [], 0
        cur.append(line)
        size += len(line) + 1
    if cur:
        chunks.append("\n".join(cur))
    return chunks or [""]


_RANK = {"confirmed": 2, "uncertain": 1, "unavailable": 0}


def _prompt(chunk, outcomes, pico):
    schema = (
        '{"eligibility": "include"|"exclude"|"uncertain", "reason": str, "evidence": str, '
        '"outcomes": [{"name": str, "status": "confirmed"|"uncertain"|"unavailable", '
        '"source": str, "evidence": str, "n": any, "mean": any, "dispersion": any, '
        '"dispersion_type": str}]}'
    )
    return (
        f"Assess this full-text excerpt. Reply as JSON: {schema}. Quote evidence exactly. "
        '"confirmed" requires mean, dispersion and n per group in the excerpt.\n'
        f"PICO: {pico}\nOutcomes: {json.dumps(outcomes, ensure_ascii=False)}\n\n{chunk}"
    )


def llm_assess_fulltext(text, outcomes, pico, model):
    """Judge each chunk, validate quotes per chunk, merge conservatively.

    Disagreement or failed chunks never yield 'exclude'; non-include results carry no 'confirmed'.
    """
    good, failed = [], 0
    for chunk in segment_text(text):
        raw = ollama_judge(_prompt(chunk, outcomes, pico), model,
                           system="You are a careful meta-analysis data screener.")
        if raw is None:
            failed += 1
            continue
        try:
            good.append(validate_assessment(raw, list(outcomes), chunk, True))
        except ValueError:
            failed += 1
    if not good:
        return None
    merged = dict(good[0])
    merged["outcomes"] = list(good[0]["outcomes"])
    for v in good[1:]:
        for i, o in enumerate(v["outcomes"]):
            if _RANK.get(o["status"], 0) > _RANK.get(merged["outcomes"][i]["status"], 0):
                merged["outcomes"][i] = o
    eligs = {v["eligibility"] for v in good}
    if "exclude" in eligs and "include" in eligs:
        final, reason, evidence = "uncertain", "Chunks disagree on eligibility", ""
    elif "exclude" in eligs:
        src = next(v for v in good if v["eligibility"] == "exclude")
        if failed:
            final, reason, evidence = "uncertain", f"{failed} chunk(s) failed; exclusion not final: {src['reason']}", src["evidence"]
        else:
            final, reason, evidence = "exclude", src["reason"], src["evidence"]
    elif "include" in eligs:
        src = next(v for v in good if v["eligibility"] == "include")
        final, reason, evidence = "include", src["reason"], src["evidence"]
    else:
        final, reason, evidence = "uncertain", merged["reason"], merged["evidence"]
    merged.update(eligibility=final, reason=reason, evidence=evidence, failed_chunks=failed)
    if final != "include":
        merged["outcomes"] = [dict(o, status="uncertain") if o["status"] == "confirmed" else o
                              for o in merged["outcomes"]]
    return merged
