# MA Feasibility Predictor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Predict, before a full SR, the per-outcome number of meta-analysable independent studies (k) as a calibrated interval, validated by replaying the user's completed SRs.

**Architecture:** Pure-python `feasibility/` package (no Streamlit imports) → stage-wise Bayesian estimator fed by a stratified human-labelled sample, priors fitted from a local SR library; `feasibility_page.py` is the only UI file. Replay harness is the acceptance gate.

**Tech Stack:** Python 3.11+, numpy/scipy/pandas/scikit-learn (already pinned), rapidfuzz, pypdf, stdlib urllib for APIs, optional Ollama over HTTP. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-15-ma-feasibility-predictor-design.md`

## Global Constraints

- Cost 0: no paid API, no new pip dependency.
- `feasibility/*` must not import streamlit.
- Library raw data (user research data) never inside the repo/zip: lives at `FEASIBILITY_LIBRARY_DIR` (default `~/Documents/SR_feasibility_library`); repo ships only `feasibility/priors.json`.
- Acceptance (leave-one-SR-out replay): 80% PI coverage 72–90%, 50% PI coverage 40–62%, go/no-go accuracy ≥ 85% at threshold k=5.
- Label map: `o/O/ㅇ`→1, `x/X`→0, `△ & $ 애매`→0.5 (passes forward), `위와 동일`→previous row, `#REF!`/blank→missing.
- Monte Carlo draws S = 10,000 (tests may use 2,000).
- Existing tests (`test_ma_core.py`, `test_upload_safety.py`, `test_figure_evidence.py`) keep passing.

## Review Focus

1. Stratum where sample found 0 includes (very common at 0.03–1% yield) → upper bound must stay > 0, not collapse to 0.
2. Sampled passes whose full text is missing → treated as missing (wider interval), never as excluded.
3. Count-only DB (Scopus/Embase hits typed in) with no records → N_U widened via overlap prior, never silently added at 100%.
4. API network failure / >10,000 PubMed hits → explicit partial-result flag and year-split, never silent truncation.
5. Corpus smaller than requested sample → census, zero uncertainty from that stratum, no crash.

---

### Task 1: File map + changelog tidy

**Files:** Create `CODEMAP.md`; move `README_V30..V35.txt`, `RELEASE_NOTES_V30.txt` → `docs/changelog/`.

- [ ] Step 1: `grep -rn "README_V3\|RELEASE_NOTES" *.py` → expect no hits (else don't move that file).
- [ ] Step 2: `git mv` the files; write CODEMAP.md table: 분석 | 진입 파일 | 핵심 함수 | 테스트.
- [ ] Step 3: `python -m pytest -q test_ma_core.py test_upload_safety.py test_figure_evidence.py` → pass. Commit.

### Task 2: Estimator core

**Files:** Create `feasibility/__init__.py`, `feasibility/estimator.py`; Test `tests/feasibility/test_estimator.py`.

**Interfaces — Produces:**
- `@dataclass StratumObs(name:str, N:int, n:int, x:float)` (x may be fractional for 0.5 labels)
- `@dataclass StagePrior(a:float, b:float)`
- `@dataclass FulltextObs(m_read:int, y_include:int, m_missing:int, quant:dict[str,int])` — quant[o] = included with quantitative data for outcome o
- `estimate_k(strata:list[StratumObs], ft:FulltextObs, outcomes:list[str], priors:dict, threshold:int=5, n_draws:int=10000, seed:int=0, extra_N:tuple[float,float]|None=None) -> pd.DataFrame` columns: outcome, median, lo50, hi50, lo80, hi80, lo95, hi95, p_ge_threshold, decision, sens_lo80_missing_excl, sens_hi80_missing_incl
- priors keys: `"stratum"` → callable(name, rel_rank)->StagePrior or StagePrior, `"q2"`, `"q3"` (dict outcome→StagePrior with `"_default"`), `"r_study"` (StagePrior on reports→study thinning), `"decision"`: (go=0.8, nogo=0.2)

- [ ] Step 1: tests
  - `test_census_stratum_no_uncertainty`: N=n=50,x=10, q2=q3=Beta(1e6,1) → all quantiles == 10.
  - `test_zero_includes_upper_positive`: N=10000,n=100,x=0 → hi95 > 0.
  - `test_coverage_synthetic`: 300 synthetic projects (true p per stratum from Beta, draw sample), 80% PI coverage within 0.80±0.05.
  - `test_missing_fulltext_widens`: same data, m_missing=5 vs 0 → (hi80−lo80) larger.
  - `test_decision_rule`: p_ge ≥0.8 → "진행", ≤0.2 → "비진행", else "보류".
- [ ] Step 2: run → fail. Step 3: implement — per draw: p_h~Beta(x+a, n−x+b), T1=Σ(x_h + BetaBinomial-equivalent Binomial(N_h−n_h, p_h)); q2~Beta(y+a, m_read−y+b); T2~Bin(T1,q2); per outcome q3~Beta(quant+a, y−quant+b); K~Bin(T2,q3); thin by r~Beta. Missing full text: MAR in main; sensitivity bounds recompute with missing→all excluded / all included.
- [ ] Step 4: pass. Commit.

### Task 3: Relevance scoring + sampling design

**Files:** Create `feasibility/sampling.py`; Test `tests/feasibility/test_sampling.py`.

**Interfaces — Consumes:** `screening._parse_pico_sections`, `screening._split_bullet_items` (reuse scoring of `pico_rank_strata`). **Produces:**
- `relevance_scores(df:pd.DataFrame, pico:str, exclusion:str="") -> np.ndarray` (same formula as `pico_rank_strata`)
- `make_strata(scores:np.ndarray, bounds:tuple[float,...]=(0.01,0.05,0.20,1.0)) -> np.ndarray[str]` (quantile cut, top stratum labelled "S1")
- `allocate(strata:np.ndarray, n_total:int, census_top:int, weights:dict[str,float]|None=None) -> dict[str,int]`
- `draw_sample(strata, alloc, seed) -> np.ndarray[int]` (row indices)
- `neyman_followup(obs:list[StratumObs], n_add:int) -> dict[str,int]`

- [ ] Steps: tests `test_scores_deterministic` (row order shuffle → same score per record), `test_allocation_sums`, `test_small_corpus_census` (N=40, n=150 → all 40), `test_neyman_prefers_uncertain`. Implement, pass, commit.

### Task 4: Library ETL

**Files:** Create `feasibility/library/manifest.py`, `feasibility/library/etl.py`, `feasibility/library/enrich.py`; Test `tests/feasibility/test_etl.py` with a tiny fixture workbook built in-test.

**Interfaces — Produces:**
- `normalize_label(v, prev) -> float|None` (label map from Global Constraints)
- `load_sr(sr_id:str, xlsx_path:Path) -> SRRecord` where `SRRecord(records:pd.DataFrame[record_id,title,year,abstract,z1,z2,ft_missing], outcomes:dict[str,int], pico:str, meta:dict)`
- `enrich_abstracts(records, sources:list[pd.DataFrame]) -> (records, match_rate)` — exact normalised-title, then rapidfuzz token_set_ratio ≥ 95 with year ±1.
- `build_library(out_dir) -> pd.DataFrame` summary per SR (N, n_z1, n_z2, k per outcome, abstract match rate).
- Manifest per SR id L1–L8: workbook glob, title-stage sheet + label columns + final column, abstract-stage sheet, full-text sheet + decision column + reason column, extraction sheets → outcome names, PICO sheet. z1 = record appears in full-text sheet (passed abstract stage); z2 = full-text final include; ft_missing = reason matches `전문확보|전문X|전문 없|full text`.

- [ ] Steps: fixture test for label map (all symbols), title linking (one fuzzy pair), ft_missing detection. Implement; run `python -m feasibility.library.etl --build` on real files; print summary; record match rates in CODEMAP notes. Commit (code only; library data outside repo).

### Task 5: Priors from library

**Files:** Create `feasibility/priors.py`; Test `tests/feasibility/test_priors.py`.

**Interfaces — Produces:** `fit_priors(lib:list[SRRecord], exclude:str|None=None, scores_by_sr:dict|None=None) -> dict` (same keys as Task 2), `save_priors(p, path)`, `load_priors(path=None) -> dict`.
- Stratum prior: per relevance-rank bucket, method-of-moments Beta over SRs' prevalence, concentration capped (a+b ≤ 20) so a project's own sample dominates.
- q2, q3: method-of-moments Beta across SRs (q3 by outcome family; `_default` pooled).
- [ ] Steps: test that `exclude="L1"` changes nothing derived from L1 (leakage test), test a+b cap. Implement, commit.

### Task 6: Replay validation (acceptance gate)

**Files:** Create `feasibility/replay.py`; Output `docs/validation/replay_report.md` + `.csv`.

**Interfaces — Consumes:** Tasks 2,3,4,5. **Produces:** `replay_sr(sr:SRRecord, priors, design:dict, n_rep:int, seed:int) -> pd.DataFrame`, `run_validation(lib, designs:list[dict], n_rep:int=200) -> pd.DataFrame`, CLI `python -m feasibility.replay`.
- Per rep: score with SR's PICO (title+abstract where enriched), sample per design, reveal z1 (0.5 → counted 0.5), reveal full text for sampled z1 (z2 / ft_missing), assign quant per outcome by drawing k_o of the true z2 records without replacement, estimate with LOSO priors, compare to true k_o.
- Designs grid: n_total ∈ {100,150,200,250,300} × census_top ∈ {50,100} × bounds ∈ 2 options.
- [ ] Steps: run; pick smallest design meeting acceptance; if none, tune prior concentration cap / decision thresholds on LOSO only, rerun; write report with per-SR breakdown and Wilson CI on coverage. Commit report + chosen `design_default.json` + `priors.json`.

### Task 7: Search APIs + population

**Files:** Create `feasibility/search_apis.py`, `feasibility/population.py`; fixtures `tests/feasibility/fixtures/{pubmed_esearch.json,pubmed_efetch.xml,europepmc.json,openalex.json}`; Test `tests/feasibility/test_search.py`.

**Interfaces — Produces:** `pubmed_search(query, api_key=None, mailto=None, cache_dir=None, fetch=True) -> SearchResult(db,query,count,records:pd.DataFrame,partial:bool,notes:list[str])`, same for `europepmc_search`, `openalex_search`; `year_split(count_fn, lo, hi, cap=9999) -> list[tuple[int,int]]`; `unique_population(results:list[SearchResult], uploads:pd.DataFrame|None, count_only:dict[str,int], overlap_prior:StagePrior) -> (records_df, extra_N:tuple[mean,sd])`.
- Transport function injectable (`_get`) so tests run offline on fixtures.
- [ ] Steps: tests parse fixtures, OpenAlex inverted-index → abstract, year_split recursion, retry then partial flag. Implement, commit. Live smoke script `tools/smoke_apis.py` for the user's PC.

### Task 8: Full-text detector + local LLM

**Files:** Create `feasibility/fulltext.py`, `feasibility/llm_local.py`; Test `tests/feasibility/test_fulltext.py`.

**Interfaces — Produces:** `detect_quant(text:str, outcome:str, synonyms:list[str]) -> dict(status in {"likely","figure_only","not_found"}, evidence:list[str], pages:list[int])`; `fetch_oa_fulltext(pmcid=None, doi=None, email=None) -> str|None`; `ollama_available() -> bool`; `ollama_judge(prompt:str, model:str) -> dict|None`.
- [ ] Steps: tests on synthetic texts: "TG (mg/dL) 120.5 ± 10.2, n = 8" → likely; "TG decreased (Fig. 2)" → figure_only; absent → not_found. Implement, commit.

### Task 9: UI page + integration

**Files:** Create `feasibility_page.py`; Modify `app.py` nav (add item "가능성 예측" and dispatch).
- Steps 1–6 per spec §11; state via `save_project_state(active, "feasibility", ...)` on every label click.
- [ ] Steps: `python -c "import ast;ast.parse(open('feasibility_page.py').read())"`; streamlit headless run with `streamlit run app.py --server.headless true` + curl health → 200. Commit.

### Task 10: Final verification + packaging

- [ ] Full test run, replay report present, CODEMAP updated with feasibility rows, zip `SR_Studio_V37_Feasibility.zip` excluding `library` data/.git, deliver.
