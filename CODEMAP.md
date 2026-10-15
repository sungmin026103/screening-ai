# CODEMAP — 어떤 분석은 어느 파일을 열어야 하나

이 문서만 읽고 수정할 파일을 정할 수 있도록 작성한 파일 지도. 전체 파일을 읽지 말고 아래 "핵심 함수/클래스"를 grep 해서 해당 위치만 연다.

## 실행 / 테스트 / 문서

| 항목 | 명령 / 위치 |
|---|---|
| 앱 실행 | `streamlit run app.py` |
| 단위 테스트 | `python -m pytest -q tests/feasibility test_ma_core.py test_upload_safety.py test_figure_evidence.py` |
| 셀프테스트(스크립트) | `python selftest_v36.py` (최신), `selftest_v31.py`, `selftest_v30.py` |
| 레거시 단독 실행 | `streamlit run ma_feasibility.py`, `streamlit run ma_feasibility_pro.py` |
| 버전별 변경 이력 | `docs/changelog/` (README_V30~V35, RELEASE_NOTES_V30), 최신은 루트 `README_V36.txt` |
| 설계/계획 문서 | `docs/superpowers/specs/`, `docs/superpowers/plans/` |

## app.py 내비게이션 → 모듈

`app.py`의 `NAV_ITEMS`가 섹션을 정의하고 `if nav == ...` 분기로 렌더링한다. `app.py`는 1,700줄이므로 해당 분기만 연다.

| nav 키 | 화면 | 담당 모듈 |
|---|---|---|
| dashboard | 프로젝트 개요 | app.py (`styles.py`, `projects.py`) |
| import | 문헌 가져오기 · 중복 제거 | importers.py, dedup.py |
| pico | PICO 설정 | app.py (저장은 projects.py) |
| feasibility | 메타분석 가능성 예측 | feasibility_page.py → feasibility/session.py 외 feasibility/ 패키지 |
| screen | AI 스크리닝 | screening.py |
| pdf_analysis | PDF 분석 | pdf_analyzer.py |
| figure_digitizer | Figure 값 추출 | figure_digitizer.py → figure_auto.py |
| analytics | 문헌 분석 | app.py (plotly 직접 사용) |
| meta | 메타분석 Figure | meta_page.py → meta_sections.py 외 |
| export | 내보내기 | app.py, utils.py |

## 분석(기능)별 지도

| 분석(기능) | 진입 파일 | 핵심 함수/클래스 | 테스트 |
|---|---|---|---|
| 문헌 파일 가져오기 (RIS/NBIB/CIW/xlsx) | importers.py | `read_uploaded_file`, `combine_uploads`, `parse_ris`, `parse_nbib`, `parse_ciw`, `standardize_dataframe` | - |
| 중복 제거 (제목/DOI, union-find) | dedup.py | `deduplicate_records`, `screening_export` | - |
| 텍스트/DOI/연도 정규화, 엑셀 저장 | utils.py | `normalize_title`, `normalize_doi`, `safe_year`, `dataframe_to_excel_bytes`, `save_json`, `load_json` | - |
| 프로젝트 저장/불러오기 | projects.py | `create_project`, `list_projects`, `save_records`, `load_records`, `save_pico`, `save_project_state`, `project_progress` | - |
| PICO 설정 | app.py (`nav == "pico"`) | `concept_gate_panel` | - |
| AI 스크리닝: 학습 표본·라벨 병합 | screening.py | `prepare_screening_data`, `build_training_sample`, `validate_human_validation_file`, `merge_training_labels` | selftest_v30/v31/v36.py (`check_screening`) |
| AI 스크리닝: 특징·임베딩 | screening.py | `EmbeddingLookup`, `RuleSignalFeatures`, `PrototypeSimilarity`, `CriteriaSimilarity`, `build_sentence_pico_lookup` | selftest_v36.py |
| AI 스크리닝: 자동 게이트(개념 규칙) | screening.py | `derive_auto_gate`, `parse_gate_rules_text`, `gate_rules_to_text`, `harvest_abbreviations` | selftest_v36.py (`check_auto_gate`) |
| AI 스크리닝: 결과·성능 figure | app.py, screening.py | `ScreeningResult`, `_screening_performance_figure_bytes` (app.py) | selftest_v36.py |
| AI 스크리닝 성능 figure (독립 스크립트) | AI_screening_figures.py | `load_result`, `cv_data`, `cv_predictions`, `main` | - |
| PDF 분석 (필드 자동 추출) | pdf_analyzer.py | `analyze_pdf_bytes`, `extract_pdf_text`, `extraction_to_dataframe`, `ExtractionField` | - |
| 추출 엑셀(outcome 시트) 읽기 | extraction.py | `read_extraction_workbook`, `read_outcome_sheet`, `outcome_name` | - |
| Figure 값 추출 (막대/점 자동 디지타이징) | figure_digitizer.py (UI) → figure_auto.py (엔진) | `analyze`, `find_panels`, `calibrate`, `calibrate_manual`, `to_rows`; UI `render_figure_digitizer` | selftest_v36.py (`check_figure_auto`) |
| Figure 오차 해석 (SE/SD/CI → SD) | figure_evidence.py | `interpret_error`, `classify_plot`, `parse_digitized_csv`, `validate_two_group` | test_figure_evidence.py |
| 메타분석 Figure 화면 (업로드→설정→내보내기) | meta_page.py | `render`, `_load_workbook`, `_results_from_table`, `_forest_tab`, `_table_s2_tab` | selftest_v36.py (`check_meta_numbers`, `check_figures_tables`) |
| 효과크기·풀링 결과 구성, 민감도 표 | meta_sections.py | `result_from_effects`, `result_from_r_sets`, `influence_table`, `loo_table`, `baujat_table`, `gosh_table`, `robustness_table`, `subgroup_analysis_3level`, `section_tables` | selftest_v36.py |
| 메타분석 figure 생성 (forest/LOO/Baujat/GOSH/funnel/trim-fill) | meta_sections.py | `make_figure`, `forest_fig`, `subgroup_fig`, `loo_fig`, `influence_fig`, `funnel_fig`, `trimfill_funnel_fig`, `build_section_zip` | selftest_v36.py |
| 풀링 통계 (3수준 REML, RVE, Egger, trim-fill) | rmeta.py, metaanalysis.py | rmeta: `fit_three_level`, `rma_reml`, `aggregate_cs`, `regtest_sei`, `trimfill_l0`, `selection_model`, `escalc_smd`; metaanalysis: `compute_effect_sizes`, `pool_random_effects`, `eggers_test`, `trim_and_fill`, `prediction_interval`, `ForestSummary` (meta_sections.py가 import하므로 레거시 아님) | selftest_v36.py (`check_meta_numbers`) |
| Forest 스타일/렌더러 | forest_styles.py | `render`, `forest_figure`, `subgroup_figure`, `ForestOptions`, `Pooled`, `save_figure` | selftest_v36.py |
| plotly 간이 forest/funnel (구형 시각화 함수만 해당; 파일 자체는 현역) | metaanalysis.py | `forest_plot_pro`, `funnel_plot_pro`, `run_meta_analysis`, `guess_columns`, `forest_plot_from_R`, `funnel_plot_from_R` | - |
| 전 outcome 일괄 분석·figure·엑셀 | auto_figures.py | `analyze_outcome`, `summary_row`, `make_figures`, `build_results_workbook`, `build_figure_zip` | - |
| 고급 분석 (meta-regression, dose-response, PET-PEESE, selection, p-curve) | advanced.py | `run_advanced`, `parse_dose` | - |
| Supplementary Table S1~S9 (xlsx/docx, Table S2 삽입) | supplementary.py | `build_tables`, `to_xlsx`, `to_docx`, `table_s2_rows`, `build_table_s2_docx`, `insert_table_s2` | selftest_v36.py (`check_table_s2`) |
| 스타일/레이아웃/랜딩 | styles.py | `apply_styles`, `hero`, `kpi`, `metric_cards`, `stepper`, `section_header`, `empty_state` | - |
| 업로드 안전성 (zip/PDF 한도) | upload_safety.py | `extract_pdfs`, `IntakeLimits`, `IntakeError` | test_upload_safety.py |

## 레거시: 메타분석 가능성 (단독 Streamlit 앱, app.py와 연결되지 않음)

신규 `feasibility/` 패키지가 대체할 예정이므로 새 기능은 여기에 추가하지 않는다.

| 분석(기능) | 진입 파일 | 핵심 함수/클래스 | 테스트 |
|---|---|---|---|
| (legacy) PICO 기반 전문 판정, outcome별 k | ma_feasibility.py | `ask_llm`, `pdf_text`, `records_from_upload`, `unique_records` | - |
| (legacy) 근거 연결형 가능성 평가 (Pro, research beta) | ma_feasibility_pro.py | `llm`, `load_corpus` (+ ma_core, figure_evidence, upload_safety 사용) | test_ma_core.py |
| (legacy 공용) 중복 제거·평가 검증·페이지 분할·감사 xlsx | ma_core.py | `deduplicate`, `validate_assessment`, `segment_pages`, `summarize`, `export_audit_xlsx`, `fingerprint` | test_ma_core.py |

## 메타분석 가능성 예측 (feasibility/)

계획: `docs/superpowers/plans/2026-10-15-ma-feasibility-predictor.md`, 설계: `docs/superpowers/specs/2026-10-15-ma-feasibility-predictor-design.md`. 순수 파이썬 패키지(Streamlit import 없음), UI는 `feasibility_page.py` 한 파일뿐. 테스트: `python3 -m pytest -q tests/feasibility`.

| 분석(기능) | 진입 파일 | 핵심 함수/클래스 | 테스트 |
|---|---|---|---|
| UI 화면 (① 입력 → ② 검색 → ③ 표본 판정 → ④ 2차 표본 → ⑤ 전문 확인 → ⑥ 결과), app.py nav `feasibility` | feasibility_page.py | `render`, `_step_inputs`, `_step_search`/`_fetch_all`, `_step_sample`/`_list_mode`/`_card_mode`, `_step_phase2`, `_step_fulltext`/`_paper_panel`, `_step_results`/`_interval_chart`; 상태 키 `feasibility`, `feasibility_records`, `feasibility_uploads`, `feasibility_texts` (projects.save_project_state) | tests/feasibility/test_page_apptest.py (AppTest, 네트워크 없음) |
| 화면 뒤 순수 로직 (입력 파싱, 표본 장부, 추정 입력, XLSX 내보내기) | feasibility/session.py | `parse_outcomes`, `parse_count_only`, `pico_text`, `input_problems`, `load_design`, `stratify`, `draw_initial`, `followup`, `stratum_obs`, `fulltext_obs`, `suggest_quant`, `run_estimate`, `recommend_more`, `validation_badge`, `methods_paragraph`, `export_workbook` | tests/feasibility/test_session.py |
| 검색 API (PubMed/Europe PMC/OpenAlex, 디스크 캐시, 연도 분할) | feasibility/search_apis.py | `pubmed_search`, `europepmc_search`, `openalex_search` → `SearchResult`, `year_split`, `parse_pubmed`, `abstract_from_inverted_index` | tests/feasibility/test_search.py |
| 후보 모집단 (병합·중복 제거, 건수만 DB 보정) | feasibility/population.py | `unique_population` (→ records, extra_N=(mean, sd)) | tests/feasibility/test_search.py |
| 관련도 점수·층화 표본 설계 | feasibility/sampling.py | `relevance_scores`, `make_strata`, `allocate`, `draw_sample`, `neyman_followup` | tests/feasibility/test_sampling.py |
| 단계별 베이지안 추정기 | feasibility/estimator.py | `StratumObs`, `FulltextObs`, `StagePrior`, `estimate_k`, `simulate_k`, `default_priors` | tests/feasibility/test_estimator.py |
| 라이브러리 기반 사전분포 (경험적 베이즈) — CLI `python3 -m feasibility.priors --lib DIR --out PATH` | feasibility/priors.py, feasibility/priors.json | `fit_priors`, `load_priors`, `save_priors`, `resolve_q3`, `outcome_family`, `beta_mom` | tests/feasibility/test_priors.py |
| 전문(full-text) 확보·정량 데이터 탐지 | feasibility/fulltext.py | `fetch_oa_fulltext_ex` (Europe PMC XML → Unpaywall PDF), `detect_quant`, `xml_to_text` | tests/feasibility/test_fulltext.py |
| 로컬 LLM 연동 (Ollama, 제안 전용) | feasibility/llm_local.py | `ollama_available`, `ollama_models`, `llm_screen_abstract`, `llm_assess_fulltext`, `CPU_DEFAULT`, `GPU_DEFAULT` | tests/feasibility/test_fulltext.py |
| SR 라이브러리 ETL (manifest/etl/enrich) | feasibility/library/manifest.py, etl.py, enrich.py — CLI `python3 -m feasibility.library.etl --build [--lib] [--data] [--uploads]` (출력: `$FEASIBILITY_LIBRARY_DIR`, 기본 ~/Documents/SR_feasibility_library, repo 밖) | `MANIFEST`, `normalize_label`, `combine_decision`, `detect_ft_missing`, `drop_junk_rows`, `link_titles`, `fuzzy_title_match`, `load_sr`→`SRRecord`, `count_outcomes`, `build_library`, `enrich_abstracts`, `load_abstract_sources` | tests/feasibility/test_etl.py |
| 재연 검증 (LOSO replay, 수용 기준) — CLI `python3 -m feasibility.replay --lib DIR [--quick \| --full] [--reps N] [--workers N]`; `--full`은 docs/validation/와 feasibility/design_default.json·priors.json을 다시 씀 | feasibility/replay.py | `replay_sr`, `run_validation`, `metrics`, `nested_loso_tune`, `select_design`, `pipeline`, `write_design_default` | tests/feasibility/test_replay.py, docs/validation/replay_report.md, replay_results.csv |
| 기본 설계·검증 성적 (화면이 실행 시점에 읽음, 값 하드코딩 금지) | feasibility/design_default.json | 키 `n_total`, `census_top`, `bounds`, `phase2`, `phase1_frac`, `decision`, `threshold`, `cap`, `validated`, `metrics_real`, `metrics_real_nested`, `srs` (session.`load_design`, `validation_badge`) | tests/feasibility/test_session.py |

**라이브러리 ETL 노트 (2026-10-15 빌드).** 전문 시트 행은 L1–L8 모두 정규화 제목 exact로 제목 모집단에 연결됨(fuzzy 0, 미매칭 0, added_from_ft 0). 초록 매칭률(전체/z1): L1 97.9/98.8%, L6 97.3/100%, L7 95.7/100%, L8 98.7/98.8%, L2 0.3/0%, L3 0.5/0%, L4 5.4/4.1%, L5 4.7/0% — L2–L5는 초록 소스(EndNote export·RIS/NBIB/CIW)에 해당 SR 레코드가 없음. fuzzy는 token_set_ratio ≥95 + 길이비 ≥0.8 가드(부분집합 제목 오매칭 방지). L7은 전문 단계 미완(ground_truth_complete=False).

## 메뉴를 옮겨도 유지되는 입력·계속 도는 작업 (V37.1)
| 하는 일 | 파일 | 핵심 | 테스트 |
|---|---|---|---|
| 입력 위젯 값·업로드 파일을 다른 메뉴에 갔다 와도 유지 (위젯을 감싸서 기억; 버튼류 제외, `fz:` 키 제외) | keepalive.py | `install`, `begin_run`, `forget(prefix)` | tests/app/test_jobs_keepalive.py |
| 오래 걸리는 계산을 스레드로 실행, 끝나면 `collect()`가 화면 상태에 반영, 사이드바·해당 화면에 진행 표시 | jobs.py | `submit(project, name, label, fn, apply)`, `panel`, `sidebar_status`, `collect`, `wait`(테스트) | tests/app/test_jobs_keepalive.py, tests/app/test_menu_switch_e2e.py |
| 작업으로 돌리는 곳 | app.py: `screen_sample`·`screen_rule`·`screen_train`·`pdf_extract` / feasibility_page.py: `fz_fetch`·`fz_llm`·`fz_oa:<rid>`·`fz_ftllm:<rid>` | | |
규칙: 작업 함수 안에서 `st.*` 금지, 디스크 저장은 작업 함수 안에서, 화면 상태 변경은 `apply`에서. 새 긴 작업은 `jobs.submit`으로 추가. 서버를 끄면 실행 중 작업은 중단된다.
