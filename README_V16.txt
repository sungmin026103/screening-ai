SR Studio V16 Abstract-Aware AI Engine

변경사항
- Title / Abstract 분리 TF-IDF feature
- Abstract sentence-level PICO matching
- Include/Exclude prototype similarity
- obvious exclusion signal features + 표시
- Recall-constrained WSS threshold optimization
- Excel multi-sheet 자동 탐지 (Title+Abstract 우선)
- 기존 Streamlit 단일 앱 구조 유지

주의
- sentence-transformers는 선택 의존성입니다. 설치되지 않은 Community Cloud에서는 sentence-level PICO가 TF-IDF similarity로 자동 폴백합니다.
- REFERENCE_BENCHMARK는 이전 모델 benchmark입니다. V16 독립 benchmark 재검증 전에는 현재 프로젝트의 CV 성능을 우선 해석하세요.
