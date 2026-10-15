# SR Studio - PICO-agnostic Full-text Feasibility (prototype)

## 실행
기존 requirements.txt 설치 후 `streamlit run ma_feasibility.py`.

## 지원
- 임의의 PICO/PECO, 제외 기준, Outcome 및 데이터베이스 검색식 입력
- CSV/XLSX/RIS 서지 레코드 가져오기, DOI/제목 기준 기본 중복 제거
- OpenAI API 키를 사용한 제목·초록 예비 적격성 판정
- 사용자가 업로드한 PDF 전문 텍스트 판독, 근거/Outcome별 정량 데이터 판정
- confirmed/uncertain/unavailable 집계, CSV 내보내기

## 한계 (필독)
- **별도 OpenAI API key 및 사용 요금 필요**. API key 없이 AI 판정 불가.
- 자동 데이터베이스 검색/유료 전문 다운로드는 포함되지 않음. 사용자가 검색 파일과 PDF를 제공해야 함.
- 검색식은 평가 문맥으로만 쓰며, 검색식을 학습하여 k를 예언하는 모델이 아님.
- 스캔 PDF OCR, 이미지 내 미세한 error bar 판독, PDF와 서지파일 간 자동 매칭, 보충자료 자동 다운로드 미지원.
- PDF 길이가 제한을 넘으면 앞부분만 입력됨. 판독 범위 제한 때문에 누락 가능.
- AI가 주장하는 수치 및 근거를 원문에서 재확인해야 함. 확정 k가 아닌 **검토 후보 수**.
- 한 연구의 중복 출판, 공유 대조군, 다중 시점, 중복 효과크기 등은 수동 확인 필요.
- 민감한 미공개 PDF는 외부 API로 전송하지 마세요. 사용권 및 기관 정책 확인 필요.
- 사용자가 제공한 API 키는 프로그램 내부에서만 요청에 쓰며 저장하지 않음.
