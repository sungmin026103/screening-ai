SR Studio v19 + Figure Data Extractor

추가 기능
- 사이드바 메뉴: Figure 값 추출
- PNG/JPG/JPEG 그래프 이미지 업로드
- Y축 전용 보정 (bar graph) / X·Y 2D 보정 (scatter/line)
- Linear / Log 축
- 3x / 5x / 8x / 10x 돋보기
- 화면 맞춤 / 확대 / 축소
- 클릭한 마지막 점을 화살표키로 1 px, Shift+화살표로 10 px 미세조정
- Mean only / Mean±SD / Mean±SE / Mean±95% CI
- SE → SD 자동 변환 (n 필요)
- 95% CI → SD는 ±1.96×SE 가정의 근사값으로 명시
- 소수점 0~4자리
- XY 모드에서 X와 Y 값을 함께 출력
- 결과 복사
- 추출 결과는 프로젝트에 누적/저장하지 않음

구현 파일
- figure_digitizer.py (신규)
- app.py (메뉴 및 라우팅 추가)

주의
- 95% CI의 SD 환산은 normal approximation(1.96)을 사용하므로 작은 n에서 정확한 t 기반 환산과 차이가 있을 수 있습니다.
