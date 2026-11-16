# V2 OCR 파일럿 구현·검증 결과

2026-11-09에 GitHub `RADU/labeling-data` 최신 SHA를 확인하고 `50aca983da9885d9e7a965c1ef4b05055d47b252`로 고정했다. 원본 라벨 SHA256은 `dc178e259b69180507176b80fd17696e6d99a2ebc91beef007af6a117e4f5299`이다. [스냅샷 manifest](snapshot.manifest.json)에 catalog/assets 해시와 이미지 revision을 기록했다. 작업 기준 브랜치는 `feat/radu-lora-training`, HEAD `b3f65d9163cba05d584d108012edcb19b99b2d37`이며 적용되는 AGENTS.md는 발견되지 않았다. 기존 미추적 프로젝트·가중치·체크포인트는 보존했다. 추가 파일만 만들었고 commit/push·학습·배포·운영 OCR 교체는 실행하지 않았다.

실행 명령·입출력 계약·의존성은 [OCR_V2_PIPELINE.md](../OCR_V2_PIPELINE.md), 실제 환경과 추가 코드 해시는 [environment.json](environment.json)에 있다.

| 항목 | 최신 파일럿 pilot-v2 |
|---|---:|
| 원본 페이지 / bbox | 163 / 3,908 |
| verified 페이지 / reviewed=true bbox | 163 / 3,908 |
| uncertain / 빈 전사 / 중복 ID·순서 페이지 | 0 / 0 / 0 |
| 원본 해시·크기 검증 | 163장 통과, 실패 0 |
| 기본 사용 가능 페이지 / 실제 crop | 147 / 3,569 |
| 보류 페이지 / 보류 bbox | 16 / 339 |
| 전체 페이지 분할 train / validation / test | 129 / 17 / 17 |
| 보류 제외 페이지 분할 | 113 / 17 / 17 |
| 실제 crop 분할 | 2,890 / 393 / 286 |
| 일반 텍스트 제안 후보 | 3,045 |
| PaddleOCR-VL 인식 학습 과제 목록 train / validation / test | 2,513 / 297 / 235 |
| VL tokenizer 한도 초과·unknown / 최대 label token 수 | 0 / 70 |
| PP-OCRv6 문자 사전 통과 train / validation / test | 733 / 77 / 50 |
| PP-OCRv6 미지원 문자 제외 | 2,185 |
| 개행 포함 / 기호 전용 bbox | 21 / 130 |
| LaTeX 흔적 제안 bbox | 434 |
| 원본 no_text 페이지 / 사용 가능한 no_text 페이지 | 1 / 0 |
| 동일 RGB 이미지 / 유사 이미지 후보 쌍 | 0 / 4 |
| IoU ≥0.8 중복 영역 후보 쌍 | 26 |
| 작성자·노트·도안·원본 변형 그룹 메타데이터 | 모두 미확보 |

LaTeX 후보 434는 백슬래시 명령·escaped 괄호·달러 기호를 넓게 잡은 결과다. 사전 수치 418과 탐지 규칙이 같다고 가정하지 않았다. `$()` JavaScript 등 오분류 가능성이 있어 유형은 확정값이 아니다. 긴 전사 기준 256 codepoint 초과는 0개이며 최대 241이다. 여러 분류는 중첩될 수 있다.

[품질 보고서](quality_report.json), [세부 감사](audit_details.json), [보류 목록](held_pages.json), [유사 이미지 후보](similarity_candidates.json), [분할 목록](splits.json)을 보존했다. 페이지/그룹을 먼저 분할했고, annotator를 writer로 사용하지 않았다. 분할은 **필자 독립성을 검증하지 못한 임시 파일럿**이다. test는 모델 선택·IoU 조정·추론 스모크에 사용하지 않았다. 자동 좌표·해시 감사는 전체 163장에 수행했지만 전체 페이지의 전사 완전성을 새로 시각 검수했다고 주장하지 않는다.

실제 원본에서 hf-0104의 두 열 손글씨, hf-0011의 분수 분모와 미해결 병합 메모를 확인해 보류했다. 대표 오버레이에서는 hf-0000의 제목·포스트잇 글자가 bbox 밖에 남은 문제를 추가 확인해 pilot-v2부터 보류했다. 이외에는 일반 텍스트 개행 영역, 거의 동일한 중복 bbox, 미해결 주석을 보수적으로 페이지 전체 보류했다. 원본 정답을 임의 수정하거나 OCR 후보로 채우지 않았다. [시각 검수 기록](visual_review.json)과 [최신 대표 train crop](representative_train_crops.jpg)이 있다. 실제 상세 큐는 `data/processed/ocr_v2/pilot-v2/review_queue.jsonl`의 590개 기록이며, 중복/유형/전사 규칙 검수 기록이 겹칠 수 있다.

최신 실제 산출물은 다음 경로에 있다. 대용량 파일은 기존 `.gitignore`에 따라 로컬에 남는다.

- 스냅샷: `data/processed/ocr_v2/snapshot-50aca98/`
- 원본과 페이지별 실제 revision 영수증: `data/processed/ocr_v2/originals/`
- 페이지 GT·crop 3,569개·오버레이 12장·검수 이미지·그룹 입력 양식: `data/processed/ocr_v2/pilot-v2/`
- PaddleOCR-VL `OCR:` query와 원본 label을 연결한 JSONL 목록: `data/processed/ocr_v2/paddleocrvl16-lists-v2-final/`
- 별도 PP-OCRv6 비교용 인식 목록과 미지원 원문: `data/processed/ocr_v2/ppocrv6-lists-v2/`
- 저장 예측 JSON 기반 평가 실행 증거: `data/processed/ocr_v2/saved_prediction_checks_v2/`
- CPU 인식 스모크: `data/processed/ocr_v2/smoke.recognition.cpu.v2.json` 및 `.log`

검출은 IoU 0.5 최대 일대일 매칭과 누락·오탐·분할/병합 후보를, 인식은 GT crop의 NFC 기반 micro CER·공백 제외 CER·줄/영역 일치·편집 수를, 전체 OCR은 모든 예측 영역의 읽기 순서를 포함한 페이지 CER을 계산한다. 외부 예측에는 데이터 manifest fingerprint가 필수다. 보류 영역은 정답과 예측 모두에 같은 정책을 적용하고 모호한 겹침은 중단한다. 실패와 빈 결과는 누락 오차에 포함한다. 수식 인식 문자열 지표는 일반 텍스트와 분리되며 수학적 동치성 평가는 아니다.

[관련 회귀 테스트 36개](tests.log)가 통과했다. 검증에는 한글 NFC와 한 글자 편집, 해시·좌표 오류, 그룹 누수, GT/prediction 분리, no_text·uncertain·보류, 일대일 매칭, 누락·오탐, 실패·빈 페이지, 스냅샷/예측 binding, 페이지 GT 변조, 같은 크기의 잘못된 crop 픽셀, VL 토큰 길이 초과가 포함된다. [실데이터 재검증](validation.json)에서 163개 원본과 3,569개 crop의 해시·크기·원본 대비 픽셀·정답·분할 검사를 통과했다. 관련 평가·라벨 공유·bbox 제안 회귀 검사를 포함했으며 저장소의 모든 GPU/브라우저 테스트를 실행한 것은 아니다.

독립 리뷰가 지적한 스냅샷/예측의 데이터셋 연결, 페이지 추론 전 원본 확인, GT 추적 및 crop 픽셀 비교 결함은 실패하는 회귀 테스트를 먼저 확인한 뒤 수정했다. [합성 저장 예측 점검](saved_prediction_checks.json)은 빈 예측과 주입한 추론 실패를 사용해 평가 CLI를 실행한 기록이며 **모델 성능 수치가 아니다**.

로컬 PP-OCRv6 인식기를 CPU에서 GT crop에 직접 실행했다. 최신 데이터의 train 페이지 hf-0001에서 2개 crop 모두 결과를 반환했다. 검출 파이프라인은 재실행하지 않았다. [실제 출력과 가중치 해시](smoke.recognition.cpu.json)를 보존했으며 CER 기준 성능은 측정하지 않았다. 이전 pilot-v1의 hf-0000 스모크는 시각 감사 후 보류된 페이지의 중간 실행 기록으로만 남겼다.

PaddleOCR-VL 1.6 가중치는 로컬에 있으나 CUDA device count가 0이고 NVIDIA 드라이버에 연결되지 않아 전체 페이지 GPU 추론은 실행하지 않았다. 따라서 검출기와 인식기 중 어느 쪽의 실제 validation 오류가 더 큰지 아직 비교할 수 없다. 현재 확인된 실질적인 제약은 PP-OCRv6 비교 인식기의 한글 문자 사전 미지원과 일부 페이지 GT 누락이다. 먼저 보류/유형·그룹 정보를 검수하고, GPU 환경에서 동일 validation 페이지의 전체 OCR과 동일 GT crop 인식을 실행해야 한다. GT crop CER이 낮고 검출 누락/순서 오류가 크면 검출 경로를, 검출 recall이 충분하나 crop CER이 크면 인식 경로를 개선한다. 주 경로인 PaddleOCR-VL과 Qwen 구조 추출은 유지했다.
