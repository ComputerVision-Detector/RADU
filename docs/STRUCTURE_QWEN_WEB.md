# RADU 웹 · Qwen3-VL LoRA 구조 인식 연결 — 2026-11-16

사용 웹(`src/web`)의 "구조 인식 모델" 선택으로 도식·수식·표 LoRA를 쓸 수 있다. 기본값은 기존 OpenCV 후보 검출이며, 선택하지 않으면 기존 동작과 API 응답이 그대로다.

## 사용

| 웹 선택 | API `structure` | 요청 내용 | adapter |
|---|---|---|---|
| OpenCV 후보 검출 (기존, 기본) | `opencv` | — | — |
| 도식 | `qwen-diagram` | nodes, arrows, relations | checkpoint-6000 |
| 수식 | `qwen-formula` | formulas | checkpoint-18000 |
| 표 | `qwen-table` | tables | checkpoint-18000 |
| 도식 + 수식 + 표 | `qwen-all` | 위 세 요청을 차례로 | 둘 다 |

```text
POST /api/analyze?ocr_engine=paddleocrvl16routed&device=gpu:0&structure=qwen-diagram
```

GPU가 필요하다(CPU는 400). OCR 모드와 독립이라 자동·한국어 전용 모드와 함께 쓴다. PaddleOCR-VL 비교 모드(`paddleocrvl16`)는 RADU 구조를 실행하지 않으므로 이 선택도 적용되지 않는다. Qwen 실행이 실패하면 OpenCV 후보로 대체하고 `structure_unavailable` 경고를 남긴다.

## 결과 형식

- 노드 → `Box` (`shape`은 편집기 도형으로 매핑, 원래 종류는 `semantic_type`). pool·lane은 `flow_node: false`.
- 화살표 → `Arrow` (`start`/`end`가 있으면 그대로 사용).
- 관계 → 모델이 화살표의 source·target 노드를 모두 준 경우 그 값을 쓴다(`endpoint_method: model_relation`). 나머지는 기존 기하 추론.
- 수식 → `Text` (`semantic_type: formula`, `text`·`latex`에 LaTeX).
- 표 → 셀 상자가 있으면 셀마다 `Box` (`semantic_type: table-cell`, `row/col/rowspan/colspan`), 없으면 표 영역 `Box` 하나. 격자는 `structure.tables`에 남는다.
- OCR 텍스트는 지우거나 바꾸지 않는다. 기존 관계 분석이 위치로 도형·셀에 붙인다(`contains`). 수식 영역과 겹치는 OCR 줄도 그대로 남는다.
- `structure.metadata`에 그룹별 adapter, 생성 토큰 수, 시간, JSON 정상 여부, adapter SHA가 있다.

편집기에서 "관계 분석 실행"을 다시 누르면 기존 기하 추론만 적용된다(모델 관계와 `flow_node` 구분은 최초 분석에만 반영).

## 구성

- `src/cv/structure_qwen.py` — 실행, 결과 변환, 관계 결합.
- `src/cv/structure_qwen_worker.py` — `.cache/venvs/train`에서 실행되는 추론 프로세스. 프롬프트·해상도·디코딩은 `src/training/infer_image.py`와 같다(greedy, 최대 4096 토큰).
- `configs/structure_qwen_lora.lock.json` — adapter·설정 SHA 고정. 다르면 추론 전에 실패한다.
- `src/web/backend/app.py`, `src/web/frontend/index.html`, `app.js` — 선택지와 전달. 이 세 파일을 바꿨으므로 `configs/ocr_cv_final.lock.json`의 설치 파일 hash를 갱신했다(release `radu-cv-final-2026-11-16-structure`). OCR 모델·정책·frozen 실험 기록은 그대로다.
- `src/cv/relation.py`, `src/web/backend/analysis_worker.py`는 두 경로 OCR 실험의 `source-lock.json`이 고정하므로 수정하지 않았다. 구조 선택은 환경 변수 `RADU_STRUCTURE_MODEL`로 분석 프로세스에 전달한다.

## 실제 실행 확인 (validation 이미지, RTX 4090)

| 입력 | OCR / 구조 | 결과 | 전체 시간 |
|---|---|---|---:|
| FC-A `writer11_1` | 자동 / 도식 | 도형 8, 화살표 8, 모델 관계 8 (정답 8/8/8), OCR 15줄 | 31초 |
| IBEM `0002130_page01` | 자동 / 수식 | 수식 2 (정답 1), OCR 21줄 | 18초 |
| PubTabNet `PMC1201145_005_01` | 자동 / 표 | 표 영역 1 (셀 상자 없음), OCR 102줄 | 42초 |
| cTDaR `t00110` (worker 직접) | — / 표 | 셀 131 (정답 131) | 74초 |
| 한국어 필기 `hf-0137` | 한국어 전용 / 전체 | 도형·셀 55, 화살표 16, 수식 1, OCR 90줄 | 73초 |
| hdBPMN `ex00_writer0065` | 자동 / 전체 | 도형 10, 화살표 9, 관계 9 + 수식 1·셀 4 오검출 | 38초 |

개수만 비교한 기능 확인이며 정확도 평가가 아니다. 정확도는 `data/benchmark/radu_final_eval/REPORT.md`(validation)를 따른다.

## 한계

- 학습 이미지는 도식·수식·표가 각각 따로 있는 자료다. 한 장에 섞인 문서와 한국어 필기 노트는 검증하지 않았다.
- "전체"는 세 요청을 모두 보내므로 해당 내용이 없는 이미지에서도 수식·표를 만들어낼 수 있다(위 hdBPMN 사례). 이미지 종류를 알면 해당 항목만 선택한다.
- 큰 표는 출력이 길어 1분 이상 걸린다. 4096 토큰에서 잘리면 읽을 수 있는 항목만 배치하고 `structure_truncated` 경고를 남긴다.
- test split 평가는 여전히 하지 않았다.
- GPU 메모리는 약 9.4GB를 쓴다. 요청마다 모델을 새로 적재한다(적재 약 2~7초).

## 라벨 데이터셋 정답 대조 (2026-11-16 추가)

웹 왼쪽 "라벨 데이터셋 (Hugging Face)"에서 직접 라벨링한 노트 페이지를 골라 분석 결과를 정답과 대조한다.

1. 페이지 선택 → "불러오기" (원본 이미지가 입력 이미지로 들어간다)
2. OCR 모드·구조 모델을 고르고 "이미지 분석" → 분석이 끝나면 자동 채점, 요약이 선택 아래에 표시된다
3. "정답 대조" → 정답 박스·예측 박스와 줄별 정답/예측/CER 표

- 이미지: `hakuuung/HybridOCR_CustomNotes` @ `93f3a5c0` 163쪽. 로컬 복원본(`data/processed/ocr_v2/originals/images`)을 쓰고, 없으면 고정 revision에서 내려받아 hash를 확인한다.
- 라벨: Hugging Face 저장소에는 이미지만 있다. 정답은 라벨링 웹이 저장한 `RADU/labeling-data` 브랜치 `50aca983`의 고정 빌드(`data/processed/ocr_v2/pilot-v2/pages.jsonl`)다. 새 라벨을 반영하려면 그 빌드를 다시 만들어야 한다.
- 채점: `src/evaluation/ocr_v2`의 매칭(IoU 0.5, 최대 매칭)과 CER 정의(NFC, 공백 병합)를 그대로 쓴다. 보류 영역은 채점하지 않고 그 안의 예측은 무시한다.
  - 전체 영역 CER: 검출되지 않은 정답 줄을 전부 삭제로 계산
  - 일치 영역 CER: 매칭된 줄만
  - 페이지 CER: 예측 전체를 OCR 순서로 이어 정답 전체와 비교 (순서·정답 밖 예측 포함)
- 전처리가 페이지를 회전·원근 보정하므로(표본 41쪽 중 34쪽 회전) 정답 박스를 같은 변환으로 옮겨 대조한다.
- API: `GET /api/labeled-dataset`, `GET /api/labeled-dataset/{id}/image`, `POST /api/sessions/{session_id}/evaluate-labels?page={id}`. 세션의 원본이 해당 페이지와 픽셀 단위로 같지 않으면 409.
- 코드: `src/web/backend/labeled_dataset.py`, `src/web/frontend/labeled-dataset.js`, `labeled-dataset.css`.

한계: 라벨은 텍스트 줄(text/formula/symbol 유형 추정)뿐이라 도식·표 구조와 수식 LaTeX는 채점하지 않는다. train 129쪽은 한국어 LoRA 학습에 쓰였고, split은 필자 독립성이 확인되지 않은 임시 분할이다. 라벨이 없는 글씨(미라벨 줄, 뒷면 비침)를 읽은 결과는 "정답 밖 예측"으로 집계된다. 한 페이지 점수이며 문서의 test 수치를 대체하지 않는다.

## 자동 모드 기본값과 프로젝트 데이터셋 (2026-11-16 추가)

- 웹에서 OCR이 "자동 모드"이면 구조 인식 모델이 "도식 + 수식 + 표"(`qwen-all`)로 시작한다. 다른 OCR 모드로 바꾸면 OpenCV로 돌아가고, 구조 인식 모델을 직접 고른 뒤에는 그 선택을 유지한다. API의 `structure` 기본값은 계속 `opencv`다.
- `qwen-all`은 해당 내용이 없는 이미지에서도 수식·표를 만들어낼 수 있다(FC-A `writer12_10b`: 수식 1개와 페이지 전체 표 1개 오검출). 이미지 종류를 알면 해당 항목만 고른다.
- "프로젝트 데이터셋"에서 `data/processed/radu_final`의 validation·test 이미지(16개 source, 5,088장)를 골라 입력으로 불러온다. train(약 187만 행)은 목록에 넣지 않았다. 불러오기만 하며, 구조 정답과의 자동 채점은 없다(목록에 정답 개수만 표시).
- API: `GET /api/local-datasets`, `GET /api/local-datasets/{source}/{split}`, `GET /api/local-datasets/image?id=`. 코드: `src/web/backend/local_dataset.py`, `src/web/frontend/local-dataset.js`.
