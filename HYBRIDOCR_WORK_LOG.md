# HybridOCR 전체 작업 기록

작성 기준: 2026-10-21 · 작업 경로: `/home/user/Desktop/RADU/HybridOCR`

이 문서는 프로젝트 준비, 모델 연결, 웹 검수 구현, 비교 실험, 정답 라벨 반영, 추가 이미지 준비와 후속 설계를 통합한 기록이다. 저장된 코드·실험 산출물·검증 문서를 기준으로 작성했다. 구현 및 실행 완료 항목과 아직 제안 단계인 항목을 구분한다. 기존 RADU의 기능 전체를 HybridOCR에서 새로 개발한 것으로 계산하지 않는다.

## 최신 추가 작업: 표·화살표 규칙 및 개별 라벨 구현

2026-10-21 후속 작업으로 표·직선 화살표 규칙을 코드로 구현하고 새 분석 경로에 연결했다. 이미지별 라벨 JSON 122개를 생성했으며 기존 5장 전사는 보존하고 나머지 117장은 미라벨 양식으로 준비했다. 개별 라벨 병합 도구와 검사도 추가했다. 상세 내용과 지원 한계는 [구현 기록](docs/STRUCTURE_RULES_IMPLEMENTED.md)에 있다. 아래 설계·권장 방향 부분은 그 시점의 이력이며 현재 구현 상태는 이 추가 기록을 따른다.

## 1. 목표와 작업 범위

목표 파이프라인은 **이미지 → PaddleOCR-VL 1.6 → 원본 이미지와 OCR JSON을 Qwen2.5-VL에 전달 → 교정 결과 생성 → 사용자의 시각 검수·수정 → PDF 다운로드**이다.

대상은 한국어·영어가 혼합된 손글씨 문서이며 표, 화살표, 수식과 도형을 포함한다. 실행 환경은 사용자의 RTX 4090 워크스테이션이다. 데이터는 Hugging Face의 `hakuuung/HybridOCR_CustomNotes`를 사용한다.

사용자의 요청에 따라 RADU 전체를 RADU 내부의 `HybridOCR` 폴더로 복사하고 그 안에서 작업했다. 이후 사용자는 JSON 직접 편집 대신 벡터 요소를 시각적으로 수정하도록 요청했다. 해당 기능을 구현한 뒤, 웹 기능의 추가 개발은 후순위로 미루고 Qwen 유무에 따른 성능 비교를 우선 진행했다. 표·화살표 구조 처리는 규칙 기반으로 개선하기로 방향을 정했다.

## 2. 진행 이력

| 단계 | 진행 내용 | 상태 |
|---|---|---|
| 작업본 준비 | RADU 복사본과 실행 진입점 준비 | 완료 |
| 모델 연결 | 로컬 PaddleOCR-VL 및 Qwen 가중치, 실제 GPU 추론 확인 | 완료 |
| 기본 웹 파이프라인 | 업로드/HF 표본 로딩, 분석, 수정 저장, PDF 출력 | 완료 |
| 벡터 검수 | 텍스트·표 셀·도형·화살표 직접 수정 및 벡터 PDF | 구현·기능 검증 완료 |
| 비교 평가 설계 | 동일 입력·동일 OCR을 공유하는 Qwen 유무 비교 | 완료 |
| 2026-10-20 예비실험 | 고정 5장 추론, 시간·메모리·변경량 비교 | 완료 |
| HTML 보고서 | 원본과 두 조건의 결과를 시각 비교 | 완료 |
| 2026-10-21 정답 반영 | 사용자 전사 5장 저장 및 기존 예측 재채점 | 완료 |
| 추가 라벨링 준비 | 나머지 117장 저장, 별도 전사 양식 생성 | 완료 |
| 표·화살표 규칙 | 기존 후보 검출 확인 및 개선 설계 문서 작성 | 초기 규칙 구현·연결 완료, 구조 정확도 미검증 |
| 추가 학습 | 모델 미세조정 | 수행하지 않음 |

## 3. 실행 환경과 모델

| 항목 | 비교 실험에 기록된 설정 |
|---|---|
| GPU | NVIDIA RTX 4090 24GB |
| OCR | PaddleOCR-VL 1.6 |
| 교정 모델 | Qwen2.5-VL-7B-Instruct |
| PyTorch | 2.7.1 |
| Transformers | 5.17.0 |
| Paddle GPU | 3.2.1 |
| PaddleOCR / PaddleX | 3.7.0 / 3.7.2 |
| Paddle 입력·출력 상한 | 602112 픽셀 / 2048 생성 토큰 |
| Qwen 입력·출력 상한 | 1605632 픽셀 / 6144 생성 토큰 |

로컬 모델은 `models/vision_language/` 아래에 있다. 두 모델은 별도 프로세스로 순차 실행하여 GPU를 사용한다. 복사된 가상환경은 기존 `/home/user/Desktop/HybridOCR/.venv`의 패키지를 `.pth`로 참조하는 구성이므로 다른 기기에 옮길 때 의존성을 새로 설치해야 한다. CLI shebang 대신 `.venv/bin/python -m ...` 실행을 사용한다.

README 하단에는 복사 전 RADU 시기의 CPU 실행 설명도 남아 있다. 이는 이번 RTX 4090 비교 실험의 실행 조건이 아니다.

## 4. 파이프라인 구현

1. 입력 이미지의 EXIF 방향을 보정하고 RGB PNG로 저장한다. 현 Hybrid 경로에서 자동 원근 보정은 적용하지 않는다.
2. PaddleOCR-VL에 `Spotting:` 질의를 보내 텍스트와 위치를 추출하고 OCR 줄마다 고정 ID를 부여한다.
3. 구조 검출 코드에서 도형·화살표 위치 후보를 생성한다. OpenCV 후보는 오검출을 포함할 수 있다.
4. Paddle 프로세스 종료 후 Qwen에 동일 원본 이미지와 OCR·구조 후보 JSON을 전달한다.
5. Qwen이 반환한 ID별 교정문을 기존 OCR 위치에 연결한다. 임의 좌표를 생성하는 방식은 기본 경로에서 제거했다.
6. 원본 OCR·Qwen 응답과 사용자 편집 벡터 상태를 구분해 저장한다.

초기 시도에서 Qwen 자유 좌표 생성이 반복되며 토큰 상한에 도달했다. 현재는 좌표와 OCR 순서를 고정하고 텍스트 교정을 중심으로 사용한다. 비정상 JSON escape를 확인한 후 파서·전사 지시도 보완했다. 잘린 생성 응답은 성공으로 처리하지 않는다.

작업별 저장 위치는 `data/hybrid/<작업 ID>/`이며 `source.png`, `ocr.json`, `review.json`, `review.raw.txt`, `vector.json`, `state.json`, `stage.txt`, `run.log` 등을 보관한다.

### 실제 실행 확인 사례

- HF 원본 → Paddle → Qwen → 웹 표시: 작업 ID `e745fbc9af124029b3acd68b2f118197`.
- 수식·표·순서도 혼합 검증 이미지: Paddle에서 59줄·59개 위치 추출 확인.
- ID 기반 교정·벡터 구성: 작업 ID `2e05f646a9934559b94752ef44a917d7`. OCR 교정 59개와 구조 후보 44개(도형 40, 화살표 4)를 합쳐 103개 벡터를 구성했다. 이는 검출 정확도가 검증되었다는 뜻이 아니다.

## 5. 웹 검수와 PDF

기존 RADU의 SVG 편집·출력 코드를 재사용해 HybridOCR 작업 저장소에 연결했다. 사용자에게 내부 JSON 직접 편집을 요구하지 않는다.

- 원본 위 텍스트 선택·더블클릭 수정, 위치 이동·크기 변경.
- 다중 선택, 원본·오버레이·벡터·나란히 비교, 확대·화면 이동.
- 표 영역을 직접 생성하고 셀별 텍스트·사각형 편집.
- 도형·화살표 추가, 화살표 끝점·방향·연결 요소 수정.
- 자동 저장, 검수 완료 저장, 수정 후 확인 상태 해제, revision 기반 저장 충돌 감지.
- 현재 편집된 벡터로 PDF 생성. 원본 이미지와 편집 핸들은 출력하지 않고 원본 페이지 좌표·비율을 유지한다.

표 셀은 독립 사각형이므로 한 셀 크기 변경 시 인접 셀 경계가 자동으로 함께 움직이지 않는다. 병합 셀 자동 복원과 원본 수준의 레이아웃 재현은 미완료다. 현재 웹은 한 이미지·한 페이지 검수 기준이며 서버 worker 1개로 실행한다. 실행 중 서버 재시작 시 분석 재요청이 필요하다.

웹 상세 사용법: [HYBRIDOCR.md](HYBRIDOCR.md).

## 6. Qwen 유무 비교 실험 설계

### 데이터와 표본 고정

- 데이터셋: `hakuuung/HybridOCR_CustomNotes`.
- revision: `7f97f382d9d82aa6b21e9b617b46218f1ec4b0aa`.
- split: `train`, image 컬럼만 있는 122장, parquet shard 2개.
- seed 42로 전역 행 `[81, 14, 3, 94, 35]`를 중복 없이 선택했다.
- 표본 ID: `hf-0081`, `hf-0014`, `hf-0003`, `hf-0094`, `hf-0035`.
- 모델 출력 확인 전에 manifest와 이미지 SHA-256을 고정했다.

웹 표본 로더는 shard·행 선택 방식이라 전체 행 균등 추출을 보장하지 않는다. 이번 평가용 prepare는 전역 행에서 균등 추출한 별도 경로다.

### 두 조건

| 조건 | 구성 |
|---|---|
| 기준선 | 원본 → PaddleOCR-VL → OCR 줄 텍스트 결합 |
| Hybrid | 같은 OCR 결과 + 원본 이미지 + 구조 후보 → Qwen → 동일 줄 ID에 교정 적용 |

이미지별 OCR은 한 번만 실행하고 두 조건이 공유한다. 좌표·OCR 줄 순서는 바꾸지 않는다. 모델별 내부 전처리와 픽셀 상한은 다르다. 정답이나 사용자 수정을 모델 입력에 포함하지 않았다. 입력·프롬프트·실행 소스 해시와 환경을 결과에 기록했다.

## 7. 정답 확보 전 측정 결과

| 지표 | Paddle 단독 | Paddle + Qwen |
|---|---:|---:|
| 평균 응답 시간 | 16.32초 | 34.32초 |
| 중앙값 응답 시간 | 12.92초 | 35.63초 |
| 표본 p95 응답 시간 | 22.21초 | 39.11초 |
| 정상 완료 | 5/5 | 5/5 |
| 실패 | 0/5 | 0/5 |
| 전체 경로에서 관측한 GPU 최대 사용량 | 21300 MiB | 21300 MiB |

Qwen 추가 시 평균 응답 시간은 2.10배이며 평균 18.00초가 추가됐다. Qwen 요청 단계 평균은 17.39초이고 나머지는 구조 후보 생성 시간이다. 전체 OCR 81줄 중 36줄(44.4%)이 변경됐다. 변경률은 올바르게 고친 비율이 아니다.

응답 시간은 매번 새 프로세스·모델 초기화·종료 및 확인 지연을 포함한다. 모델 상주 서버의 처리량을 측정한 수치가 아니다. GPU 수치는 0.5초 간격으로 샘플링한 장치 전체 사용량으로 바탕화면 등 다른 프로세스도 포함한다. Qwen 단계만의 최대는 19072 MiB였지만 Hybrid 전체 최대값으로 사용할 수 없다.

이 시점에는 정답이 없어 CER·WER를 계산하지 않았다. 당시 기록은 [초기 실험 보고서](docs/QWEN_ABLATION_PILOT_RESULTS.md)에 보존했다.

## 8. 시각 비교 HTML

원본 이미지, Paddle 결과, Qwen 반영 결과를 한 화면에서 비교하는 정적 HTML을 생성했다.

- 줄 선택 시 해당 원본 영역 확대 및 OCR 영역 표시.
- 변경된 줄 필터, 텍스트 검색, 문자 추가·삭제 강조.
- 표본별 TXT 다운로드, 모델 확인 요청·실행 조건 확인.
- 정답 반영 후 전체 CER·WER, 표본 CER, 전사 원문·작성자·notes 표시.

파일: [비교 보고서](data/benchmark/qwen_ablation_pilot/index.html).

빨강·초록 강조는 두 모델 사이의 문자열 차이다. 정답과 대조한 문자별 정오답 색상은 아니다. HTML은 저장된 결과를 사용하며 열 때 모델 추론을 실행하지 않는다. 직접 파일로 열거나 로컬 HTTP 서버로 제공할 수 있다. 서버 접속 불가 시 아래 실행 명령으로 다시 띄운다. 다른 기기에서 localhost 주소를 열면 이 워크스테이션에 연결되지 않는다.

## 9. 사용자 정답 라벨 반영

사용자가 작성한 5장 전사를 `references.json`에 저장하고 모두 `verified`로 반영했다. 전사 철자·기호·대소문자·작성자·주석은 제공 내용을 보존했다.

| 최종 ID | 작성자 | 주요 전사 조건 |
|---|---|---|
| hf-0014 | 이나현 | 특이사항 빈 문자열 |
| hf-0003 | 정하경 | 화살표·오른쪽 포스트잇 제외, 곱셈은 X |
| hf-0035 | 김혜린 | 제공된 notes 문구 그대로 유지 |
| hf-0081 | 김혜린 | 줄 사이 괄호 내용 제외 |
| hf-0094 | LEEJINWOO | 분수는 /, 반복부호 ″ 유지 |

사용자가 제출한 첫 ID `hf-001`은 이미지 내용 확인 후 `hf-0014`에 대응시켰다. 원래 ID는 `submitted_id`와 provenance에 남겼다.

모델 추론은 다시 실행하지 않았다. 기존 예측을 그대로 재채점했으며 원본 결과와 표본별 출력 해시는 `prediction_integrity.json`으로 확인했다. 정답 반영 전 집계·HTML 스냅샷·그래프는 `archive/before_user_labels/`에 보관했다.

## 10. 정답 기반 성능 비교

CER·WER는 낮을수록 좋은 오류율이다. CER은 문자, WER는 공백으로 구분한 어절 기준 Levenshtein 거리다. NFC 정규화·연속 공백 축약·앞뒤 공백 제거 후 대소문자와 문장부호를 유지한다. micro는 전체 오류 수 합 / 전체 정답 길이 합이며 macro는 문서별 비율 평균이다. 삽입이 많으면 100%를 넘을 수 있어 100%로 제한하지 않는다.

| 지표 | Paddle 단독 | Hybrid | 감소 폭 |
|---|---:|---:|---:|
| CER micro | 63.26% | 61.49% | 1.76%p |
| 공백 제거 CER micro | 69.79% | 66.99% | 2.80%p |
| WER micro | 88.87% | 83.30% | 5.57%p |
| CER macro | 63.03% | 62.34% | 0.69%p |
| WER macro | 89.36% | 86.00% | 3.36%p |
| 문서 완전 일치 | 0/5 | 0/5 | 동일 |

- CER: 정답 2267자, 오류 1434 → 1394.
- 공백 제거 CER: 정답 1751자, 오류 1222 → 1173.
- WER: 정답 521어절, 오류 463 → 434.

| 표본 | Paddle CER | Hybrid CER | Paddle WER | Hybrid WER | CER 변화 |
|---|---:|---:|---:|---:|---|
| hf-0081 | 40.32% | 33.00% | 72.18% | 52.63% | 개선 |
| hf-0014 | 49.71% | 39.77% | 92.31% | 76.15% | 개선 |
| hf-0003 | 83.74% | 80.95% | 98.08% | 93.27% | 개선 |
| hf-0094 | 79.90% | 96.48% | 100.00% | 123.71% | 악화 |
| hf-0035 | 61.47% | 61.47% | 84.21% | 84.21% | 동일 |

3장 개선, 1장 악화, 1장 동일이다. 5장 평균의 소폭 개선이 확인됐지만 전체 122장 또는 다른 손글씨 문서의 개선을 입증한 것은 아니다. 오류율을 단순히 1에서 빼서 인식 정확도라고 표현하지 않는다.

### 오류 분석

**반복 출력:** hf-0094의 `ocr-text-0004`, `0005`, `0006`에서 각 줄 대신 동일한 3줄 묶음이 반환됐다. 이를 각 ID에 반영하면서 반복이 생겨 문자 삽입 오류가 39 → 147로 증가했다. 재채점 때 이 반복을 제거해 점수를 개선하지 않았다.

**OCR 누락:** hf-0003의 정규화 정답은 609자이지만 Paddle 출력은 151자였다. 삭제 오류는 466 → 464로 크게 남았다. 현 교정 경로는 기존 OCR 줄 ID에 연결되므로 OCR에 없는 내용을 새 줄로 복원하지 못한다.

**전사 범위 차이:** 포스트잇과 괄호 내용을 정답에서 제외한 일부 표본도 예측은 전체 페이지 출력이다. 제외 영역 마스크는 적용하지 않았다. 표기 차이와 누락·읽기 순서 영향도 점수에 포함되므로 검출된 글자만의 인식률이 아니다.

## 11. 추가 라벨링 준비

동일 revision에서 나머지 117장을 RGB PNG로 추출해 기존 샘플 폴더에 저장했다. 원본 HF 캐시와 기존 5장은 삭제·이동하지 않았다.

| 항목 | 현재 상태 |
|---|---|
| images/ 전체 | 122장 |
| 검수된 전사 | 5장 |
| 추가 미라벨 이미지 | 117장 |
| 추가 이미지 실제 추론·재평가 | 아직 수행하지 않음 |
| 개발/최종 평가 분할 | 권장 방향만 제시, ID 확정 전 |

파일은 `hf-0000.png`부터 `hf-0121.png` 형식이다. 모든 이미지에 대해 파일 열기 검증을 완료했고 기존 manifest·정답·raw 결과의 해시 불변을 확인했다.

- 전체 출처·행·해시: [additional_images.manifest.json](data/benchmark/qwen_ablation_pilot/additional_images.manifest.json).
- 추가 전사 양식: [references.additional.template.json](data/benchmark/qwen_ablation_pilot/references.additional.template.json).
- 안내: [ADDITIONAL_ANNOTATION.md](data/benchmark/qwen_ablation_pilot/ADDITIONAL_ANNOTATION.md).

기존 `manifest.json`은 여전히 5장 평가용이다. images 폴더가 122장이 됐다고 평가 표본도 자동으로 늘어나지는 않는다. 추가 전사 양식은 기존 평가에 자동 반영되지 않는다. 추가 평가 전 새로운 표본 manifest와 예측·정답 ID 대응이 필요하다.

전사는 원본을 보고 독립 작성하며 확인 후 `status=verified`로 바꾼다. 특이사항이 없으면 `notes`는 빈 문자열로 두고 미전사 `text`는 null로 둔다. JSON 내 따옴표와 줄바꿈을 올바르게 이스케이프한다. 일부 내용이 판독되지 않거나 전사가 미완성이면 draft로 남기고 검수 전 공식 지표에 넣지 않는다.

## 12. 표·화살표 규칙 설계

기존 `src/cv/detector.py`에는 모폴로지 기반 선 처리, Hough 선분, 화살촉 후보 검출이 있다. 현재 후보는 구조 복원 정답이 아니다. 아래는 후속 구현 제안이다.

| 대상 | 규칙 처리 단계 |
|---|---|
| 테두리 표 | 가로·세로 선 추출 → 끊긴 선 결합 → 교차점 → 행·열 → 병합 셀 → OCR 텍스트 배치 |
| 테두리 없는 표 | 반복되는 텍스트 x 정렬·y 간격으로 후보 구성, 목록·문단 오검출 검증 |
| 화살표 | 선 경로 결합 → 화살촉 확인 → 방향 결정 → 경계 거리·진행 방향으로 연결 대상 판정 |
| 모호한 구조 | 연결·병합을 강제하지 않고 미확정과 검수 사유 저장 |

고정 픽셀 임계값보다 페이지 크기·글자 높이·선 굵기 비율을 사용하고, 좌표는 원본 이미지 기준으로 보관한다. 선 제거가 글자 획을 훼손할 수 있으므로 OCR 입력과 구조 검출용 처리를 구분한다. 교차점만으로 화살표 연결을 확정하거나 가장 가까운 글상자에 무조건 연결하지 않는다. 초기 범위는 직선·단방향 화살표와 테두리 표로 시작하는 것이 권장된다.

텍스트 전사와 별도로 표의 cell bbox·row·col·rowspan·colspan, 화살표의 경로·화살촉·source·target·direction 정답이 필요하다. 구조 검출 P/R, 셀 구조 일치, 방향·연결 정확도와 미확정 비율을 따로 평가한다. 아직 이런 구조 지표는 산출하지 않았다.

상세 설계: [TABLE_ARROW_RULES.md](docs/TABLE_ARROW_RULES.md).

## 13. 검증 기록

아래는 기존 작업에서 수행한 검증 기록이다. 이 통합 문서 작성 중 모델 추론이나 브라우저 검사를 다시 실행한 것은 아니다.

- 초기 구현 시 unittest 26개 통과 기록.
- 정답 반영 및 보고서 변경 이후 unittest 36개 통과 기록.
- 실제 브라우저에서 텍스트 수정·이동·크기 변경, 표 생성·셀 수정, 화살표 끝점 변경, 저장·새로고침 복원과 PDF 다운로드 확인.
- PDF에서 수정된 한글과 표 셀 내용을 `pdftotext`로 확인.
- 데스크톱·모바일에서 JavaScript 오류 및 가로 넘침 확인.
- 비교 HTML에서 줄 선택·원본 확대·필터·검색·TXT 다운로드·file:// 열기 확인.
- 정답 공개 후 CER·WER를 별도 동적 계획법 구현과 대조해 기존 거리 계산 결과 일치 확인.
- 재채점 전후 원본 추론 JSON 해시 동일성 확인.
- 추가 이미지 122장 파일 열기 검증 및 기존 평가 파일 보존 확인.

대표 테스트는 `tests/test_hybrid.py`, `tests/test_hybrid_vector.py`, `tests/test_ablation.py`, `tests/test_ablation_html.py`, `tests/browser/hybrid_vector_review.py`, `tests/browser/ablation_html_report.py`다. 기능 검증 성공이 손글씨·표·화살표 인식 정확도 검증 완료를 뜻하지 않는다.

## 14. 현재 판단과 후속 우선순위

시간 제약을 고려해 즉시 학습보다 관측된 반복 출력·누락과 평가 기준을 먼저 개선하는 방향을 제안했다. 아직 다음 개선을 적용한 후의 새 성능 수치는 없다.

1. 전사 포함 영역·기호·읽기 순서를 통일하고 평가 범위 차이를 해결한다.
2. Qwen의 비정상 여러 줄 병합·동일 블록 반복을 감지하고 해당 수정만 원본 OCR로 복구하는 규칙을 개발 표본에서 검증한다. 정상 반복 글자·표 값은 일괄 삭제하지 않는다.
3. 누락이 많은 페이지에 영역 분할·확대 OCR·재시도를 실험한다. 아직 효과가 입증된 방법은 아니다.
4. 다양한 추가 개발용 10~20장과 별도 최종 평가용 약 10장을 우선 라벨링한다. 권장 시작 규모이며 충분한 성능이나 통계적 신뢰도를 보장하지 않는다.
5. 이미 살펴본 5장은 개발용으로 활용하고 새 평가 표본은 규칙 조정에 사용하지 않는다. 중복·같은 원본 변형은 서로 다른 분할에 배치하지 않는다.
6. 표·화살표는 구조 규칙과 소량의 구조 정답으로 별도 검증한다.
7. 그 후에도 선명한 손글씨 인식 오류가 지속되면 대표성 있는 학습쌍을 확보해 미세조정을 검토한다. 페이지 전사만으로 해당 모델의 학습 입력이 준비되는 것은 아니며 영역·정답 대응이 추가로 필요할 수 있다.

단어별 오인식을 나열한 치환 사전으로 특정 5장에 맞추는 방식은 권장하지 않는다. 아직 모델 학습·미세조정, 전체 122장 추론, 표·화살표 구조 정확도 측정, 신규 독립 평가, 다중 페이지 웹 검수는 완료하지 않았다.

## 15. 주요 파일과 재현 명령

| 경로 | 용도 |
|---|---|
| src/hybrid/app.py | HybridOCR 웹/API 진입점 |
| src/hybrid/pipeline.py | OCR·Qwen 순차 실행 및 응답 처리 |
| src/hybrid/document.py | OCR 위치와 교정문으로 벡터 문서 구성 |
| src/hybrid/dataset.py | 웹 HF 표본 로딩 |
| src/cv/ocr_engine.py | OCR 어댑터 |
| src/cv/detector.py | 구성요소 후보 검출 |
| src/evaluation/ablation.py | 표본 고정·실행·재채점 |
| src/evaluation/ocr_metrics.py | 정규화·문자/어절 거리·교정 적용 |
| src/evaluation/ablation_analysis.py | 그래프와 변경 내역 |
| src/evaluation/ablation_html.py | 시각 비교 HTML 생성 |
| data/benchmark/qwen_ablation_pilot/results/results.json | 원본 예측·측정·환경 |
| data/benchmark/qwen_ablation_pilot/results/summary.json | 현재 정답 기반 집계 |
| data/benchmark/qwen_ablation_pilot/results/sample_scores.json | 표본별 치환·삭제·삽입 오류 |
| data/benchmark/qwen_ablation_pilot/results/comparison.csv | 표본별 시간·오류율 |
| data/benchmark/qwen_ablation_pilot/results/accuracy_comparison.png | 오류율 비교 그래프 |
| data/benchmark/qwen_ablation_pilot/results/line_changes.csv | 실제 변경된 OCR 줄 |

프로젝트 폴더에서 실행한다.

```bash
# 웹 검수 서버 (현재 우선 작업은 평가이며 웹 추가 개발은 후순위)
.venv/bin/python -m uvicorn src.hybrid.app:app --host 127.0.0.1 --port 8010

# 저장된 5장 예측의 재채점: GPU 재추론 없음
.venv/bin/python -m src.evaluation.ablation score \
  --manifest data/benchmark/qwen_ablation_pilot/manifest.json \
  --results data/benchmark/qwen_ablation_pilot/results/results.json \
  --references data/benchmark/qwen_ablation_pilot/references.json

# 그래프·변경 내역과 HTML 생성
.venv/bin/python -m src.evaluation.ablation_analysis \
  --results data/benchmark/qwen_ablation_pilot/results/results.json
.venv/bin/python -m src.evaluation.ablation_html \
  --manifest data/benchmark/qwen_ablation_pilot/manifest.json \
  --results data/benchmark/qwen_ablation_pilot/results/results.json \
  --references data/benchmark/qwen_ablation_pilot/references.json

# 정적 비교 페이지: http://127.0.0.1:8013/
.venv/bin/python -m http.server 8013 --bind 127.0.0.1 \
  --directory data/benchmark/qwen_ablation_pilot
```

기존 manifest가 있는 폴더에 prepare를 다시 실행하지 않는다. 새로운 비교 실험은 별도 출력 폴더와 새 manifest를 사용한다. 원본 예측은 보존하며 개발 후 결과와 최초 결과를 구분한다.

## 16. 상세 기록 링크

- [HybridOCR 사용법](HYBRIDOCR.md)
- [구현·브라우저·PDF 검증](docs/HYBRIDOCR_VALIDATION.md)
- [Qwen 유무 평가 프로토콜](docs/QWEN_ABLATION.md)
- [정답 확보 전 예비실험](docs/QWEN_ABLATION_PILOT_RESULTS.md)
- [정답 5장 반영 결과·한계](docs/QWEN_ABLATION_LABELLED_RESULTS.md)
- [표·화살표 규칙 설계](docs/TABLE_ARROW_RULES.md)
- [추가 이미지 전사 안내](data/benchmark/qwen_ablation_pilot/ADDITIONAL_ANNOTATION.md)

복사 전 RADU 웹 기능과 UX 기록은 `docs/WEB_IMPLEMENTED_FEATURES.md`, `docs/WEB_UX_AUDIT_2026-10-19.md`에 별도로 남아 있다. 초기 검증 문서의 무라벨·본문 편집 설명은 당시 상태이며 현재 정답 반영 결과와 벡터 검수 상태는 본 문서 및 최신 상세 문서를 따른다.

### 표·화살표 초기 구현 검증 추가

후속 코드 검증은 전체 unittest 43개 통과. 기존 손글씨 5장에 CPU 구조 검출을 실행해 별도 JSON·오버레이를 저장했다. 제본·종이 선 오검출과 작은 화살표 누락이 남으며 구조 정확도는 미측정이다. 기존 모델 출력 해시가 보존되는지 다시 확인했다. 기존 36개 통과 기록은 당시 이력이다.

## GitHub 공유 준비 — feat/hybridocr

사용자의 요청으로 RADU 원격 저장소에 공유할 feat/hybridocr 브랜치를 만들었다. 이미지별 라벨 122개, 모델·웹·평가·규칙 코드, 실험 증거와 문서를 Git에 포함하도록 설정했다. 원본 이미지 1.7GB, 모델 가중치, 가상환경, 로컬 검수 작업 저장소는 제외한다. 고정 HF revision에서 PNG를 복원하는 CPU 전용 도구, 공동 라벨링 안내, 독립 라벨 검사 및 PR 자동 검사 workflow를 추가했다. 공동 작업 절차는 [ANNOTATION_COLLABORATION.md](docs/ANNOTATION_COLLABORATION.md)에 있다.

검증: unittest 45개 통과, 122개 라벨 검사(verified 5 / unlabeled 117) 통과. 임시 폴더에 hf-0003을 HF 캐시에서 재생성하여 고정 PNG 해시가 일치함을 확인했으며 GPU/OCR 모듈을 불러오지 않았다. Python 표준 라이브러리만으로도 라벨 검사가 실행됨을 확인했다.

기존 편집 모델 JavaScript 검사 4개도 Playwright 내장 Node와 Deno.test 등록 호환 래퍼로 실행해 통과했다. 일반 node/Deno 명령은 설치되어 있지 않아 내장 런타임을 사용했다. 생성된 CSV·SVG 및 원본 JSON의 바이트·해시를 유지하도록 Git 속성을 설정하고 라벨 JSON은 LF로 관리한다.
