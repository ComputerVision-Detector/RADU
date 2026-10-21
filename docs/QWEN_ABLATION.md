# Qwen 유무 비교 평가

**2026-10-21 업데이트:** 사용자 검수 정답 5장을 `references.json`에 반영해 저장된 예측을 재채점했습니다. 현재 수치는 [정답 반영 결과](QWEN_ABLATION_LABELLED_RESULTS.md)를 참고하세요. 아래 무라벨 설명은 최초 실행 시점과 정답이 없는 향후 표본에 대한 안내입니다.

웹 검수·수정 개발보다 동일 입력에 대한 PaddleOCR-VL 단독과 Qwen 검수 추가의 비교를 우선합니다.

## 현재 평가 조건

- 데이터: `hakuuung/HybridOCR_CustomNotes`, `train`, revision `7f97f382d9d82aa6b21e9b617b46218f1ec4b0aa`
- 전체 122장 중 전체 행에 대해 중복 없이 무작위 추출한 5장. seed 42. 고정 global row는 81, 14, 3, 94, 35.
- **데이터셋에는 image 컬럼만 있습니다. 사람이 검수한 전사 정답이 없으므로 CER/WER·정확도는 산출하지 않습니다.** 교정된 줄 수와 문자열 변화량은 정확도 지표가 아닙니다.
- 하드웨어·패키지 버전, 입력 PNG SHA-256, 프롬프트·실행 소스 SHA-256, 모델 설정을 실행 결과에 기록합니다.
- 모델 지시는 평가 결과를 본 뒤 변경하지 않습니다. 수동 수정이나 정답은 모델 입력에 포함하지 않습니다.

두 조건은 동일한 원본 RGB PNG와 **한 번 실행한 동일 OCR 결과**를 공유합니다. 원본 좌표와 OCR 줄 순서를 유지합니다. Paddle 단독은 줄 텍스트를 합친 결과이며, Qwen 추가 조건은 원본 이미지·OCR·구조 후보를 입력하여 반환받은 id별 교정문을 같은 줄에 반영한 결과입니다. 일치하지 않는 id는 반영하지 않고 별도 기록합니다. 현 파이프라인은 OCR에 위치가 없는 누락 문자를 새 요소로 추가하거나 읽기 순서를 복원하는 비교가 아닙니다.

## 실행

```bash
cd /home/user/Desktop/RADU/HybridOCR

# 고정 이미지·출처·정답 양식 생성. 기존 manifest는 덮어쓰지 않습니다.
.venv/bin/python -m src.evaluation.ablation prepare \
  --samples 5 --seed 42 --output data/benchmark/qwen_ablation_pilot

# GPU 추론. 이미지별 결과와 부분 집계가 즉시 저장됩니다.
.venv/bin/python -m src.evaluation.ablation run \
  --manifest data/benchmark/qwen_ablation_pilot/manifest.json
```

다른 표본 수에는 새로운 출력 폴더를 사용합니다. 중단 후에는 같은 코드·manifest에서 `run --resume`으로 완료된 이미지의 재실행을 피할 수 있습니다. 코드가 달라지면 새 결과 폴더를 지정합니다. OCR·Qwen 프로세스는 순차 실행하며, 시간 제한 시 해당 프로세스 그룹을 종료합니다.

## 측정 항목

- **응답 시간**: 각 요청의 새 프로세스·모델 로딩·추론·종료 시간을 포함합니다. 상주 모델의 정상 상태 처리량은 아닙니다. 0.5초 간격의 종료 확인 때문에 짧은 관찰 지연이 포함됩니다.
- **Hybrid 총 시간**: 동일 OCR 요청 시간 + CPU 구조 후보 생성 + Qwen 요청 시간. Qwen 추가 비용과 모델 내부 순수 생성 시간도 별도 기록합니다.
- **메모리**: `nvidia-smi`로 GPU 전체 사용량을 0.5초 간격으로 샘플링합니다. 바탕화면·다른 프로세스도 포함되고 순간 최고점을 놓칠 수 있습니다. 실행 직전 사용량을 뺀 증가량도 기록합니다. Qwen은 Torch 최대 allocated/reserved도 별도 기록합니다.
- **성공·실패**: OCR와 Qwen의 실패를 숨기지 않고 기록합니다. Qwen이 잘린 JSON을 출력하면 실패입니다. 두 조건 모두 성공한 표본에서 응답 시간과 출력 변화량을 비교합니다. 실패율은 전체 실행 표본에서도 보고합니다.
- **교정 변화**: 바뀐 줄/전체 OCR 줄, 문자열 편집 거리, 문자열 변화 비율, 모델이 표시한 불확실 항목 수. 정답과 비교한 수치가 아닙니다.
- **정답 확보 후**: CER, 공백 제거 CER, WER, 완전 일치율. NFC 정규화 및 연속 공백 축약 후 대소문자·문장부호를 유지합니다. WER의 한국어 단위는 공백으로 구분한 어절입니다. micro는 오류 수 합/정답 길이 합, macro는 문서별 비율 평균입니다. 삽입이 많으면 오류율은 100%를 넘을 수 있습니다.

정답 기반 지표는 두 조건이 모두 성공하고 사람이 검수한 정답이 있는 동일 표본에서 비교합니다. Qwen 실패 시 OCR 원문을 반환하는 운영 정책의 지표는 별도 집계하며, 그 집계에는 같은 표본의 단독 OCR 지표도 함께 제공합니다.

## 산출물

`data/benchmark/qwen_ablation_pilot/`:

- `manifest.json`, `images/`: 고정 입력·출처·해시
- `references.template.json`, `ANNOTATION.md`: 독립 전사 정답 작성 양식과 규칙
- `results/results.json`: 단계별 실행 결과·환경·시간·메모리·텍스트
- `results/summary.json`, `results/comparison.csv`, `results/REPORT.md`: 집계와 표본별 비교
- `results/<표본 ID>/ocr.json`, `review.json`, `review.raw.txt`, `paddle.txt`, `hybrid.txt`: 모델 원문·교정 결과
- `results/<표본 ID>/ocr.log`, `qwen.log`: 실패 분석용 로그

## 정답 작성 후 재채점

모델 출력과 독립적으로 이미지만 보고 전사합니다. `references.template.json`을 `references.json`으로 복사하고, 확인한 각 항목의 `text`를 작성해 `status`를 `verified`로 바꿉니다. `draft`와 `unlabeled`는 정확도 집계에서 제외됩니다. 행 id나 manifest 해시는 바꾸지 않습니다. 읽기 순서·표 전사 규칙과 판독 불가 항목은 주석으로 기록합니다.

```bash
.venv/bin/python -m src.evaluation.ablation score \
  --manifest data/benchmark/qwen_ablation_pilot/manifest.json \
  --results data/benchmark/qwen_ablation_pilot/results/results.json \
  --references data/benchmark/qwen_ablation_pilot/references.json
```

재채점은 기존 추론 결과를 사용하며 GPU 추론을 다시 실행하지 않습니다. 표 셀·화살표 연결 정확도는 전사 문자열만으로 평가할 수 없으므로 별도 구조 정답과 평가 설계가 필요합니다. 5장 예비실험으로 전체 122장의 일반화 성능을 주장하지 않습니다.

완료된 결과의 그래프·줄별 변경 내역 생성:

```bash
.venv/bin/python -m src.evaluation.ablation_analysis \
  --results data/benchmark/qwen_ablation_pilot/results/results.json
```

`comparison.png`·`comparison.svg`는 시간·샘플링 GPU 최대 사용량·출력 변경률을 표시하며, `line_changes.csv`·`OUTPUT_CHANGES.md`는 실제 바뀐 문자열을 보존합니다.

## 사람이 보는 HTML 비교 보고서

```bash
.venv/bin/python -m src.evaluation.ablation_html \
  --manifest data/benchmark/qwen_ablation_pilot/manifest.json \
  --results data/benchmark/qwen_ablation_pilot/results/results.json
```

`data/benchmark/qwen_ablation_pilot/index.html`이 생성됩니다. 원본 이미지, Paddle 단독 결과, Qwen 반영 결과를 같은 화면에서 비교합니다. 줄을 선택하면 OCR 좌표에 맞춰 원본을 확대합니다. OCR 영역 표시, 변경된 줄 필터, 텍스트 검색, 문자 단위 삭제·추가 강조, 표본별 TXT 다운로드를 제공합니다. 모델의 확인 요청과 실행 조건은 별도 버튼에서 확인합니다. 정답이 없다는 설명과 강조 색상이 정확성 판정이 아니라는 설명을 유지합니다.

HTML은 모델 추론이나 서버 API 없이 저장된 결과만 읽습니다. 파일을 직접 열거나 로컬 HTTP 서버로 볼 수 있습니다. 원본 PNG를 상대 경로로 참조하므로 다른 곳으로 옮길 때에는 `index.html`, `images/`, `results/comparison.csv`를 같은 폴더 구조로 함께 보관하세요.

```bash
.venv/bin/python -m http.server 8013 --bind 127.0.0.1 \
  --directory data/benchmark/qwen_ablation_pilot
```

접속 주소: http://127.0.0.1:8013/

검증: `tests/test_ablation_html.py`는 diff 문자열 보존·입력 해시 일치·HTML 스크립트 이스케이프를 검사합니다. `tests/browser/ablation_html_report.py`는 실제 Chrome에서 줄 선택과 원본 위치 대응, 필터·검색·강조, TXT 다운로드, 모바일 가로 넘침 및 `file://` 열기를 확인합니다.
