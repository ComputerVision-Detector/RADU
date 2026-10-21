# 사용자 정답 반영 재채점 — 2026-10-21

저장된 5장 예측을 사용자 제공 전사 정답으로 재채점했다. GPU 모델 추론은 다시 실행하지 않았고, 원본 OCR·Qwen 출력·측정 시간은 그대로 보존했다.

## 전사 반영

정답 파일: `data/benchmark/qwen_ablation_pilot/references.json`.

제공된 `hf-001`의 레이블·원샷·퓨샷 학습 전사는 원본 내용과 manifest의 `hf-0014`에 대응한다. 채점 id를 `hf-0014`로 연결하고 원래 id는 `submitted_id`에 보존했다. 나머지 네 항목의 id는 그대로 사용했다. 전사 텍스트, 철자, 기호, 대소문자, 작성자와 notes는 제공 내용을 유지했다. `"hip"`, `"hello"` 같은 따옴표는 유효한 JSON 문자열로 저장했다.

이전 무라벨 집계·그래프는 `archive/before_user_labels/`에 보관했다. 모델 결과의 `results.json`과 표본별 JSON 해시는 `prediction_integrity.json`에 기록하여 재채점 전후 일치를 확인한다.

## 전체 오류율

**아래 값은 낮을수록 좋으며 정확도(1−CER)로 환산하지 않는다.** micro는 전체 오류 수 합을 전체 정답 길이 합으로 나눈 값이다.

| 지표 | PaddleOCR-VL 단독 | PaddleOCR-VL + Qwen | 감소 폭 |
|---|---:|---:|---:|
| CER · micro | 63.26% | 61.49% | 1.76%p |
| 공백 제거 CER · micro | 69.79% | 66.99% | 2.80%p |
| WER · micro | 88.87% | 83.30% | 5.57%p |
| 문서별 CER 평균 · macro | 63.03% | 62.34% | 0.69%p |
| 문서별 WER 평균 · macro | 89.36% | 86.00% | 3.36%p |
| 문서 전체 완전 일치 | 0/5 | 0/5 | 동일 |

- CER 분모: NFC·공백 정규화 후 정답 2267자. 문자 오류 합 **1434 → 1394**.
- 공백 제거 CER 분모: 정답 1751자. 오류 합 **1222 → 1173**.
- WER 분모: 공백 기준 521어절. 어절 오류 합 **463 → 434**.
- 처리 시간은 기존 측정 그대로 평균 **16.32초 → 34.32초**. 모델 로딩 포함 응답 시간이다.

## 표본별 결과

| 표본 | Paddle CER | Qwen 반영 CER | Paddle WER | Qwen 반영 WER | CER 판정 |
|---|---:|---:|---:|---:|---|
| hf-0081 | 40.32% | 33.00% | 72.18% | 52.63% | 개선 |
| hf-0014 | 49.71% | 39.77% | 92.31% | 76.15% | 개선 |
| hf-0003 | 83.74% | 80.95% | 98.08% | 93.27% | 개선 |
| hf-0094 | 79.90% | 96.48% | 100.00% | 123.71% | 악화 |
| hf-0035 | 61.47% | 61.47% | 84.21% | 84.21% | 동일 |

CER 기준 **3장 개선, 1장 악화, 1장 동일**이다. 삽입이 많으면 WER는 100%를 넘을 수 있으며, 결과를 100%로 잘라 표시하지 않았다.

### 확인된 악화 사례

`hf-0094`의 `ocr-text-0004`, `ocr-text-0005`, `ocr-text-0006`에서 Qwen은 각 줄의 개별 텍스트 대신 같은 3줄 묶음을 각각 반환했다. 같은 문구가 반복되어 문자 삽입 오류가 **39 → 147**로 증가했다. 따라서 평균 개선만으로 모든 표본에 Qwen 적용이 유리하다고 판단할 수 없다. 이번 점수 계산 중 반복 제거·후처리로 예측을 변경하지 않았다.

### 높은 오류율과 누락

`hf-0003`의 정규화 정답은 609자이지만 Paddle 출력은 151자뿐이다. 문자 삭제 오류는 **466 → 464**로 여전히 크다. 현재 교정 경로는 기존 OCR id·줄 위치를 유지하므로 OCR에 없는 내용을 새 줄로 복구하지 못한다. 이 점수는 텍스트 인식·누락·읽기 순서·표기 차이를 함께 반영하며, 검출된 글자만의 인식 정확도는 아니다.

## 평가 범위 주의

**제공된 전사와 기존 전체 페이지 출력을 그대로 비교한 예비 점수**다. 전사에서 빠진 영역을 모델 출력에서도 제외하는 마스크를 적용한 점수는 아니다.

- `hf-0003`: 화살표와 오른쪽 포스트잇은 전사에서 제외했고 곱하기 기호를 X로 작성했다. 화살표 연결 정확도는 평가하지 않으며, 포스트잇 글자가 예측에 있으면 삽입 오류가 될 수 있다.
- `hf-0081`: 줄 사이 괄호 내용은 전사에서 제외했다. 이 내용이 예측에 있으면 삽입 오류가 될 수 있다.
- `hf-0094`: 분수는 `/`로 표현하고 반복부호 `″`를 유지했다.
- NFC 정규화·연속 공백 축약·앞뒤 공백 제거만 적용한다. 목록 기호, 문장부호, X/×, 부등호, 대소문자, 전사 철자는 자동으로 통일하지 않는다.
- 5장은 예비평가이며, 정답 작성의 독립성·검수 과정은 별도로 감사하지 않았다. 전체 122장 또는 다른 문서에서의 개선을 입증한 결과가 아니다.

다음 평가에서는 제외할 영역과 전사 표기 규칙을 먼저 고정하고, 다른 개발 표본에서 교정 규칙을 개선한 뒤 새로운 평가 표본으로 재검증하는 것이 좋다.

## 결과 확인과 재현

- HTML: `data/benchmark/qwen_ablation_pilot/index.html`, 로컬 서버 http://127.0.0.1:8013/
- 표본별 수치: `results/comparison.csv`
- 치환·삭제·삽입 수: `results/sample_scores.json`
- 집계: `results/summary.json`, `results/REPORT.md`
- 오류율 그래프: `results/accuracy_comparison.png`·`.svg`

```bash
.venv/bin/python -m src.evaluation.ablation score \
  --manifest data/benchmark/qwen_ablation_pilot/manifest.json \
  --results data/benchmark/qwen_ablation_pilot/results/results.json \
  --references data/benchmark/qwen_ablation_pilot/references.json

.venv/bin/python -m src.evaluation.ablation_analysis \
  --results data/benchmark/qwen_ablation_pilot/results/results.json

.venv/bin/python -m src.evaluation.ablation_html \
  --manifest data/benchmark/qwen_ablation_pilot/manifest.json \
  --results data/benchmark/qwen_ablation_pilot/results/results.json \
  --references data/benchmark/qwen_ablation_pilot/references.json
```

HTML 상단은 정답 기반 micro CER·WER를, 각 표본은 개별 CER를 보여준다. **정답 전사** 버튼에서 작성자·원문·주석·표본 오류율을 확인할 수 있다. 줄별 빨강·초록 강조는 여전히 두 모델 사이의 출력 차이이며, 정답과 문자별로 정렬한 정오답 표시가 아니다.
