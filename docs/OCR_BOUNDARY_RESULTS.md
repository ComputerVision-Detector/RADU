# 고정 LoRA 후속 실험: 텍스트·수식 GT 경계와 이미지 기반 crop

실행일: 2026-11-10, Asia/Seoul. 앞선 제안에 따라 **원본 기반 GT 경계 재검토**와 **validation에서 고정한 이미지 기반 crop 처리의 실제 추론 비교**를 수행했다. 학습·새 모델·임계값 탐색·운영 변경·원격 라벨 수정은 없다.

## 1. 결과와 현재 판단

**이번 일괄 여백 축소 방법은 채택하지 않는다.** validation CER이12.9877%→13.7003%로 악화했고, 어두운 배경에서 문장 대부분을 잘라내는 실제 실패를 확인했다. validation 결과로 기존 crop 유지를 고정했으며, 이후 진단 결과로 재선택하지 않았다.

별도 GT 경계11개를 수정한 뒤 같은500개 예측을 재채점하면 페이지 CER은25.0084%→23.6041%로 달라졌다. 이는 **평가 GT geometry와 ignore 판정의 변화**이며 OCR 모델이나 배포 파이프라인의 개선이 아니다. 새 GT도 AI 검토본이고 독립 사람 검수는0건이다.

이번 근거는 ‘모든 crop 개선이 실패한다’가 아니라 **고정 밝기 임계값에 의존한 현재 방법은 사용하면 안 되고, 일부 mixed 판정은 GT 경계 오류였지만 나머지는 실제 영역 침범/복합 구조 문제**라는 것이다.

## 2. 고정 조건

| 요소 | 조건 |
|---|---|
| 선택 adapter | `data/experiments/ocr_v2_vl_lora/train/best-adapter` |
| adapter weight SHA256 | `ab1bfcfba5c0711ebfc82e83432eca13b47fbb48ca0800d9e7b4d590e15421e5` |
| base/processor/tokenizer | 기존 로컬 PaddleOCR-VL1.6, 파일 hash 유지 |
| 인식 | BF16 / SDPA / batch4 / left padding / `OCR:` / greedy / beams1 /128 new tokens |
| 이미지 예산 | 112896~200704 pixels, max_sequence768 |
| GT 경계 평가 bank | 앞선 복원500개 bbox·인식문 그대로 |
| 평가 규칙 | NFC·공백 병합, IoU0.5 최대수 일대일 매칭, 기존 ignore 정책 유지 |
| 실행 환경 | RTX4090 24GB / driver580.178.04, 기존 Python3.11·PyTorch2.14.1+cu126·Transformers5.22.1·PEFT0.21.2 |
| 학습/배포 | optimizer step0, 운영 경로 변경0 |

기존 변경·가중치·체크포인트·실험 산출물을 보존했다. 새 루트는 `data/experiments/ocr_page_audit_boundary_v1/`이다. 시작 시 GPU에 다른 학습이 없었고 기존 프로세스를 종료하지 않았다. 새 패키지·가중치 다운로드·유료 자원은 없다.

## 3. 단일 후보와 validation 고정

`configs/ocr_boundary.json`의 단일 방법을 실행 전에 정했다.

1. 원본 crop의 grayscale<160을 어두운 ink 후보로 본다.
2. 가로 최소120px 또는 crop 폭50% 이상, 세로 최소160px 또는 높이90% 이상인 연속 dark run을 bbox 추정 마스크에서 제거해 노트 선의 영향을 줄이려 했다.
3. 남은 마스크 bbox에3px 여백을 붙여 crop 안에서 자른다. 유효 마스크가 비면 원래 crop을 유지한다.
4. 마스크로 이미지 픽셀 자체를 지우지는 않는다. **남겨진 픽셀은 원본 그대로라는 뜻이지, 글자가 모두 보존된다는 보장은 아니다.**

처리 방법은 밝기/여백 sweep 없이 고정했다. validation14페이지를 대표하는21 crop을 실제 보았지만, 모든244개를 육안 보증하지는 않았다. 대표 표본에서 놓친 어두운 배경 실패가 전체 validation 추론에서 드러났다.

동일 validation244영역·14페이지·NFC 정답5,613문자로 원본과 처리 crop을 각각 추론했다. 두 조건의 region ID·정답·batch4 멤버·순서는 같다. CER이 엄격히 낮을 때만 후보를 채택하고, 동률/악화는 기존 crop 유지로 정했다. 진단17페이지의 결과로 이 규칙을 바꾸지 않았다.

## 4. 실제 validation 결과: 채택 실패

| 동일244영역 | 원본 crop | 여백 축소 후보 | 후보−원본 |
|---|---:|---:|---:|
| micro CER | **12.9877%** | 13.7003% | +0.7126%p |
| 공백 제외 CER | **13.1751%** | 14.0679% | +0.8929%p |
| 완전 일치율 | 20.9016% | 23.3607% | +2.4590%p |
| S / D / I | 505 / 101 / 123 | 507 / 159 / 103 | +2 / +58 / −20 |
| 오류 / 정답 문자 | 729 / 5,613 | 769 / 5,613 | +40오류 |

23영역 개선·22영역 악화·199영역 오류 수 동일이었다. 완전 일치율이 올랐지만 CER 선택 기준을 사후 바꾸지 않았다. 페이지 bootstrap2,000회/seed42의 개선량95% 구간은−2.6852~+0.8833%p로0을 포함한다. 일괄 적용의 이점을 입증하지 못했고, 아래 글자 잘림 실패도 있어 기존 crop을 유지했다.

과거 기록의 같은244개 CER12.7917%와 이번 원본12.9877%는 생성 batch 구성이 다르다. 과거에는297개를 생성한 후53개를 채점에서 제외했고, 이번에는244개 자체를 batch4로 묶었다.13개 문자열이 달랐다. 이 때문에 과거 캐시를 Before로 섞지 않고 이번 새 원본/처리 추론끼리 비교했다.

### 어두운 배경의 실제 실패

| validation 영역 | 원본 폭 | 남긴 구간 x | 남긴 폭 비율 | 원본→후보 오류 |
|---|---:|---|---:|---:|
| hf-0157-line-0026 | 1,209 | 933~1209 | 22.8% | 5→33 |
| hf-0157-line-0027 | 1,609 | 1005~1609 | 37.5% | 4→29 |

두 영역에서 잘린 앞부분의 grayscale 중앙값은122였고, 모든 픽셀이 임계값160 아래였다. 어두운 종이/그림자를 foreground mask로 잡은 뒤 긴 run을 노트 선처럼 제거해 bbox 추정에서 문장 앞부분을 잃었다. 원본/처리 사진에서도 대부분의 문장이 사라진다. 두 줄만 합계53오류가 늘었다.

이것은 단지 BF16 수치 차이라고 설명할 수 없는 **명백한 물리적 글자 잘림**이다. `systematic-debugging` 절차로 원본·추정 구간·밝기·실제 출력을 대조했다. 이번에는 결과를 보고 임계값을 바꾸거나 실패 표본을 빼서 다시 점수를 만들지 않았다. 실패한 후보로 보존하고 적용하지 않았다.

[실패/개선 사례 화면](../data/experiments/ocr_page_audit_boundary_v1/crop-errors.html), `dark-background-failure.json`, `review/errors-0.jpg`에 실제 근거를 남겼다. 도구 계약 테스트 통과는 이런 입력 품질 실패가 없다는 뜻이 아니다.

## 5. 사전 선언한 진단 M138 비교

이전 감사의 명확 대응 집합 M138개·16페이지·정답2,501문자를 그대로 사용했다. 여기의 입력은 **검출 bbox crop**이다. validation은 GT bbox crop이므로 선택 조건과 입력 분포가 완전히 같지는 않다.

후보는 이미 validation에서 탈락했지만 사전 계획에 따라 동일 방법의 진단 비교도 실행했다. 해당 결과로 채택 여부를 바꾸지 않았다. 새 모델·임계값·원문 수정은 없다.

| 동일138영역 | 원본 검출 crop | 여백 축소 후보 |
|---|---:|---:|
| micro CER | **15.3139%** | 15.5138% |
| 공백 제외 CER | **16.0741%** | 16.2744% |
| 완전 일치율 | 18.8406% | 18.8406% |
| S / D / I | 263 / 62 / 58 | 262 / 66 / 60 |
| 오류 / 정답 문자 | 383 / 2,501 | 388 / 2,501 |

10영역 개선·11영역 악화·117영역 오류 수 동일이다.5오류 차이로 통계적 악화를 크게 단정하지는 않지만 개선을 확인하지도 못했다. 페이지 bootstrap 개선 구간은−0.8303~+0.4144%p다. 기존과 동일한 M 원본 batch의138개 출력은 앞선 C1 캐시와 모두 일치했다.

validation에서는119 crop, 진단에서는56 crop의 경계가 바뀌었다. 픽셀이 그대로인 crop에서도 이웃 batch 입력 변화 때문에 문자열이 각각6개/4개 달라졌다. 같은 batch 크기만으로 수치 경로가 동일한 것은 아니다. 두 조건 모두 동일 row 구성으로 묶고 반복 batch는 일치함을 확인했으며, 변화에는 crop 및 batch 내 상호작용이 포함된다. 잘 나온 배치를 골라 쓰지 않았다.

선택된 처리는 **baseline**이므로 실제 채택 후 성능은 원본과 동일하다. 후보15.5138%를 채택 모델의 After 성능으로 제시하지 않는다. 이 M 결과는 매칭 가능한 영역의 인식 진단이며 전체 페이지 CER이 아니다. 탈락 방법으로500개 전체 페이지를 다시 추론하지 않았다.

## 6. 별도 GT 경계 감사와 v4

복원500개 예측 중 기존 정책상 mixed_keep인30개와 교차하는 유효 GT23개·7페이지를 감사 대상으로 고정했다. 문자 인식 정오로 대상을 고르지 않았다.23개 모두 원본/제안 crop을 시각 확인했다. 일부는 원래 GT가 옆 줄 ink·취소선·도식 연결선을 포함하고 있었고, 일부는 검출 crop이 실제 본문까지 침범해 여백만으로 분리할 수 없었다.

| GT 경계 검토 | 수량 |
|---|---:|
| 검토한 GT | 23 / 7페이지 |
| bbox 변경 승인 | 11 / 4페이지 |
| 그중 자동 여백 제안 | 5 |
| 그중 원본을 보고 수동 경계 지정 | 6 |
| 추가 축소 제안 보류 | 2 |
| 변경하지 않은 제안 | 10 |
| 독립 사람 검수 | 0 |

수동 경계는 본문 글자는 남기고 다른 줄의 ink, 취소된 오른쪽 단어, `for-each 문` 앞의 도식 연결선을 분리하도록 지정했다. 원본/수정 crop을 다시 나란히 확인했다. 이미 `NO.` 첫 줄이 일부 잘린 머리말2개는 더 줄이는 것을 보류했다. **이 두 보류는 제안만 미적용한다는 뜻이며, 기존 GT 포함 상태를 제외로 바꾸지는 않았다.**

GT v4는 전체479개 레코드의 ID·전사·include·order를 그대로 유지하고11개 bbox만 바꿨다. 유효318 GT·17페이지·페이지 정답5,982문자도 동일하다. 원격 V2는 수정하지 않았다. geometry origin이 검출 도움을 받았던 항목도 있으므로 독립 검출 정답이라는 한계가 남는다. 원래 provenance는 before/after trace와 원본 prediction 연결에 보존했다.

[GT23개 전체 확인 화면](../data/experiments/ocr_page_audit_boundary_v1/geometry-review.html), `geometry-review.json`, `geometry-changes.json`, `audited-gt-v4.json`을 제공한다. 이는 **AI geometry 재검토본**이며 사람 인증본이 아니다. inline 수식과 본문을 모든 줄에서 새로 분할·전사한 완전 GT도 아니다.

## 7. 동일500개 인식문에서 GT 경계만 바꾼 재채점

인식문·예측 bbox·순서·모델·정규화·ignore 기준은 그대로다. 예측 면적50% 이상이 ignore bbox에 있고 유효 GT와 전혀 겹치지 않을 때만 제외하는 기존 정책을 유지했다.

| 같은17페이지·318GT·5,982문자 | GT v3 | bbox 수정 GT v4 |
|---|---:|---:|
| raw 예측 | 500 | 500 |
| ignore 후 채점 예측 | 351 | 346 |
| mixed_keep 예측 | 30 | 25 |
| TP / FP / FN | 311 / 40 / 7 | 310 / 36 / 8 |
| Precision / Recall / F1 | 88.6040 / 97.7987 / 92.9746% | 89.5954 / 97.4843 / 93.3735% |
| 페이지 CER | 25.0084% | **23.6041%** |
| 공백 제외 CER | 25.6851% | 24.3149% |
| S / D / I | 541 / 159 / 796 | 541 / 160 / 711 |
| 오류 | 1,496 | 1,412 |
| 페이지 완전 일치 | 0/17 | 0/17 |

추가 ignore는 hf-0033의 det-0014/0018/0024, hf-0087의 det-0004, hf-0137의 det-0017로5개다. GT 경계를 바로잡으면서 취소선·행렬·불확실 영역이 옆 유효 GT와 거짓으로 겹치던 경우가 해소됐다. 반면 더 타이트한 GT 때문에 기본 IoU TP가1개 줄고 FN이1개 늘었다.

CER 차이는1.4042%p, 삽입85자 감소·삭제1자 증가다. **GT 기준이 달라진 재채점 결과이고, 실제 OCR가 새로 잘 읽은 것이 아니다.** 개별 제외 문자열 길이와 Levenshtein 삽입 감소를 같은 것으로 취급하지 않았다. 검출기 개선이나 배포 가능한 GT-free 혼합 영역 처리라고도 표현하지 않는다. 유효 본문과 실제로 겹치는 나머지 혼합 예측은 유지했다.

## 8. 검증·실행량과 한계

실제 인식은 validation244×2+반복8=496 crop, 진단138×2=276 crop, **합계772입력**이다. validation phase41.369초, 진단21.658초였다. 모델 로딩·검증을 포함한 단일 실행 시간이며 속도 개선 벤치마크가 아니다. 추론 실패0건, optimizer0step이다. GT 경계 비교에는 기존500개 캐시만 사용했다.

원본·처리 row ID/정답/분모·batch 구성이 같음을 검사했고, adapter/base/processor/config hash를 검증했다. 실제 실행 소스는 각 phase 시작 시 SHA256과 정확히 일치하는 사본을 보존했다. 입력/선택 guard를 보강한 후에도 당시 실제 row/설정/가중치가 일치함을 다시 확인했다.

- 신규8개+기존60개, **68개 관련 회귀 테스트 통과**.
- bbox 내부 축소·문장부호·원본 픽셀 비변경·빈/희미한 마스크 fallback·GT 범위/문자/순서 유지·ID/분모·엄격한 validation 선택·row drift 거절을 검사했다.
- 테스트가 모든 촬영 조건에서의 글자 보존을 보증하지는 않는다. 어두운 배경의 실제 품질 실패는 validation에서 탐지해 채택을 막았다.
- 독립 코드 검토에서 paired 지표, row lock, GT479개 불변성과11개 bbox 변경을 재확인했다. 검토자가 독립 사람으로23개 이미지를 재검수한 것은 아니다.
- 별도 새 경로에서 method lock, geometry revision, 저장 예측 분석·재채점·보고서 재생성을 실행해 모든 수치를 재현했다. GPU 추론을 불필요하게 반복하지 않았다.

전체 프로젝트 discovery는172개 집계 중7개 모듈 import 오류였다. `test_hybrid`, `test_hybrid_vector`, `test_image_upload`는 fastapi, `test_ocr_first_analysis`, `test_structure_rules`, `test_text_box_candidates`는 cv2, `test_relation_analysis`는 defusedxml이 전용 LoRA 환경에 없다. 실제 전체 로그를 보존했고 전체 프로젝트 통과로 표현하지 않는다.

기존17페이지는 반복 사용한 진단 집합이다. 독립 필자·미사용 test 일반화 성능이 아니다. GT 포함과 전사를 그대로 둔11개 경계 변경만으로 GT 완전성이나 판독 불확실을 해결하지 못했다. 독립 사람 검수는 별도로 필요하며 이번에 자동 완료했다고 표시하지 않았다.

## 9. 재현 코드와 명령

- `src/evaluation/ocr_boundary.py`: 고정 후보 crop 생성, method lock, paired 실제 추론, validation 선택, GT bbox-only revision.
- `src/evaluation/ocr_boundary_analysis.py`: 저장된 출력의 오류 사례·어두운 배경 실패 근거 재생성.
- `src/evaluation/ocr_boundary_report.py`: 동일 조건 지표·bootstrap·GT 재채점·전체 확인 HTML.
- `configs/ocr_boundary.json`: 실행 전 고정한 단일 후보.
- `configs/ocr_page_audit/boundary_validation_review.json`: 사전 validation 대표 검토의 crop hash.
- `configs/ocr_page_audit/boundary_geometry_v4.json`:23개 원본 검토 결정과 evidence hash.

```bash
# 기존 출력은 덮어쓰기 거절. 같은 로컬 모델/데이터가 필요하다.
export RADU_OCR_BOUNDARY_OUTPUT=/absolute/path/to/new-boundary-run
.venv/bin/python -m src.evaluation.ocr_boundary --stage prepare
.venv/bin/python -m src.evaluation.ocr_boundary --stage method-lock
.cache/venvs/ocr-vl-lora/bin/python -m src.evaluation.ocr_boundary --stage validation
.venv/bin/python -m src.evaluation.ocr_boundary --stage geometry
.cache/venvs/ocr-vl-lora/bin/python -m src.evaluation.ocr_boundary --stage diagnostic
.venv/bin/python -m src.evaluation.ocr_boundary_analysis
.venv/bin/python -m src.evaluation.ocr_boundary_report

.cache/venvs/ocr-vl-lora/bin/python -m unittest \
  tests.test_ocr_boundary tests.test_ocr_scope_recovery tests.test_ocr_page_audit \
  tests.test_ocr_clean_vision tests.test_ocr_train_audit tests.test_ocr_lora_round2 \
  tests.test_ocr_vl_lora tests.test_ocr_v2 tests.test_ocr_finetune \
  tests.test_ocr_finetune_page -v
```

검수 재현 명령은 동일 이미지 hash에서 저장된 AI 결정을 다시 적용하는 것이다. 새로운 독립 사람 검수를 수행하는 명령이 아니다. 최종 코드·입력·설정·원본·모델·예측·환경 hash는 `completion-manifest.json`에 기록한다.

## 10. 적용 판단

기존 LoRA와 기존 crop을 유지한다. 이번 고정 밝기 trim은 적용하지 않는다. 어두운 배경에서 글자를 잃었고 validation CER 개선을 보이지 않았기 때문이다. GT v4는 경계 진단·사람 검수용 초안으로 보존하며, 그 점수 하락을 모델 성과로 합산하지 않는다.

다음 작업에 앞서 확인할 것은 **배경 밝기·그림자에 대한 crop 글자 보존 조건**, **inline 수식/본문의 실제 분할 기준**, **독립 사람이 확인한 GT geometry**다. 이번 결과를 보고 진단 집합에 맞춰 밝기나 ignore 임계값을 재튜닝하지 않았다.

[전체 수치 JSON](ocr_boundary/results.json), [비교 CSV](ocr_boundary/comparisons.csv), [실제 테스트 로그](ocr_boundary/tests-regression.log)를 제공한다.
