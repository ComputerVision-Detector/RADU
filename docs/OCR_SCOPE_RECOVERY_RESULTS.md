# 기존 LoRA 고정: CTC 범위 필터 복원과 GT 재검토 실험

실행일: 2026-11-10, Asia/Seoul. 앞선 페이지 감사의 1순위인 **GT 보완·평가 범위/필터 계약 정리**를 실제 실행했다. 기존17페이지 진단 집합을 재사용했으며 독립 test가 아니다. 새 학습·모델 탐색·임계값 조정·운영 배포·원격 라벨 수정은 없다.

## 1. 결론

**CTC 문자 사전 때문에 정상 텍스트가 검출 후 제거되던 문제를 확인했고, 같은 검출기로 제외된103개를 정확히 복원했다.** 기존397개의 ID·bbox·crop hash·인식문은 모두 그대로다. 인식 모델을 개선한 실험이 아니라 고정 인식 결과의 평가 범위/포함 경로를 바로잡은 진단이다.

기존 GT v2를 고정하면 페이지 CER은 **30.0615% → 28.9025%**, 절대 **1.1589%p**, 상대 **3.8551%** 감소했다. 삭제는395자 줄었지만 삽입은341자 늘었다. 페이지 bootstrap95% 개선 구간은 **−8.0551~+10.1283%p**로0을 포함한다. 따라서 안정적인 전체 CER 개선을 입증했다고 말하지 않는다.

검출 대상 누락의 주요 원인이 검출기 학습 부족이 아니라 **인식기와 맞지 않는 CTC 평가 범위 필터**였음은 확인됐다. 그러나 필터를 해제하면 수식·행렬과 일반 텍스트가 섞이는 문제가 남는다. 현재 우선 투자는 독립 GT 검수와 혼합 영역 geometry/범위의 정리다. 기존 LoRA 추가 학습이나 신규 검출기 학습은 실행하지 않았다.

## 2. 고정 조건과 산출물 보존

- 선택 adapter: `data/experiments/ocr_v2_vl_lora/train/best-adapter`.
- `adapter_model.safetensors` SHA256: `ab1bfcfba5c0711ebfc82e83432eca13b47fbb48ca0800d9e7b4d590e15421e5`.
- 공식 base/processor/tokenizer: 기존 로컬 PaddleOCR-VL1.6, 파일별 hash 유지.
- 인식: BF16/SDPA, batch4, left padding, `OCR:`, greedy/beams1, max_new_tokens128, min/max pixels112896/200704. 활성 adapter는 `default` 하나다.
- 검출: 같은 로컬 `PP-OCRv6_medium_det`, 당시 `inference.yml` 기본값. box_thresh0.45, thresh0.2, unclip_ratio1.4를 변경하지 않았다. 모델 파일 모두 당시 detector receipt와 hash를 대조했다.
- GT v2와 기존397개 예측을 실험 시작에 고정했다. 새 GT v3 검토·동결도 추가103개 인식 전에 끝냈다.
- 새 루트: `data/experiments/ocr_page_audit_scope_recovery_v1/`. 이전 감사·학습 산출물은 덮어쓰지 않았다. 운영 경로를 변경하지 않았다.

RTX4090 24GB에서 실행했으며 시작 시 GPU에는 다른 학습이 없었다. 검출은 기존 전용 Paddle 환경(Paddle3.2.1/PaddleX3.7.2/Python3.10), 인식은 기존 LoRA 환경(PyTorch2.14.1+cu126/Transformers5.22.1/PEFT0.21.2/Python3.11)을 사용했다. 의존성 추가 설치나 유료 자원 사용은 없다.

## 3. 검출500개 복원과 batch 통제

동일17개 원본에서 같은 검출기를 실제 다시 실행했다. GT를 보거나 문자 사전을 적용해 검출 결과를 제거하지 않고500개 전부를 저장했다.

| 검증 항목 | 실제 결과 |
|---|---:|
| raw 검출 수 | 500 |
| 이전 저장 예측 | 397 |
| 정확히 복원된 추가 영역 | 103 |
| 기존397 ID/bbox/crop hash 일치 | 397/397 |
| 기존397 인식 문자열 유지 | 397/397 |
| 새103 crop 인식 실패 | 0 |
| optimizer step | 0 |

새103개만 기존 page/order 순서로 batch4 인식했다. 기존397개는 원래 batch4 캐시를 그대로 사용했다. 두 비교 조건에서 공유 인식문이 같으므로, 500개를 자연 순서로 새 batch에 재배치해 기존 출력까지 바뀌는 혼입을 피했다. 출력 페이지 조립은 원래 검출 order로 수행했다. 이는 기존 캐시397과 추가103을 각각 묶는 진단용 고정 schedule이며, 자연 순서500개 batch 실행과 동일하다고 주장하지 않는다.

기존 첫4개 batch를2회 다시 실행해 tensor hash와 출력 일치, 기존 캐시와 일치, adapter 활성화를 확인했다. GPU 인식 입력은 **추가103+반복8=111 crop**이었다. 인식 phase 전체8.334초, 추가103 인식5.999초, 검출17페이지12.819초였다. 검출 시간은 모델 생성 이후이고, 인식 phase 시간은 로딩·검증을 포함한다. 속도 개선 비교는 아니다.

## 4. 복원103개 전수 확인

103개 모두9 contact sheet에서 AI가 시각 확인했다. 추가 인식문을 보고 GT를 채우거나 포함 기준을 고르지 않았다. 독립 사람 검수는0건이다. 재현 시에는 frozen visual decision의 ID와 crop hash가 모두 일치해야 기존 시각 판정을 재사용한다.

| 시각 유형 | 전체 | GT v2에서 채점 유지 | 일대일 매칭 |
|---|---:|---:|---:|
| 일반 본문·코드 라벨 | 34 | 32 | 28 |
| 수식·행렬 | 60 | 10 | 0 |
| 단독 화살표·중괄호 | 7 | 0 | 0 |
| 덮어쓴 제목 | 1 | 0 | 0 |
| 덮어쓴 제목의 분리 글자 | 1 | 0 | 0 |
| 합계 | **103** | **42** | **28** |

이전과 같은 평가-only ignore를 유지했다. 예측 면적50% 이상이 ignore에 들어가고 유효 GT와는 전혀 교차하지 않을 때만 제외한다. 추가103개는 ignore61 / keep31 / mixed_keep11개다. 새 결과가 나쁘다는 이유로 임계값을 바꾸거나 mixed 영역을 통째로 버리지 않았다.

수식10개가 작은 유효 GT 교차 때문에 유지돼 인식문222자가 채점에 유입됐다. 이는 Levenshtein 삽입341자의 정확한 기여량과는 다르다. 전체 문자열 alignment와 주변 영역의 영향도 있어 문자 길이를 삽입 수로 치환하지 않았다.

[복원103 전수 화면](../data/experiments/ocr_page_audit_scope_recovery_v1/restored103-gallery.html), `restored103-audit.csv/json`, 원본 crop과9장 sheet를 보존했다.

## 5. 1차 비교: 기존 GT v2 고정

두 조건은 **17페이지·306 GT·정답5,695문자**로 동일하다. 모델·인식문·GT·정규화·ignore·읽기 순서 규칙을 유지하고 추가103개 검출 영역의 포함 경로만 달리했다. 기본 IoU0.5 maximum-cardinality 일대일 매칭과 NFC·공백 병합을 사용했다.

| 지표 | 기존397개 bank | 복원500개 bank | 변화 |
|---|---:|---:|---:|
| 실제 채점 예측 | 304 | 346 | +42 |
| TP / FP / FN | 271 / 33 / 35 | 299 / 47 / 7 | +28 / +14 / −28 |
| Precision | 89.1447% | 86.4162% | −2.7286%p |
| Recall | 88.5621% | 97.7124% | +9.1503%p |
| F1 | 88.8525% | 91.7178% | +2.8653%p |
| 페이지 micro CER | 30.0615% | **28.9025%** | −1.1589%p |
| 공백 제외 CER | 31.2230% | 29.3783% | −1.8447%p |
| 치환 | 540 | 528 | −12 |
| 삭제 | 542 | 147 | −395 |
| 삽입 | 630 | 971 | +341 |
| 편집 오류 / 문자 | 1,712 / 5,695 | 1,646 / 5,695 | −66 |
| 페이지 완전 일치 | 0/17 | 0/17 | 동일 |

GT가 평가 ignore에만 사용되므로 raw500개를 출력하는 구현에는 GT 의존 필터가 없다. 다만 이 지표는 정답의 ignore 범위를 아는 오프라인 평가이며, 배포 시 수식을 자동 구분하는 기능을 구현한 것은 아니다.

새 GT의 일부 bbox는 앞선 검출 도움을 받은 geometry다. 독립적인 검출 recall 벤치마크라는 한계는 이번 복원으로 해결되지 않았다. 검출 가중치가 바뀌지 않았으므로 ‘검출기 학습 성능 개선’이라고 표현하지 않는다.

### CTC가 버린 정상 텍스트

GT v2의 정상 텍스트32개가 CTC 준비 단계에서 사전 미지원 문자 때문에 제외돼 있었다. 실제 미지원 표시는 `├`11개, `└`6개, `─`2개, `◦`12개, `中`1개다. 한국어·영어 본문 자체보다 연결선·불릿·인용 표기가 전체 줄 제외를 유발했다. VL tokenizer와 다른 CTC 사전의 제한을 VL 페이지 경로에 재사용하면 안 된다는 구체적 근거다.

32개 중28개가 복원 검출과 기본 IoU 일대일 매칭됐다. 나머지4개도 cos/sin/RoPE처럼 짧은 텍스트 bbox와 넓은 기존 GT, 혹은 한 줄의 분할·병합 문제여서 완전 미검출로 확정하지 않는다. 남은 FN7개 전체 ID와 bbox 대응은 요약 JSON에 있다.

### 페이지별 변화와 불확실성

17페이지 중6개는 오류 감소,3개 증가,8개 동일했다. 큰 변화는 hf-0008 −179오류, hf-0029 −45오류인 반면 hf-0087 +174오류, hf-0033 +53오류였다. 일반 코드 라벨의 복원과 수식/행렬의 유입이 함께 나타났다.

페이지 paired bootstrap2,000회/seed42의 CER 개선량95% 구간은−8.0551~+10.1283%p다.0을 포함하므로 페이지 집합 전체에 안정적인 개선이 있다고 단정하지 않는다. 이미 반복 사용한 진단17페이지의 불확실성 추정이며 독립 일반화 구간은 아니다.

## 6. 별도 GT v3: 확대 원본37개 재검토

앞선13개 FP 보류, 잘린 웹사이트1개, 원본 GT 보류23개를 확대 원본에서 다시 확인했다. 모든37개를 확인했으며 모델 전사로 채우지 않았다.

- 원본 GT13개를 포함으로 복원했다.9개는 이미지 근거에 따라 전사를 새 버전에 명시적으로 수정했고4개는 원문 그대로 복원했다.
- 예: `Taken IDs`→`Token IDs`, `악한`→`약한`, `홈럭`→`출력`, `노릅`→`노름`. 이미지에서 구별되는 글자만 반영했고 원격 V2 원문은 수정하지 않았다.
- 포스트잇 두 줄과3차원 합성곱 제목/최소평균제곱오차 제목은 이전 취소선·불확실 판정을 확대 원본에서 정정해 원문 그대로 복원했다. 포스트잇은 전체 문맥 crop도 남겼다.
- 오른쪽 원본 경계에서 잘린 웹사이트1개는 보이지 않는 접미사를 추정하지 않고 보류로 바꿨다.
- 나머지23개는 여전히 보류다.13개 FP 보류도 확실한 전사로 해결하지 못해 그대로 남겼다.

포함 GT는306→318개, 페이지 정답 문자는5,695→5,982자다. geometry·order는 바꾸지 않았다. 이 자료도 **AI 재검토 초안**이며 독립 사람 검수 완료가 아니다.

| 별도 GT v3 / 17페이지·318GT·5,982문자 | 기존397개 bank | 복원500개 bank |
|---|---:|---:|
| 채점 예측 | 309 | 351 |
| TP / FP / FN | 282 / 27 / 36 | 311 / 40 / 7 |
| Precision / Recall / F1 | 91.2621 / 88.6792 / 89.9522% | 88.6040 / 97.7987 / 92.9746% |
| 페이지 CER | 28.0843% | 25.0084% |
| 공백 제외 CER | 29.2518% | 25.6851% |
| S / D / I | 517 / 631 / 532 | 541 / 159 / 796 |

v3 내부의 bank 복원 차이는3.0759%p다. v2의30.0615%와 v3의25.0084%를 직접 비교해 모델 개선이라 부르면 안 된다. GT 전사·포함·분모가 달라졌기 때문이다. v3 내부 bootstrap 개선 구간도−6.2846~+11.4828%p로0을 포함한다. v3 결과가 더 낮다는 이유로 v2 1차 비교를 버리거나 평가 기준을 사후 선택하지 않았다.

사람 검수용 [37항목 화면](../data/experiments/ocr_page_audit_scope_recovery_v1/human-review.html)과 미검수 template도 만들었다. 기본값은 모두 미검수이고, 사람이 직접 전사·근거를 입력해야 JSON으로 저장된다. 파일을 만들었다는 사실을 사람 검수 완료로 집계하지 않았다.

## 7. 코드·검증·재현

- `src/evaluation/ocr_scope_recovery.py`: 준비/고정, 원본 검출 전량 저장, 기존397 identity 검사, GT 버전 적용, 추가103 인식과 bank 조립. detector/recognition 경로에서 GT로 영역을 제거하지 않는다.
- `src/evaluation/ocr_scope_recovery_report.py`: 공유 인식문 불변 검사, 동일 GT의 paired 범위 비교, bootstrap·페이지별 결과·103 전수 CSV/HTML.
- `configs/ocr_page_audit/scope_review_v3.json`: 확대 원본37개 판정 원문.
- `configs/ocr_page_audit/scope_restored_visual.json`:103개 시각 유형과 ID/crop hash가 결합된 frozen 판정.
- `tests/test_ocr_scope_recovery.py`:6개 새 테스트. 기존54개와 합계 **60개 회귀 테스트 통과**.

전체 프로젝트 unittest discovery도 실행했으나164개 집계 중7개 모듈 import 오류가 남았다. `test_hybrid`, `test_hybrid_vector`, `test_image_upload`는 fastapi, `test_ocr_first_analysis`, `test_structure_rules`, `test_text_box_candidates`는 cv2, `test_relation_analysis`는 defusedxml이 전용 LoRA 환경에 없다. 전체 프로젝트 통과라고 주장하지 않으며 관련60개 회귀 통과와 구분한다.

검출 geometry/crop drift, ID 누락/중복, 추가 인식 coverage, 기존 전사 변경 거절, GT 근거 없는 수정 거절, GT 입력 불변, paired 분모·오탐 삽입, review-freeze drift를 확인했다. 독립 코드 검토에서도 실제397개 불변·103개 추가·두 GT 점수·입력 해시를 재확인했다. 코드 검토자는103개 이미지를 독립 사람이 다시 검수한 것은 아니다.

코드 검토 후 review-freeze를 명시적으로 검사하는 guard를 보강했다. 실제 실행 당시 파일도 freeze와 일치함을 재확인했고 수치 경로를 변경하지 않았다. 실제 인식 실행 소스는 당시 기록된 SHA256과 정확히 일치하는 사본으로 보존했다.

```bash
# 새 위치에서 전체 재현. 기존 출력은 덮어쓰기 거절.
export RADU_OCR_SCOPE_OUTPUT=/absolute/path/to/new-scope-run
.venv/bin/python -m src.evaluation.ocr_scope_recovery --stage prepare
.venv/bin/python -m src.evaluation.ocr_scope_recovery --stage review-gt
.cache/venvs/ocr-v2/bin/python -m src.evaluation.ocr_scope_recovery --stage detect
.venv/bin/python -m src.evaluation.ocr_scope_recovery --stage restored-review
.cache/venvs/ocr-vl-lora/bin/python -m src.evaluation.ocr_scope_recovery --stage recognize
.venv/bin/python -m src.evaluation.ocr_scope_recovery_report

.cache/venvs/ocr-vl-lora/bin/python -m unittest \
  tests.test_ocr_scope_recovery tests.test_ocr_page_audit tests.test_ocr_clean_vision \
  tests.test_ocr_train_audit tests.test_ocr_lora_round2 tests.test_ocr_vl_lora \
  tests.test_ocr_v2 tests.test_ocr_finetune tests.test_ocr_finetune_page -v
```

재현은 frozen AI 판정과 정확히 같은 이미지 hash에서 그 판정을 다시 적용하는 것이다. 새로운 독립 검수를 자동 수행하는 명령이 아니다. 기존GT/Paddle/LoRA 로컬 자료와 같은 GPU 실행 환경이 필요하다.

## 8. 현재 적용 판단과 다음 대상

**확정한 원인:** CTC 문자 사전의 연결선·불릿 등 미지원 때문에 정상 줄32개가 범위에서 빠졌다. raw detector는 그중28개를 기본 IoU로 매칭할 수 있었다. 검출기 학습이 이 문제를 해결하는 첫 수단은 아니다.

**남은 문제:** 혼합 수식/일반 텍스트의 GT bbox 교차, source GT의 넓거나 분할된 geometry, 판독 불확실 영역과 독립 사람 검수 부재다. 필터 복원은 삭제를 줄이지만 삽입 증가 때문에 전체 CER 개선은 작고 불확실하다. LoRA 자체가 잘못 읽는 잔여 오류도 있어 범위 복원만으로 페이지 품질이 충분해지지 않았다.

다음 우선 대상은 **독립 사람 GT 검수와 일반 텍스트/수식 경계의 검증**, 이어 고정 validation에서 crop 경계·혼합 영역 처리의 비교다. 이번17페이지에서 좋은 점수가 나오게 ignore 임계값을 바꾸거나 새 모델을 학습하지 않았다. 운영 반영은 하지 않았고, 이번 산출물은 로컬 진단과 후속 검증의 근거로 유지한다.

[정량 비교 CSV](ocr_scope_recovery/comparisons.csv), [전체 결과 JSON](ocr_scope_recovery/results.json), [실제 회귀 테스트 로그](ocr_scope_recovery/tests-regression.log)에 근거를 보존했다. 모델·입력·설정·환경·실행 소스·최종 산출물 hash는 새 실험 루트의 `completion-manifest.json`에 기록한다.
