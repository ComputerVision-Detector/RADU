# 고정 LoRA의 페이지 GT·apparent FP·crop·읽기 순서 감사

실행일: 2026-11-10, Asia/Seoul. 기존17페이지 진단 집합을 사용했다. 새 독립 test나 일반화 성능이 아니다. **새 학습·모델 탐색·임계값 탐색·운영 변경 없이 실제 감사·재채점·필요 crop 추론을 수행했다.**

## 1. 판단

현재 높은 페이지 CER에는 **GT 누락과 CTC 평가 범위의 잔존 필터**가 크게 섞여 있다. apparent FP201개를 검출기의 비텍스트 오탐201개로 해석하면 안 된다. 이 집합에서는 GT 범위·품질을 먼저 보완하고 평가/필터 계약을 일치시키는 것이 우선이다. crop 경계는 대응 가능한 동일 영역에서 인식을 일부 악화시켰다. 순서만 바꾼 공통 영역의 차이는 작았지만, 대응 불명확한 영역까지 순서 문제가 작다는 결론은 낼 수 없다.

감사한 새 GT는 **AI 시각 검토본**이다. 독립 사람 검수0건이며, 새 bbox가 기존 검출에 도움받아 작성돼 독립적인 검출 재현율의 정답이 아니다. 결과는 사람이 인증한 완전 페이지 OCR 성능으로 발표할 수 없다. 원본 전체를 살폈지만 미검출·미전사 영역이 더 없다는 독립적인 완전성 보증은 하지 않는다.

## 2. 기존 상태와 입력 고정

현재 브랜치는 `feat/radu-lora-training`이다. 저장소 및 상위 경로에서 적용할 AGENTS.md가 발견되지 않았다. 기존 수정·미추적 실험·체크포인트를 보존했고 Git checkout/reset/push를 하지 않았다. 지정된 종합 기록과 각 실험 보고서, 실제 데이터/추론/평가 코드를 대조했다.

| 고정 요소 | 조건 |
|---|---|
| base | `models/vision_language/PaddleOCR-VL-1.6` |
| adapter | `data/experiments/ocr_v2_vl_lora/train/best-adapter` |
| adapter weight 파일 SHA256 | `ab1bfcfba5c0711ebfc82e83432eca13b47fbb48ca0800d9e7b4d590e15421e5` |
| 해시 의미 | 디렉터리가 아닌 `adapter_model.safetensors` 파일의 SHA256 |
| processor/tokenizer | 공식 base의 로컬 파일; 전체 파일 hash 고정 |
| dtype / attention | bfloat16 / SDPA |
| 생성 | `OCR:`, left padding, greedy, beams1, max_new_tokens128 |
| 이미지 예산 / 길이 | min_pixels112896 / max_pixels200704 / max_sequence768 |
| 실제 평가 batch | 4; 마지막 잔여 batch만 실제 잔여 수 |
| 활성 adapter | 기존 `default` 하나, 모든 parameter requires_grad=False |
| 학습 | optimizer 생성·step 모두0 |

base·adapter·processor·tokenizer·설정·기존 예측·원본17장·397검출 crop을 `input-lock.json`에 고정했다. 실행 전후 hash drift 검사를 통과했다. 모델 파일을 다시 다운로드하거나 다른 adapter를 선택하지 않았다.

환경: RTX4090 24GB / driver580.178.04 / Python3.11.15 / PyTorch2.14.1+cu126 / Transformers5.22.1 / PEFT0.21.2. 기존 전용 LoRA venv를 사용했다. GPU 확인 당시 다른 학습 프로세스가 없었고 기존 프로세스를 종료하지 않았다.

## 3. legacy 재현

당시 `pages_for()`의 CTC 통과199 GT, 저장된397개 인식문, IoU0.5 maximum-cardinality 일대일 매칭, NFC·공백 병합·대소문자/문장부호 유지, 기존 순서 조립을 그대로 사용했다.

| 항목 | 기록값 | 재현값 |
|---|---:|---:|
| GT / 예측 / 페이지 | 199 / 397 / 17 | 동일 |
| TP / apparent FP / FN | 196 / 201 / 3 | 동일 |
| Precision / Recall / F1 | 49.3703 / 98.4925 / 65.7718% | 동일 |
| 페이지 CER | 84.7647% | 동일 |
| S / D / I | 400 / 90 / 2,915 | 동일 |
| 정답 문자 | 4,017 | 동일 |
| 읽기 순서 역전 | 20 | 동일 |
| GT crop235 CER | 15.2238% | 동일: 670오류 / 4,401문자 |

재현 불일치0건이다. 기존235 crop와 페이지199 GT의 범위는 다르므로 두 CER 차이를 검출·순서의 기여량으로 계산하지 않았다. legacy20역전의 페이지 분포는 hf-0013:18, hf-0033:1, hf-0094:1이다. legacy 매칭196영역 내 가능한 페이지별 쌍 합은1,565개다.

**중요: 최초 검출은500개였고 legacy GT 기반 ignore에서103개를 제거한 후397개만 저장했다.** 제거103개의 bbox와 인식문은 해당 캐시에 없다. 이번에는 같은397개를 고정했으며 검출을 새로 실행하지 않았다. 따라서 이후 재현율은 raw500 검출기 자체의 새 평가가 아니라 **기존 범위 필터를 통과한397개 파이프라인의 조건부 진단**이다.

## 4. 페이지 GT 감사와 버전

17개 원본 전체를 확인하고, 모든 FP201개 crop을17 contact sheet에서 각각 확인했다. 이미지 전사 중 모델 출력은 정답으로 복사하지 않았다. 기존 V2 전사는 임의 교정하지 않고, 명백한 불일치·취소선·불확실 사례는 원문을 유지한 보류 항목으로 남겼다.

평가 규칙은 읽을 수 있는 일반 텍스트·코드/도식의 텍스트 라벨·인쇄된 노트 머리말/꼬리말 포함이다. 수식, tensor-only shape, 표 숫자, 단독 화살표/기호, 취소/판독 불확실 영역은 별도 ignore다. 수식과 섞인 문장은 이번 일반 텍스트 점수로 억지 전사하지 않았다. 순수 텍스트를 잘 읽었는지를 보고 포함 여부를 고르지 않았다.

| 범위 구성 | 영역 수 |
|---|---:|
| 17페이지 원본 V2 전체 영역 | 286 |
| legacy CTC GT | 199 |
| legacy 포함 중 새 감사에서 보류 | 19 |
| CTC에서 제외됐던 정상 텍스트 중 복원 포함 | 32 |
| 이미지에서 보완한 일반 텍스트 GT | 94 |
| 최종 포함: 199−19+32+94 | **306** |
| 포함 원본 V2 / 새 추가 GT | 212 / 94 |
| ignore·불확실 등 별도 영역 | 173 |
| 전체 새 GT 레코드 | 479 |

각 레코드에는 image_id/annotation_id/bbox/text/order, 유형, include, ignore·불확실 사유, 원본 GT 링크, 변경 근거, AI 검수 상태를 기록했다. 원본 전사 변경0건이다. 새 geometry는 검출 bbox를 시각 확인한 detector-assisted geometry다. crop 비교 M에서는 이 새 geometry를 제외했다.

- `audited-gt-v1.json`: 원본+추가/보류, 단순 공간 순서의 1차 감사본.
- `audited-gt-v2.json`: 같은 행의 좌우 라벨 두 쌍을 원본에서 확인해 순서만 수정한 최종 감사본. hf-0033의 열/행 개수, hf-0094의 Date/Page다. bbox·전사·포함 집합·인식문은 동일하다.
- `order-policy-revision.json`: y좌표만으로 정답 순서라 부를 수 없는 문제와 변경 근거.
- `summary/gt-full-change-trace.json`: 원본 대비 모든 order·legacy 포함 변화·추가를 기록한409개 변경 레코드. 초기 `gt-changes.json`은 전체 변경 장부가 아니므로 이 파일을 사용한다.

v2를 점수가 좋아지는 순서를 탐색해 선택하지 않았다. 실제로 P0 CER은 v1의29.9737%에서 v2의30.0615%로 소폭 상승했다. 기존 결과를 삭제하거나 맞춰 쓰지 않았다. 흐름도 전체의 의미적 읽기 순서는 아직 확정하지 못해 provisional 상태다.

ignore 정책은 **예측 면적의50% 이상이 ignore bbox 안에 있고, 유효 GT와는 교차하지 않을 때만 제외**다. 유효 GT와 조금이라도 교차하면 mixed로 유지한다. 최종93개가 제외되고26개 mixed 예측이 유지됐다. 범위가 섞인 예측 전체를 손쉽게 버려 점수를 높이지 않았다. bbox 기준이므로 실제 글자 단위 ignore mask와 같지는 않다.

## 5. apparent FP201 전수 분류

주 분류는 예측마다 하나이고 최종 합계201이다. confidence는 기존 캐시에 없어 null이며 새로 만들어 넣지 않았다. 원본/새 GT 연결, bbox, 예측문, 매칭, 사유, 불확실성, 시각 근거와 검수 상태를 CSV/JSON에 기록했다.

| 최종 주 분류 | 개수 | 해석 |
|---|---:|---|
| 정상 텍스트의 legacy GT 누락 | **94** | 인쇄된 노트 텍스트14개 포함. 기존 V2와 IoU>.1 연결도 없는 범위 누락 |
| 수식·단독 기호·표 숫자·취소선 등 제외 영역 | **82** | 일반 텍스트 평가의 대상과 다름 |
| 분할·병합/IoU 기준 | **11** | 부분 반복 후보3개 포함 |
| 확정된 독립 전체 영역 중복 | **0** | 부분 반복을 확정 중복으로 과장하지 않음 |
| 실제 비텍스트 오탐 | **1** | 읽을 수 있는 본문이 없는 낙서 |
| 독립적인 crop 경계 오류를 주 원인으로 확정 | **0** | 이웃 줄 유입은 보조 태그10개; 주로 범위 누락과 함께 발생 |
| 판단 보류·재검수 필요 | **13** | 어려운 전사·덮어쓰기·경계의 독립 판단 미완료 |
| 합계 | **201** | 모두 AI 검토, 사람 확정 아님 |

1차에는 부분 중첩3개를 duplicate로 기록했지만, NO.와 큰 머리말 crop, int[]와 큰 코드 crop처럼 추가 텍스트도 포함했다. 전체 영역의 중복임을 입증하지 못해 최종 v2에서는 분할·병합으로 분류했다. 변경 기록과1차 CSV도 보존했다.

FP201의 정규화된 예측문 길이 합은2,637자이고 legacy Levenshtein 삽입은2,915자다. 서로 다른 개념으로 취급했다. 인식문 길이가 곧 편집 삽입 오류의 기여량은 아니다.

[전수 확인 화면](../data/experiments/ocr_page_audit_v1/review.html)에서201개 모두와 원본/새 GT overlay를 확인할 수 있다. 최종 파일은 `fp201-audit-v2.csv/json`이다.

## 6. A: 동일397개에서 평가 범위·ignore 영향

| 조건 | GT | 채점 예측 | TP/FP/FN | P/R/F1 | 페이지 CER | S/D/I | 정답 문자 |
|---|---:|---:|---|---|---:|---|---:|
| legacy 재현 | 199 | 397 | 196/201/3 | 49.3703/98.4925/65.7718% | 84.7647% | 400/90/2915 | 4,017 |
| 새 GT v2, 추가 ignore 미적용 | 306 | 397 | 271/126/35 | 68.2620/88.5621/77.0982% | 46.9008% | 668/428/1575 | 5,695 |
| 새 GT v2, 명시한 ignore 적용 P0 | 306 | 304 | 271/33/35 | 89.1447/88.5621/88.8525% | **30.0615%** | 540/542/630 | 5,695 |

legacy에서 최종 감사 정책으로 CER은54.7033%p 달라졌다. 이는 **모델 개선이 아니라 GT·범위·순서/ignore 정책 변경**이다. 중간 단계 간 차이는 이 순서로 정책을 적용했을 때의 관찰이며 순수 원인별 기여량이 아니다.

새 FN35개 중32개는 기존 CTC 범위 밖이었던 정상 텍스트다. upstream에서 제거103개의 geometry가 없어32개를 모두 실제 검출 누락이라 할 수 없다. 나머지3개도 넓은 GT 대비 낮은 IoU, 한 줄 분할 등으로 일대일 매칭에 실패한 사례여서 FN=완전 미검출이라고 단정하지 않는다.

새 GT94개의 bbox가 검출 도움을 받았으므로 P/R/F1 상승에는 정답 구축 방식의 편향이 있다. raw500에서 새 GT에 따른 참 검출 정확도를 검증한 결과가 아니다.

## 7. B: crop 경계, 동일 집합 M

M은 원본 V2 GT 중 IoU≥.5이고, IoU>.1 이웃이 양쪽에서 각각 하나인 유일 대응만 포함했다. 새 detector-assisted GT, ignore와 교차하는 예측, 분할·병합·불명확 대응을 제외했다. 예측문 정오로 M을 고르지 않았다.

**16페이지·138영역·정답2,501문자**로 고정했다. C0/C1의 region ID·정답·batch4 멤버와 순서는 같고 입력 crop만 다르다. P1용 다른 batch 결과나 이전 cached C1을 섞어 채점하지 않았다.

| 동일 M | C0 정답 bbox | C1 검출 bbox | C1−C0 |
|---|---:|---:|---:|
| micro CER | 13.8745% | 15.3139% | **+1.4394%p** |
| 공백 제외 CER | 14.5719% | 16.0741% | +1.5023%p |
| 영역 완전 일치 | 23.9130% | 18.8406% | −5.0724%p |
| S / D / I | 240 / 49 / 58 | 263 / 62 / 58 | +23 / +13 / 0 |
| 오류 | 347 | 383 | +36 |

C1에서21영역 개선·37영역 악화·80영역 오류 수 동일이었다. 검출 crop이 평균적으로 불리했으나 모두 악화하지는 않았다. 이 결과는 매칭 가능한 영역에서 crop 경계·resize 결과가 바뀐 영향이며, 전체 페이지 성능이나 검출 누락까지 포함하지 않는다.

오류 차이 상위 개선/악화12개를 양쪽 crop 이미지로 추가 확인했다. 여백·종횡비·미세 경계 변화가 많았고, 크게 잘린 글자/이웃 줄 유입만으로 모든 차이를 설명할 수 없었다. 예를 들어 hf-0052의 자연어 설명은 검출 crop에서7→2오류로 개선됐지만 hf-0013의 ‘기계학습에서 각 특징’은0→3오류였다. hf-0087 일부 GT crop에는 다음 줄 ink가 보였는데 이를 줄인 검출 crop에서도8→11오류로 악화했다. crop 크기·수치 경로의 상호작용 가능성은 있으며 확정 인과로 단정하지 않았다.

`crop-cases.json`과 `review/crop-compare-0..2.jpg`, `crop-visual-examples-index.json`에 정답·출력·bbox·사례를 보존했다.

## 8. C: 문자열·영역 고정, 순서만 변경

전체397개에 의미적 정답 순서를 강제하지 않았다. 대응이 명확한 동일 M138개에서 **기존 cached 검출 인식문을 그대로** 사용했다. unmatched·부분 반복·불확실·분할/병합·새 검출 도움 GT는 제외했다. 인식문 수정·불리한 예측 삭제·문자열 유사도 기반 순서 최적화는 하지 않았다.

| 지표 | O0 기존 순서 | O1 감사 GT 순서 |
|---|---:|---:|
| 페이지 / 영역 | 16 / 138 | 동일 |
| 정답 문자, 줄 사이 구분 공백 포함 | 2,623 | 동일 |
| 가능한 페이지 내 쌍 | 813 | 동일 |
| 역전 | 1 | 0 |
| 순서 오류율 | 0.1230% | 0% |
| 공통 영역 조립 CER | 14.3729% | 14.2204% |
| S / D / I | 257 / 62 / 58 | 253 / 62 / 58 |

순서만 바꾸면4편집 오류,0.1525%p 줄었다. 이 점수는 **M만의 조립 CER**이며 전체 페이지 CER이 아니다. legacy20역전과 다른 분모/GT 순서이므로 ‘20→1로 파이프라인을 개선했다’고 표현하지 않는다. 흐름도 등 의미적 순서가 불명확한 나머지 영역에 대한 순서 영향을 배제하지 않는다.

## 9. D: 실제 고정 파이프라인과 정답 영역 참조

| 같은 감사 GT v2 / 17페이지 | P0 고정 검출+기존 순서 | P1 정답 bbox+감사 순서 |
|---|---:|---:|
| 평가 GT | 306 | 306 |
| 페이지 정답 문자 | 5,695 | 5,695 |
| 페이지 CER | 30.0615% | **14.1001%** |
| 공백 제외 CER | 31.2230% | 15.0080% |
| S / D / I | 540 / 542 / 630 | 519 / 125 / 159 |
| 페이지 완전 일치 | 0/17 | 0/17 |

P1은 배포 가능한 파이프라인이 아닌 진단용 정답 영역 조건이다. 차이15.9614%p에는 기존 CTC 필터, 검출·범위·crop·분할/병합·순서·batch 구성 등이 함께 작용한다. 순수 검출 효과라고 하거나 crop/순서 차이와 더해 원인 기여율을 만들지 않았다. P0는 매칭 성공 영역만 채점한 점수가 아니며 누락·불필요 예측을 포함한다.

P1의 영역별 인식은306개·5,406문자 기준 CER15.0573%다. 페이지 조립에서는 공백 구분과 전역 alignment 때문에14.1001%로 다르다. 정답 bbox에서도 인식/잔여 GT 오류가 남는다.

## 10. 재현성·실행량

모든 기존397개는 저장된 예측으로 감사·재매칭·재채점했다. 신규 추론은 C0 138개+C1 138개+P1 306개+대표4개2회, **총590 crop 입력**뿐이다. 검출 재실행·전체 페이지 재추론·모델 탐색은 하지 않았다.

대표는 기존 검출 목록의 첫4개로 미리 고정했다. 반복 tensor hash(input_ids/attention_mask/이미지 tensor 등)와 출력이 같았고 기존 동일 batch 캐시도 재현했다. eval 모드·left padding·활성 adapter·strict base load를 검증했다. 추론 stage 전체는 로딩·검증 포함41.416초, C0 9.370초/C1 9.599초였고 추론 실패0건이다.

**batch4라는 크기가 같아도 묶인 이미지가 달라지면 출력이 달라질 수 있다.** 새 paired C1은 기존397 목록에서 뽑은 cached C1과138개 중20개 문자열이 달랐다. 동일 batch 반복은 일치했으므로 입력 묶음 변경에 민감한 수치/전처리 경로가 확인됐지만 정확한 내부 원인은 분리하지 못했다. BF16/SDPA 영향은 가능성이지 확정 원인이 아니다. 좋아 보이는 batch를 선택하지 않았다. C0/C1은 새 동일 쌍 구성만, O0/O1은 기존 cached 문자열만 사용했다. P0/P1은 batch 멤버가 달라 그 차이도 결합 진단의 한계다.

## 11. 검증 결과와 남은 검수

- 신규13개 감사 테스트와 기존41개 OCR 회귀 테스트, **합계54개 통과**.
- FP ID 정확히201개·중복/누락 거절, legacy/new GT 분리, 변경 추적, 유효 GT의 ignore 우선권, 일대일/분할 대응 분리, M 분모·NFC, 순서-only 불변성, 누락·빈 출력·실패 분모, 파일 drift·덮어쓰기 거절을 확인했다.
- 전용 LoRA 환경에서 프로젝트 전체 unittest discovery도 실행했으나158개 집계 중7개 모듈 import 오류가 있었다. `test_hybrid`, `test_hybrid_vector`, `test_image_upload`: fastapi 없음; `test_ocr_first_analysis`, `test_structure_rules`, `test_text_box_candidates`: cv2 없음; `test_relation_analysis`: defusedxml 없음. 감사 관련54개 통과와 구분한다.
- 주 저장소 .venv에서 전체 discovery를 추가 시도했으나 `test_hybrid.HybridTests.test_invalid_inputs_and_job_failures`에서 장시간 진행되지 않아 이번에 시작한 해당 테스트만 중단했다. 전체 프로젝트 통과라고 보고하지 않는다. 환경 패키지 설치/운영 변경은 하지 않았다.
- 독립 코드 검토에서 지적된 upstream103 조건부 평가와 불완전 변경 장부를 보고서/전체 change trace에 반영했다. 검토자는 원본201개를 독립 시각 재검수한 사람이 아니다.

미완료는 독립 사람 GT 검수,13개 FP 불확실 사례,부분 반복3개 의미 확정,흐름도 의미 순서,raw103개 복구와 독립 검출 recall,누락 없는 페이지 완전성 인증이다. 모델 개선이나 완전한 새 벤치마크 구축으로 포장하지 않는다.

## 12. 다음 투자 우선순위

1. **GT 보완·평가 범위/필터 계약 정리**: 일반 텍스트 누락94개와 CTC 제외 정상 텍스트32개가 확인됐다. 독립 사람이 전사·geometry·페이지 완전성을 검수하고, VL 평가에서 CTC 문자 사전 필터를 잔존시키지 않는 계약을 마련해야 한다. 이번에는 그 필터를 운영에서 수정하지 않았다.
2. **crop·분할/병합 품질과 batch 재현성**: M에서 검출 crop이36오류 더 많고 분할/병합11개가 관찰됐다. 경계/여백의 효과와 BF16 batch 민감성을 분리해 검증할 가치가 있다.
3. **읽기 순서**: 분명한 공통 영역에서는4편집 오류 차이지만, GT 순서 오류와 흐름도 순서 불확실을 먼저 해결해야 한다. 현재 결과만으로 신규 순서 모델의 투자 효과를 보장할 수 없다.
4. **중복 처리·검출기 학습은 근거 확보 후**: 확정 전체 중복0개, 비텍스트1개이며 FN 상당수가 이전 범위 필터와 연결됐다. ‘FP201개가 많다’는 이유만으로 새 검출기 학습이나 강한 NMS를 우선하지 않는다. 중복 후보3개에 대한 무조건 삭제도 정당화되지 않는다.

기존 LoRA의 추가 학습은 이번에 실행하지 않았고 현재 증거에서 우선 투자로 선택하지 않았다.

## 13. 산출물과 재현

새 실험 루트: `data/experiments/ocr_page_audit_v1/`. 기존 실험 파일은 덮어쓰지 않았다.

| 산출물 | 파일 |
|---|---|
| 입력·모델·설정 해시 | `input-lock.json`, `inference-lock.json` |
| legacy 재현 / FP ID 동결 | `legacy-reproduced.json`, `legacy-gt.json`, `frozen-fp201.json` |
| 새 GT와 변경 | `audited-gt-v1.json`, `audited-gt-v2.json`, `summary/gt-full-change-trace.json` |
| 최종 FP 감사 | `fp201-audit-v2.csv`, `fp201-audit-v2.json`, `review-revision-v2.json` |
| 전수 확인 화면 / 원본 overlay | `review.html`, `review/` |
| 페이지 범위·미확정 메모 | `coverage-by-page.json`, `page-review-notes.json` |
| M·제외 목록 / 원본 GT crop | `M.json`, `M-excluded.json`, `new-crops/` |
| 실제 추론 캐시·환경·재현 검사 | `inference/{C0,C1,P1,repeat1,repeat2,reproducibility,receipt}.json` |
| 최종 정량 결과·비교표 | `summary/results.json`, `summary/comparisons.csv` |
| 실제 테스트 로그 | `tests-audit.log`, `tests-regression.log`, `tests-project*.log` |

코드: `src/evaluation/ocr_page_audit.py`, `ocr_page_audit_infer.py`, `ocr_page_audit_report.py`. AI 시각 판정 원문: `configs/ocr_page_audit/visual_decisions.py` 및 v2 변경 JSON. 버전 수정 재현 코드는 `src/evaluation/ocr_page_audit_revision.py`다. 독립 사람이 판정한 것으로 재현되지 않으며 사람이 다른 판정을 하면 새 버전을 만들어야 한다.

```bash
# 실제 실행한 기본 단계. 기존 루트가 있으면 덮어쓰기 거절.
.venv/bin/python -m src.evaluation.ocr_page_audit --stage prepare
.venv/bin/python -m src.evaluation.ocr_page_audit --stage audit
.venv/bin/python -m src.evaluation.ocr_page_audit_revision
.venv/bin/python -m src.evaluation.ocr_page_audit_infer --stage prepare
.cache/venvs/ocr-vl-lora/bin/python -m src.evaluation.ocr_page_audit_infer --stage infer

# 저장된 최종 감사본/예측으로 새 위치에 재채점 요약
.venv/bin/python -m src.evaluation.ocr_page_audit_report --output /tmp/radu-page-audit-new-summary

# 회귀 테스트
.cache/venvs/ocr-vl-lora/bin/python -m unittest \
  tests.test_ocr_page_audit tests.test_ocr_clean_vision tests.test_ocr_train_audit \
  tests.test_ocr_lora_round2 tests.test_ocr_vl_lora tests.test_ocr_v2 \
  tests.test_ocr_finetune tests.test_ocr_finetune_page -v
```

새 실험을 처음부터 재현하려면 위 단계 전 `export RADU_PAGE_AUDIT_ROOT=/absolute/path/to/new-run`으로 새 루트를 지정한다. 기본값은 기존 실험 루트라 이미 존재하면 prepare가 중단한다. v2 revision CLI는 별도 새 경로에서 실제 재실행해 동일 GT pages·수치·FP 주 분류를 재현했다. 최종 v2의 기하/전사와 v1은 동일해 기존 P1 추론을 재사용하고 순서 조립만 바꿨다. 모든 결과의 source/manifest 해시는 최종 `completion-manifest.json`에 추가 기록한다.

[최종 비교 CSV](ocr_page_audit/comparisons.csv), [전체 정량 결과 JSON](ocr_page_audit/results.json), [회귀 테스트 로그](ocr_page_audit/tests-regression.log)를 함께 제공한다.
