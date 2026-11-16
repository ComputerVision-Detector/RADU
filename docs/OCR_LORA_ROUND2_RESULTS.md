# OCR LoRA 2차 CER 개선 실험

목적은 앞선 VL LoRA의 잔여 오류를 근거로 개선 가설을 만들고, 실제 추가 학습·추론으로 검증하는 것이다. 원본 V2 라벨, 기존 checkpoint, 운영 Paddle/Qwen 경로는 보존한다. test로 하이퍼파라미터를 선택하지 않으며, 앞선 라운드에서 사용한 test의 후속 결과는 계속 **탐색적 평가**로 표시한다.

## validation 진단과 데이터 정책

앞선 LoRA의 validation297 CER은14.1733%, 오류973개(치환678/삭제132/삽입163)였다. 공백 삽입65/삭제50을 제외하면 글자·기호 오류가 대부분이었다. 원래 crop 면적20만 픽셀 초과62개는379오류/2,538문자, 나머지235개는594오류/4,327문자였다. 이는 상관관계이며, 크기나 해상도가 원인이라고 확정하지 않았다.

오류가 큰 validation crop을 원본에서 육안 확인했다. `hf-0019-line-0005`에는 GT의 전사 오기와 이웃 줄 조각이 보였고, `hf-0095-line-0028`에는 GT에 없는 인쇄 URL과 이웃 줄 손글씨가 포함됐다. OCR로 정답을 채우거나 임의 수정하지 않았다. 두 페이지를 **이번 라운드의 재검수 보류**로 미리 고정했다. 영역 몇 개만 고르면 부분 GT가 페이지 전체처럼 취급될 수 있어 페이지 전체53 crop을 보류했다. 원본 `pilot-v2` 파일은 바꾸지 않고 별도 derived view로 관리한다.

주 validation은 **244 crop / 14페이지 / 정답5,613문자**다. 이전 모델도 같은244개에서 다시 채점해 **CER12.7917%, 공백 제외13.0444%, 완전 일치율21.3115%**를 기준선으로 삼았다. 이전의297개 CER14.1733%와 새244개 결과를 직접 비교해 개선이라고 말하면 안 된다. 원래297개 결과는 모든 조건에서 진단용으로 계속 저장하며 모델 선택에는 사용하지 않는다. 보류와 오류 증거는 `quarantine.json`, `quarantine-visual-evidence.jpg`에 있다. test는235 crop /17페이지 /4,401문자를 그대로 유지한다.

## 제한된 개선 가설

| 후보 | 출발점 | 변경 | 확인하려는 원인 |
|---|---|---|---|
| 이전 LoRA·기존 해상도 | 이전 선택 어댑터 | 없음 | 유지해야 할 기준선 |
| 이전 LoRA·높은 해상도 probe | 같은 어댑터 | 이미지 min/max 픽셀만2배 | 재학습 없는 해상도 효과 |
| A: 추가 학습 | 같은 어댑터 | lr2e-5로3 epoch 추가 | 부족한 적응인지 확인 |
| B: 높은 해상도 추가 학습 | 같은 어댑터 | A와 같고 픽셀 예산만2배 | 학습된 해상도 효과 |

A/B 모두 공식 base와 이전 LoRA rank8/alpha16/dropout0.05를 유지하고 기존 어댑터를 `is_trainable=True`로 로드한다. 새 LoRA를 중복 삽입하지 않는다. 동일한 seed42·train2,513개 전체·순서·AdamW 재초기화·batch2/accumulation4·학습률2e-5·3epoch·grad clip1을 사용한다. 이전 optimizer momentum을 이어받은 실험이 아니라 **가중치를 이어받고 optimizer를 공통으로 새로 시작한 실험**이다. train loss나 test로 선택하지 않는다.

A의 이미지 예산은 min112896/max200704 pixels, B는 min225792/max401408 pixels다. 가로·세로 길이를 각각2배로 키운다는 뜻은 아니다. 실제 processor 점검에서 동일 train crop의 image grid는1×8×82→1×12×116, 입력 token190→374로 바뀌었다. 긴 crop 예시는299→544 token, supervised token은41로 동일했다. 최대 sequence768과 생성128 token, `OCR:` 프롬프트·greedy decode·정규화는 유지했다. 라벨을 절단하거나 미지원 문자를 지우지 않는다. 시각 모델·projector·기본 LLM 가중치는 동결하고 언어 attention LoRA만 학습한다.

선택 규칙은 **주 validation micro CER 최솟값**으로 실행 전에 고정했다. 최초 네 후보에 아래의 제한된 보간 후보 하나를 추가해 총다섯 후보를 비교했다. 이전 모델도 후보라서 추가 학습이 악화되면 자동으로 이전 모델을 유지한다. 매 epoch는 전체297개의 생성 결과를 얻은 뒤 주244개와 진단297개를 각각 채점한다. secondary 공백 제외 CER·완전 일치율도 보고하지만 사후 선택 기준으로 바꾸지 않는다. 높은 해상도와 추가 학습 효과를 섞지 않도록 A와B를 함께 비교하고, 최종 test에서는 필요하면 부모 어댑터를 선택된 해상도에서도 추론해 가중치 효과를 따로 기록한다.

A가 기준선을 넘지 못한 validation 증거를 보고, test를 열기 전에 **부모와 A의 선택 가중치 변화를0.5:0.5로 합치는 후보 한 개**를 추가했다. 비율 탐색은 하지 않았으며 선언 시점·사유·test 미접근을 `adaptive-plan-extension.json`에 기록했다. 단순히 LoRA의A/B 행렬을 각각 평균하면 교차항이 생기므로 PEFT `combination_type=cat`으로 실제 delta 평균을 정확히 표현했다. 결과 rank는16이며, CPU 단위 시험에서 `.5*delta_parent + .5*delta_fitted`와 일치함을 검증했다. 추가 optimizer step은 없다. 이 후보까지 포함하면 validation에서 비교한 고유 점수는 부모2개·학습 epoch6개·보간1개다. 이런 validation 선택에 의한 낙관성과 기존 test 재사용 한계는 남는다.

## 재현과 파일

설정은 `configs/ocr_v2_vl_lora_round2.json`, 실행기는 `src/training/ocr_lora_round2.py`다. 기존 GPU 실행 환경 `.cache/venvs/ocr-vl-lora`를 사용한다. 원격 다운로드나 유료 자원은 필요하지 않았다. 새 출력 root를 지정해야 재실행할 수 있으며 기존 디렉터리는 덮어쓰지 않는다.

```bash
# 같은 설정의 output만 새 경로로 변경한 뒤 실행
.cache/venvs/ocr-vl-lora/bin/python -m src.training.ocr_lora_round2 \
  --config configs/ocr_v2_vl_lora_round2.json --stage all

.venv/bin/python -m unittest tests.test_ocr_lora_round2 tests.test_ocr_vl_lora \
  tests.test_ocr_v2 tests.test_ocr_finetune tests.test_ocr_finetune_page -v
.venv/bin/python -m src.evaluation.summarize_ocr_lora_round2
```

실제 대용량 산출물은 `data/experiments/ocr_v2_vl_lora_round2/`에 둔다. `start-receipt.json`은 실행 시작 시의 설정·소스·데이터·부모 어댑터 해시, `*.executed.py`는 시작 소스 사본이다. 학습 loop 소스는 A/B에서 동일하게 유지한다. 검토 후 선택/최종 입력 무결성 검사만 보강했으며 stage별 실제 실행기 소스를 `*.source.py`에 남긴다. 학습 가중치나 metric 계산을 변경하지 않았다.

`probe/`는 해상도만 바꾼 validation 생성, `smoke/`는 기존 어댑터의 실제 미소 학습·가중치 갱신·새 모델 객체 재로딩 검증이다. `continue/`, `highres/`는 각각 전체 train 학습의 epoch 예측·history·best/latest 어댑터·optimizer·검증 결과다. `selection-lock.json`은 test 전에 네 후보와 선택된 어댑터 가중치/config 해시를 고정한다. `final/`에는 이전 모델·선택 모델·동일 해상도 부모 비교와 고정 검출기 페이지 추론을 보존한다.

기존 shared trainer의 `selection.json` 설명에 full validation이라고 남는 legacy 문자열은 실제 주244개 점수와 맞지 않아 round 선택 단계에서 설명·표본 수만 바로잡고 원래 기록을 `selection-description-correction.json`에 보존한다. CER 값이나 선택 가중치를 사후 변경하는 작업이 아니다.

## 실제 실행 결과

**두 학습 조건 모두 train2,513개 전체로3 epoch·945 optimizer step을 완료했다.** 총1,890 step의 새 학습과 해상도 probe, 보간 추론을 실행했다. 학습 준비나 소량 smoke에서 멈춘 결과가 아니다. A/B 모두 finite loss·실제 가중치 갱신·선택 어댑터 저장 후 새 모델 객체의 validation4 crop 생성 일치를 확인했다. A의 부모 대비 adapter 최대 변화는0.00604568, B는0.00716800이다.

| validation 조건 | 주244 CER | 공백 제외 CER | 완전 일치율 | 진단297 CER |
|---|---:|---:|---:|---:|
| 이전 LoRA·기존 해상도 | **12.7917%** | **13.0444%** | 21.3115% | **14.1733%** |
| 이전 LoRA·픽셀 예산2배, 재학습 없음 | 13.7894% | 13.7848% | 20.9016% | 15.2804% |
| A·추가 학습 최선 epoch2 | 12.8095% | 13.1098% | **22.9508%** | 14.2753% |
| B·높은 해상도 학습 최선 epoch3 | 13.3262% | 13.1315% | 21.3115% | 14.5666% |
| 50:50 정확한delta 보간 | 12.9521% | 13.3275% | 22.1311% | 14.3190% |

주 CER의 정답 문자는 모두5,613자로 동일하다. 기준선718편집 오류, A719, B748, 보간727개다. A와 기준선의 차이는 **단1개 편집 오류(0.0178%p)**이므로 A가 의미 있게 나빠졌다고 단정하지 않는다. 반면 **CER 개선을 입증하지도 못했다.** A의 줄 완전 일치율이 높아도 처음 고정한 CER 선택 기준을 사후 변경하지 않았다. 보류 전 전체297개 진단에서도 새 후보들은 이전 모델14.1733%보다 나아지지 않았다.

| 새 학습 epoch | A 주 CER / 진단297 CER | B 주 CER / 진단297 CER |
|---|---|---|
| 1 | 13.7360 / 15.0182% | 13.5756 / 14.8725% |
| 2 | 12.8095 / 14.2753% | 13.4687 / 14.9745% |
| 3 | 12.9521 / 14.2462% | 13.3262 / 14.5666% |

A train loss는0.4175→0.3823→0.3667, B는0.4626→0.4228→0.4030으로 낮아졌지만 validation CER은 부모보다 개선되지 않았다. 관찰은 **추가 fitting과 validation 일반화가 일치하지 않았다**는 것이다. 과적합·라벨 잡음·입력 분포 변화 가능성은 있지만 이를 분리한 실험은 아니므로 확정 원인이라고 표현하지 않는다.

B는 재학습 없는 높은 해상도13.7894%보다13.3262%로 좋아졌으므로 높은 해상도에 적응하는 효과는 일부 관찰됐다. 그러나 원래 해상도의 부모12.7917%에는 못 미쳤다. 픽셀 증가가 새 원본 정보를 보장하지 않으며, 고정된 시각 인코더와 언어 LoRA가 달라진 token 분포를 처리하는 문제도 있어 해상도를 높이는 것만으로 개선된다고 볼 수 없다.

A의 학습+validation epoch 시간은103.7/102.6/102.3초, B는128.3/128.4/129.2초였다. 초기 모델 로딩·학습 전 평가·저장/재로딩은 별도다. GPU allocated memory peak는 A 약3.63GB / B 약4.98GB, OOM은 없었다. 기존 체크포인트나 프로세스를 종료·삭제하지 않았다.

validation에서는 개별 개선과 악화가 함께 있었다. A에서 `Logits[2,201088]`→정답 `Logits [2, 201088]`, `공간 과표를 변환`→정답 `공간 좌표를 변환`으로 좋아진 반면, 정답이던 `CONV1`이 `conv1`로 바뀌었고 `학습이 끝난`이 `학습이 곁난`으로 악화됐다. 일부 줄의 개선만 전체 CER 개선처럼 제시하지 않았다. 후보별 전체 paired validation 사례·페이지/언어/길이 지표는 [validation-comparisons.json](ocr_lora_round2/validation-comparisons.json)에 있다. 재사용 test의 신규 모델 오류 분석으로 후보를 다시 선택하지 않았다.

## 선택과 test 확인

**validation 기준으로 이전 LoRA·기존 해상도를 유지했다.** 선택은 test 추론 전에 `selection-lock.json`에 고정했고, 선택 가중치 SHA256은 이전과 같은 `ab1bfcfba5c0711ebfc82e83432eca13b47fbb48ca0800d9e7b4d590e15421e5`다. 채택되지 않은 A/B/보간을 test에 모두 실행해 우승자를 다시 고르지 않았다.

| 동일 탐색적 test235 | 이전 선택 LoRA | 이번 유지 모델 | 차이 |
|---|---:|---:|---:|
| CER | 15.2238% | 15.2238% | 0 |
| 공백 제외 CER | 15.5129% | 15.5129% | 0 |
| 완전 일치율 | 19.5745% | 19.5745% | 0 |
| 치환/삭제/삽입 | 447 / 102 / 121 | 동일 | 0 |

같은 모델을 재실행한 확인이므로 변화가 없는 것이 예상되는 결과다. CER 상대 감소율0%, paired bootstrap 개선량 구간[0,0]은 **동일 출력 확인**이며, 채택하지 않은 신규 모델들의 test 효과가0이라는 통계적 증거는 아니다. 이번 라운드에서는 test CER의 추가 개선을 확인하지 못했다. 이전의 최초 VL 원본 대비29.55% 개선을 이번 라운드의 새 성과로 다시 계산하지 않는다.

같은 원본에서 앞서 동결한 검출397 crop /199 GT 영역 /17페이지의 전체 OCR도 재실행했다. 페이지 CER은 **84.7647%**, 공백 제외87.8511%, 치환400/삭제90/삽입2,915로 이전과 동일했다. IoU0.5 일대일 매칭의 TP196/FP201/FN3, P49.3703/R98.4925/F1 65.7718%도 같고 읽기 순서 역전20건이다. 인식기 선택만 확인했으며 검출 개선은 주장하지 않는다. 최종 GT crop 추론 실패0/235, detector crop 실패0/397, 페이지 실패0/17이었다. 선택 모델 GT crop 처리16.992초, detector crop26.877초(모델 로딩 제외, 단일 실행)이며 기존 검출 실행11.700초는 별도다. 시간 차이를 속도 개선으로 해석하지 않았다.

최종 적용 판단은 **이전 LoRA 유지**다. 현재 후보를 더 오래 학습하거나 픽셀 예산만 올리는 방향의 우선순위는 낮아졌다. 다음 연구에서는 train의 전사·crop 완전성을 먼저 감사하고, 필자/노트가 분리된 새 검증 데이터를 확보한 뒤 필요한 시각 적응을 검토하는 편이 근거에 맞다. 이번 라운드 안에서 test를 보고 그 방향으로 재튜닝하지 않았다.

새 어댑터와 실제 학습·추론 로그는 그대로 남겼다. [요약 JSON](ocr_lora_round2/summary.json), [validation 전 조건 CSV](ocr_lora_round2/validation.csv), [test 확인 CSV](ocr_lora_round2/test.csv), [보류 이미지 근거](ocr_lora_round2/quarantine-visual-evidence.jpg)를 제공한다. 저장된 본 학습 어댑터는 로컬 `data/experiments/ocr_v2_vl_lora_round2/{continue,highres}/best-adapter/`, 보간 어댑터는 `halfstep/adapter/halfstep/`다. 운영 배포·원격 라벨 쓰기·push·유료 자원 사용은 없었다.

LoRA 전용 Python 환경에서 관련 단위·통합 테스트 **36개가 모두 통과**했다. assistant-only masking, quarantine 원문 보존, A/B 설정 차이 제한, checkpoint drift 거부, 실제 PEFT delta 평균, 기존 데이터·지표·bbox 매칭 검증을 포함한다. 결과는 [tests.log](ocr_lora_round2/tests.log)에 있다. A/B의 실행된 학습 loop는 시작 소스 사본과 SHA가 정확히 같았다. 종료 후 shared trainer의 설명 문자열과 표본 수 기록도 향후 실행에서 올바르게 남도록 고쳤으며, 이는 학습·평가 수치를 바꾸지 않는 metadata 수정이다.
