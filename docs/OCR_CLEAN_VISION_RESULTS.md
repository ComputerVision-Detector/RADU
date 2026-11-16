# 정제 train의 언어부·시각부 LoRA 실제 비교

이번 라운드는 train 품질 감사 후 같은 정제 데이터에서 **기존 언어 LoRA의 추가 학습**과 **기존 언어 LoRA를 동결한 시각부 LoRA 학습**을 비교한다. 기존 V2 라벨·분할·원본·모델·운영 Paddle/Qwen 경로는 보존했다. 모델 선택에는 고정 validation만 사용하고, 이전에 사용한 test는 탐색적 확인으로 취급한다.

## train 감사와 정제의 실제 범위

기존 train 후보2,513개 /111페이지 전부의 crop 해시·이미지 읽기·크기와 geometry-derived 위험 표시를 검사했다. 모든 페이지를 포함하도록 페이지별 위험 상위2개와 추가 위험 표본을 골라 **260개 crop /17개 원본 픽셀 contact sheet**를 AI가 시각적으로 점검했다. 이는 train 전사 전체의 독립 전문가 검증이 아니다. 본문에서 '정제'는 **보수적으로 선별한 후보**를 뜻하며, 남은 전사가 모두 정확하다는 인증이 아니다.

자동 표시 수는 여러 ink band251, 긴 전사에 비해 세로로 긴 crop132, 경계 ink143, 긴 전사35, 고립 자모3개였다. 중복 가능하며 표시만으로 제거하지 않았다. 단일 정상 손글씨도 band 여러 개로 탐지되는 등 false flag가 많았다. **OCR 예측·모델 정답률을 보고 train을 골라내지 않았다.** 실제 이미지에서 확인하거나 판독 불확실하다고 기록한 사례를 보류했다.

| 보류 사유 | crop |
|---|---:|
| hf-0160 흐름도·진리표 페이지 전체, 별도 구조/기호 과제로 재검토 | 114 |
| hf-0135 미적분 페이지 전체, text 후보 유형 재검토 | 11 |
| hf-0130 수식 중심 페이지 전체, 유형 재검토 | 3 |
| crop에 이웃 줄이 보이며 재박스 필요 | 26 |
| 전사 불일치·글자 불확실, 재검수 필요 | 6 |
| 수식·도식·단독 기호 등 일반 텍스트 여부 불확실 | 8 |
| 경계 잘림으로 전사 범위 불확실 | 2 |
| 취소선·덮어쓰기 처리 기준 재검토 | 2 |
| 합계 | **172** |

세 페이지를 통째로 보류한128개와 개별44영역을 합친 것이다. 보류가 발생한 페이지는40개이며, 그중3개는 전체 보류, 나머지는 문제 crop만 제외했다. 학습은 독립 GT crop 단위이므로 같은 페이지의 다른 정상 후보까지 일괄 삭제하지 않았다. 평가 페이지 GT를 부분 정답으로 바꾼 작업은 아니다. 전사 문자열·좌표를 임의로 수정하거나 OCR로 채우지 않았고, 원본 `pilot-v2`와 GitHub V2는 변경하지 않았다.

train은 **2,341 crop /108페이지 /NFC 정답47,164문자**로 줄었다. 원래48,878문자 중1,714문자를 보류했다. 한글 포함·영문 없는 후보1,090→1,070, 혼합668→657, 나머지 영문·숫자 등755→614였다. 삭제가 대부분 truth-table의 짧은 기호/숫자에 집중돼 언어 구성도 달라졌다. 이 변화가 순수 라벨 품질 효과와 별개로 영향을 줄 수 있다.

감사 코드: `src/ocr_data/train_audit.py`. 재현 가능한 판정 원문·해시·원본 crop 링크는 `data/processed/ocr_v2/train-audit-round3/{audit.json,review-index.json,decisions.json,clean-train.json,held-train.json,clean-manifest.json}`에 있다. contact sheet는 같은 디렉터리의 `panel-00.jpg`~`panel-16.jpg`다. 리뷰되지 않은 나머지 전사의 잡음과 잘못된 유형이 남을 수 있다.

## 공통 조건과 비교 대상

validation은 앞선 라운드의 **244 crop /14페이지 /5,613문자**를 그대로 사용한다. hf-0019/hf-0095의53 crop은 기존대로 제외하고, 전체297개 결과는 별도 진단용으로 저장한다. 이전 LoRA의 같은244개 CER **12.7917%**가 유지 후보 기준선이다. 새로운 train 정제를 이유로 validation이나 test를 유리하게 바꾸지 않았다.

두 조건은 같은 이전 선택 언어 LoRA에서 시작한다. 이 부모는 앞서 정제 전 train을 학습했으므로 이번 실험은 그 가중치에서의 **정제 후 추가 적응**이다. 잘못된 학습의 영향을 전혀 받지 않은 clean-only 학습이라고 주장하지 않는다.

| 조건 | 학습하는 부분 | 동결하는 부분 |
|---|---|---|
| 언어부 | 기존18 layer q/k/v/o LoRA,1,253,376파라미터 | 기본 LLM·전체 시각 인코더·projector |
| 시각부 | 마지막4개 시각 block(23~26) q/v의 새 rank8 LoRA,147,456파라미터 | 기존 언어 LoRA·기본 LLM·기본 시각 가중치·projector |

학습 파라미터 수가 같지 않으므로 모든 언어/시각 튜닝 전략의 우열 실험이 아니다. 지정된 보수적인 두 전략의 비교다. 같은 clean train, seed42, AdamW lr2e-5, weight_decay0.01, batch2/accumulation4, clip1,3epoch, 증강 없음, image112896~200704 pixels, max sequence768, `OCR:`, greedy/생성128, NFC 정규화를 유지했다. 문자·정답을 자르지 않는다.

각 epoch는293 optimizer step,3epoch는879 step이다. 정제 전 라운드의945 step보다 적으므로 전사 정제의 **독립적인 인과 효과**를 분리한 비교는 아니다. 이번 언어/시각 조건끼리는 같은 데이터·순서·step 수를 쓴다. 부모 모델도 후보로 두고 validation CER이 나아지지 않으면 유지한다.

## 실제 학습·동결·재로딩 검증

시각부를 켜면서 부모 언어 어댑터가 꺼지는 문제를 막기 위해, PEFT underlying LoraModel에서 `['language','vision']`을 동시 활성화한다. `requires_grad`는 `.vision.` LoRA에만 허용한다. 시각 학습 중 기존 language model을 `eval()`로 두어 부모 LoRA의 dropout도 동결하되, loss gradient는 language model을 거쳐 시각부로 흐르게 한다. projector는 학습하지 않는다.

실제 GPU smoke에서 zero-initialized vision LoRA를 켜도 부모의 validation4 crop 출력이 같았다. train8개에 실제 optimizer step을 실행해 의도한 LoRA의 비영 변화·유한 loss/gradient를 확인했다. 학습 전후 **모든 비학습 파라미터의 byte hash**를 비교했고 동일했다. 시각 smoke에서는 부모 언어 LoRA를 포함한906,855,024파라미터가 그대로였다. 저장한 named adapter를 새 base 객체에 재로딩해 생성이 같은지도 확인했다.

시각 checkpoint는 공식 base에 단독으로 적용하면 안 된다. **같은 부모 언어 어댑터와 새 시각 어댑터를 함께 활성화**해야 한다. 원본 base나 부모에 merge해 덮어쓰지 않았다. 선택·추론 도구는 이 구성을 복원한다. `optimizer.pt`는 저장하지만, 현재 CLI의 학습 재개 entrypoint는 제공하지 않는다.

## 재현·임의 crop 추론

```bash
# 감사 산출물은 새 output 경로에서 생성한다.
.venv/bin/python -m src.ocr_data.train_audit
.venv/bin/python -m src.ocr_data.train_audit \
  --decisions data/processed/ocr_v2/train-audit-round3/decisions.json

# 기존 출력과 다른 output을 설정한 뒤 전체 실제 실험
.cache/venvs/ocr-vl-lora/bin/python -m src.training.ocr_clean_vision \
  --config configs/ocr_clean_vision.json --stage all

# 선택된 구성 또는 --arm vision / --arm language 를 명시해 추론
.cache/venvs/ocr-vl-lora/bin/python -m src.cv.ocr_clean_vision_infer \
  --image /absolute/path/to/crop.png \
  --output data/experiments/ocr_v2_vl_lora_round3_demo

.cache/venvs/ocr-vl-lora/bin/python -m unittest tests.test_ocr_clean_vision \
  tests.test_ocr_train_audit tests.test_ocr_lora_round2 tests.test_ocr_vl_lora \
  tests.test_ocr_v2 tests.test_ocr_finetune tests.test_ocr_finetune_page -v
.venv/bin/python -m src.evaluation.summarize_ocr_clean_vision
```

감사 결정은 재실행 시 제공된 `decisions.json`을 복사해 사용한다. 판정 없이 후보 검출만 재실행한 결과를 자동 정답 검수로 해석하지 않는다. 기존 출력 디렉터리는 덮어쓰지 않는다. 실행 환경은 전용 `.cache/venvs/ocr-vl-lora`(기존 train runtime read-only 공유+별도 rapidfuzz), PyTorch2.14.1+cu126/Transformers5.22.1/PEFT0.21.2/RTX4090다. 패키지 변경·공식 가중치 추가 다운로드·원격 쓰기·유료 자원은 없었다.

## 실제 결과: 추가 CER 개선 없음

각 arm에서 정제 train2,341개 전체를 3epoch·879optimizer step 학습했다(총1,758step). validation244개·14페이지·5,613문자의 micro CER로 선택했으며, 동률에서는 기존 부모를 유지했다.

| 구성 | 선택 epoch | CER | 공백 제외 CER | 완전 일치율 | 부모 대비 CER 차이 |
|---|---:|---:|---:|---:|---:|
| 기존 부모 LoRA | 기존 | 12.7917% | 13.0444% | 21.3115% | — |
| 정제 언어 LoRA | 1 | 12.7917% | 13.1533% | 23.7705% | 0.0000%p |
| 정제 시각 LoRA | 2 | 12.8630% | 13.1315% | 22.5410% | +0.0713%p |

부모는718오류, 시각부는722오류로 차이는4문자다. 언어부 완전 일치는52→58개로 늘었지만 CER은 같고 공백 제외 CER은 악화했다. 따라서 CER 개선이라고 주장할 수 없다. 언어부 train loss는0.4252→0.3672로 줄었으나 validation CER은12.79→13.02%로 상승했다. 학습 loss 감소가 목표 지표 개선으로 이어지지 않았다. 전체297 validation의 언어부 최저 CER14.1588%, 시각부14.2899%는 보조 진단이며 선택에 사용하지 않았다.

두 arm 모두 유한 loss, 실제 가중치 갱신, 모든 비학습 파라미터 hash 불변, 새 base에 adapter bundle 재로딩 후4개 출력 일치를 통과했다. 최종 epoch의 초기 대비 최대 weight 변화는 언어0.0061440, 시각0.00950395였다. 최대 GPU 할당량은 각각3,633,872,384/3,584,155,136bytes였다. 실제 checkpoint와 named export는 `data/experiments/ocr_v2_vl_lora_round3/{language,vision}/best/`에 보존했다. 원본 모델을 덮어쓰지 않았다.

선택을 고정한 뒤 test235개·17페이지·4,401문자를 평가했다. 선택 결과가 동일 부모이므로 Before/After는 모두 CER15.2238%, 공백 제외15.5129%, 완전 일치19.5745%(46/235), 치환447·삭제102·삽입121로 동일하다. 상대 CER 감소율0%다. 같은 예측의 paired bootstrap 차이 구간[0,0]은 동일 모델 비교의 결과이며 새 방법의 유효성 증거가 아니다. 선택된 부모의 실제 test 추론은 기존 동일 조건 예측과 일치해 그 예측을 재사용했다고 기록했다. 탈락 arm의 test 성능으로 재선택하지 않았다.

고정 검출397영역/199GT·17페이지 전체 OCR은 실제 다시 추론했다. 이전 선택 모델과 동일하게 페이지 CER84.7647%(3,405오류/4,017문자), 공백 제외87.8511%였다. 검출TP196/FP201/FN3, Precision49.3703%/Recall98.4925%/F165.7718%로 변화가 없다. 추론24.745초, 실패0건이며 이는 저장된 검출 crop의 인식 실행 시간으로 검출·모델 로딩을 포함하지 않는다. 누락 GT와 다량 삽입이 페이지 지표에 영향을 주므로 GTcrop CER과 혼동하지 않는다. 정상 no_text 페이지가 이 test에 없으므로 no_text 오탐0건을 그 성능의 검증으로 해석하지 않는다.

## 오류 사례와 해석

validation의 저장된 batch4 평가에서는 시각부가 `conv2`→`CONV2`, `layer1`→`layer 1`을 개선했지만, 맞던 `Temperature ≠ 0`를 `Temper ature ≠ 0`로 악화시켰다. 전체 사례·페이지/언어/길이별 지표·페이지 paired bootstrap은 [validation-comparisons.json](ocr_clean_vision/validation-comparisons.json)에 있다. 새 CLI로 해당 CONV2 crop을 단독 재로딩 추론했을 때는 부모·시각부 모두 `conv2`였다. 두 adapter 동시 활성화는 확인했으며, batch 수치 경로에 따른 출력 차이 가능성이 있어 단독 demo를 batch 평가와 같은 결과라고 주장하지 않는다. 최종 비교 지표는 동일 batch4 조건으로 고정했다.

이번 결과는 보수적 데이터 제외와 시각부 마지막4block q/v LoRA만으로 추가 CER 개선을 입증하지 못했다는 뜻이다. 모든 시각 학습 방식이 실패한다는 결론은 아니다. 정제 언어부는 완전 일치에 일부 이점이 있지만 CER 기준 교체 근거는 부족하다. 부모가 이미 제외 전2,513개로 학습됐으므로 이번은 정제 데이터로 재학습한 신규 모델이 아니라 정제 데이터 추가 학습이다. 이전보다 step 수도945→879로 달라졌고 숫자·표 라벨 제외 비중이 높아 데이터 정제의 단독 인과 효과를 분리하지 못했다. 남은 정답 잡음, 작고 편향된 표본, 현재 적응 범위의 한계는 가능한 원인이며 확정 원인으로 단정하지 않는다.

필자 정보가 없어 독립 필자 일반화가 보장되지 않으며, test는 이전 라운드부터 재사용한 탐색용이다. 새 미사용 최종 test 성능으로 표현하지 않는다. 전체2,513개 구조·hash 검사와111페이지를 대표하는260crop 시각 검토를 수행했지만, 남은 모든 전사를 사람이 검수한 데이터라고 부르지 않는다.

## 검증 산출물

회귀 테스트 **41개 통과**([실제 로그](ocr_clean_vision/tests.log)). train-only 제외·prediction 정답 혼입 방지·한글 NFC·S/D/I·좌표/해시·분할 누수·bbox 일대일 매칭·빈 결과/실패·LoRA 동시 활성화/동결·설정 drift를 포함한다. 학습 전후 bundle 재로딩 일치는 실제 GPU smoke와 본 학습에서 추가 검증했다. 검토 후 effective config drift guard를 보강했으며 실제 실행 설정은 시작 receipt와 동일함을 확인했다.

- [전체 실제 요약과 환경·hash](ocr_clean_vision/summary.json)
- [epoch별 validation CSV](ocr_clean_vision/validation.csv), [동일 선택 모델 test 비교 CSV](ocr_clean_vision/test.csv)
- [정제 manifest](ocr_clean_vision/clean-manifest.json), [보류 결정](ocr_clean_vision/decisions.json)
- [동시 활성화 단독 추론 결과](ocr_clean_vision/demo-vision.json)

운영 적용 판단은 기존 부모 유지다. 이번 정제 언어·시각 checkpoint는 재현·오류 분석용 후보로 보존하며 운영 경로를 변경하지 않았다.
