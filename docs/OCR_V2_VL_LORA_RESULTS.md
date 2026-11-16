# PaddleOCR-VL 1.6 LoRA 실제 실행

이번 후속 실험은 공식 VL 1.6 원본과 **그 원본에서 학습한 LoRA 어댑터**를 비교한다. 앞선 PP-OCRv5 미세 조정과 다른 모델 계열이며, 그 실험의 test 결과가 LoRA 선택에 영향을 줬으므로 후속 test는 **탐색적 평가**다. 추가 독립 필자·노트 데이터 없이 최종 일반화 성능이라고 주장하지 않는다. 기존 운영 Paddle/Qwen 경로와 원본 가중치는 수정하지 않는다.

## 고정 조건과 실행 방식

- 같은 고정 V2 스냅샷 `50aca983da9885d9e7a965c1ef4b05055d47b252`, 같은 원본·분할·보류 페이지 정책을 재사용했다. `hf-0104`, `hf-0011` 등 재검수 보류는 유지한다. 수식·단독 기호는 제외하며 text 유형은 기존 휴리스틱 후보다. prediction은 정답에 사용하지 않는다.
- VL tokenizer가 CTC 사전 미지원 문자도 표현하므로 train **2,513** / validation **297** / test **235** crop을 사용한다. 앞선 CTC 실험의 2,348/273/199와 표본 수가 다르다. 문자 대체·라벨 절단 없이 모든 후보의 토큰 인코딩을 검사했고 unknown token은 0이었다. 최대 정답 토큰 수는 train68/validation70/test51이다.
- 모델은 로컬 `models/vision_language/PaddleOCR-VL-1.6`의 공식 가중치다. [공식 모델 카드의 Transformers 경로](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6)를 따라 native `AutoModelForImageTextToText`와 `AutoProcessor`로 로드했다. Transformers의 공식 weight-renaming 변환을 사용했고, missing/unexpected/mismatched/error는 모두 비어 있었다. 임의 초기화로 Before를 약화시키지 않았다.
- PyTorch2.14.1+cu126 / Transformers5.22.1 / PEFT0.21.2, BF16·SDPA·RTX4090. 전용 `.cache/venvs/ocr-vl-lora`는 기존 train 의존성을 읽기 전용 `.pth`로 공유하고 rapidfuzz3.14.5만 별도로 설치했다. 기존 train·Paddle 환경 패키지는 변경하지 않았다. 독립 패키지 전체 설치를 완료한 환경이라고 표현하지 않는다.
- 언어 모델 18개 layer의 q/k/v/o projection **72개**에만 LoRA rank8 / alpha16 / dropout0.05를 적용했다. 기본 가중치·시각 모델·projector·lm_head는 동결했다. 학습 파라미터 **1,253,376 / 906,855,024 (0.1382%)**다.
- AdamW lr5e-5 / weight_decay0.01 / batch2 / accumulation4 / grad clip1 / seed42 / 최대3epoch. 매 epoch 전체 validation **생성 결과의 micro CER**로 선택한다. teacher-forcing loss나 test CER로 선택하지 않는다. 이미지 증강은 없다.
- Before/After 모두 `OCR:` 프롬프트, 원본 crop, image min112896/max200704 pixels, 최대 sequence768, greedy/beam1, 최대 생성128을 사용했다. 라벨을 자르지 않고 예산 초과면 실패한다. 이미지·user·prompt·padding은 loss에서 마스킹하고 assistant 전사·EOS만 지도한다. 추론 시 left padding, 학습 시 right padding을 사용한다.
- 평가는 기존 `ocr_metrics.normalize`의 NFC·공백 병합·대소문자/기호 보존을 사용한다. LaTeX/문자 기호를 정답과 같게 임의 치환하지 않는다. 실패를 빈 문자열로 기록하고 분모를 보존한다. 결과 문자열을 보고 confidence threshold나 읽기 순서를 바꾸지 않는다.

## 재현과 추론

실행 코드: `src/training/ocr_vl_lora.py`, 설정: `configs/ocr_v2_vl_lora.json`. 각 stage는 기존 출력 디렉터리를 덮어쓰지 않는다. 재학습하려면 설정의 `output`을 새 경로로 바꾸고 동일 설정을 모든 stage에 사용한다.

```bash
.cache/venvs/ocr-vl-lora/bin/python -m src.training.ocr_vl_lora --stage smoke
.cache/venvs/ocr-vl-lora/bin/python -m src.training.ocr_vl_lora --stage train
.cache/venvs/ocr-vl-lora/bin/python -m src.training.ocr_vl_lora --stage final

# 원하는 줄·영역 crop 한 장에서 Before/After 실제 추론
.cache/venvs/ocr-vl-lora/bin/python -m src.cv.ocr_vl_lora_infer \
  --image /absolute/path/to/handwriting-crop.png \
  --output data/experiments/ocr_v2_vl_lora_demo

# 이전 실험에서 동결한 검출 crop을 그대로 공유한 페이지 비교
.cache/venvs/ocr-vl-lora/bin/python -m src.evaluation.ocr_vl_lora_pages

.venv/bin/python -m unittest tests.test_ocr_vl_lora tests.test_ocr_v2 tests.test_ocr_finetune tests.test_ocr_finetune_page -v
.venv/bin/python -m src.evaluation.summarize_ocr_vl_lora
```

어댑터는 `data/experiments/ocr_v2_vl_lora/train/best-adapter/`의 `adapter_model.safetensors`와 `adapter_config.json`이다. 전체 모델을 재초기화하거나 기존 원본에 merge하지 않았다. 추론은 동결된 공식 base + 해당 어댑터로 실행한다. `optimizer.pt`에는 optimizer/RNG를 저장했지만 현재 CLI에 학습 재개 entrypoint는 없다. 검증 범위는 **어댑터 저장·새 모델 객체로 재로딩·실제 생성 일치**이며 완전한 학습 재개를 검증했다고 주장하지 않는다.

학습 시작 전에 config·소스·모델 파일·데이터 manifest·train/validation 목록 해시를 저장했다. 당시 소스 SHA와 일치하는 `training_source.executed.py` 사본을 보존한다. 독립 코드 검토 후 생성 길이 제한 표시와 final 입력 해시 검사를 보강했으며 학습·디코딩 로직과 하이퍼파라미터는 바꾸지 않았다. 기존 학습 프로세스는 로드된 원래 코드를 계속 사용했다. 해당 코드의 batch padding 기반 생성 제한 표시에는 과대 집계 가능성이 있으므로 validation의 제한 플래그를 정량 결론에 사용하지 않는다. 최종 추론에서는 EOS 존재 여부를 검사하도록 수정했고 CER 문자열 자체에는 영향이 없다. 이번 실행의 어댑터 해시는 selection 시점이 아닌 **test 전 final lock**에서 최초 고정됐다. 후속 새 실행은 selection 시점에도 해시를 저장한다.

## 실제 전체 학습과 GT crop 최종 결과

**train 2,513개 전체로 3 epoch / 945 optimizer step의 실제 학습과 test 추론을 완료했다.** epoch별 train 평균 loss는 0.7213 / 0.5592 / 0.4990, validation CER은 15.7320 / 14.3918 / **14.1733%**였다. 학습 전 validation CER은 **24.2243%**다. validation의 정답 문자는 6,865자다. epoch당 학습+validation 생성은 100.5 / 103.9 / 101.1초였고 초기 로딩·학습 전 평가·저장은 별도다. GPU allocated memory peak는 약 3.63GB(드라이버 전체 사용량과 다름), OOM은 없었다.

validation CER로 **epoch3 / step945**를 선택했다. 실제 어댑터 최대 가중치 변화는 **0.0236570**이다. smoke는 train8/validation4, 본 학습은 train2,513/validation297로 구분해 기록했다. 선택 어댑터를 새 base 모델 객체에 재로딩해 validation 4 crop의 생성 문자열이 일치함을 확인했다. 손실·gradient는 유한했으며 기존 base 가중치는 저장·수정하지 않았다.

같은 GT bbox crop에 직접 `OCR:` 인식만 실행했다. crop에서 검출을 다시 수행하지 않았다. 다음 결과의 분모는 **235 crop / 17페이지 / 16개 고정 페이지·유사 이미지 그룹 / NFC 정답 4,401문자**다. 필자 그룹은 없다.

| 탐색적 test235 | 공식 VL Before | VL + LoRA After | After − Before |
|---|---:|---:|---:|
| micro CER | 21.6087% | 15.2238% | −6.3849%p |
| 공백 제외 CER | 22.1532% | 15.5129% | −6.6403%p |
| 줄 완전 일치율 | 14.4681% (34/235) | 19.5745% (46/235) | +5.1064%p |
| 치환 | 585 | 447 | −138 |
| 삭제 | 190 | 102 | −88 |
| 삽입 | 176 | 121 | −55 |
| 총 오류 | 951 | 670 | −281 |

CER 상대 감소율 `(Before−After)/Before`는 **29.5478%**다. 이전 CTC 실험과 달리 공백 제외 오류와 전체 치환·삭제·삽입도 함께 감소했다. 실제 추론 실패는 **0/235**씩, 생성128 token 제한 도달은 **0개**씩이다. 처리 시간은 Before14.402초 / After16.765초이며 배치4, crop 읽기·processor·생성·decode를 포함하고 모델 로딩은 제외한다. 단일 실행의 순서·warm-up 차이가 있어 정밀 속도 벤치마크는 아니다.

페이지 단위 paired bootstrap 2,000회(seed20261109)의 CER 개선량(Before−After) 95% 구간은 **+4.2956~+9.0026%p**다. 고정 유사 이미지 연결 성분 16그룹으로 재표집(seed42)하면 **+4.3928~+8.8612%p**다. 이 표본 안의 개선은 확인되지만, **이미 참고한 test의 재사용과 필자 독립성 미확보를 해소해 주지는 않는다.** test 이후 설정을 재튜닝하지 않았다.

| 전사 기반 휴리스틱 구분 | crop / 문자 | Before CER | After CER |
|---|---:|---:|---:|
| 한글 포함·ASCII 영문 없음 | 131 / 2,563 | 21.6933% | 16.0749% |
| 한글 + ASCII 영문 | 37 / 845 | 22.3669% | 13.0178% |
| 나머지 영문·숫자 등 | 67 / 993 | 20.7452% | 14.9043% |

세 구분 모두 평균 CER이 개선됐지만 모든 개별 샘플이 개선된 것은 아니다. 오류 수 감소112 / 증가40 / 동일83 crop이다. 동일 오류 수와 동일 문자열은 다르다. 전체 페이지·언어·길이별 CER·S/D/I는 [summary.json](ocr_vl_lora/summary.json)에 있다.

| 사례 | 정답 | Before | After |
|---|---|---|---|
| 개선 hf-0013 | `- 상태가 변하는` | `-상태가 범위는` | `- 상태가 변하는` |
| 개선 hf-0048 | `입력 데이터 크기 (H, W)` | `업격 데이터 크기 (H, W)` | `입력 데이터 크기 (H, W)` |
| 개선 hf-0057 | `◦통계 기반 기법` | `ㅇ통계 기반 기법` | `◦통계 기반 기법` |
| 악화 hf-0008 | `- Position_ids` | `- Position_ids` | `- Position 记录` |

공백·글자·기호 전사 적응이 함께 관찰됐다. 다만 중국어 문자열을 잘못 생성하는 개별 악화 사례가 남아 있어 LoRA가 기본 능력을 완전히 보존한다고 주장할 수 없다. 어떤 원인이 개선을 만들었는지 분리한 ablation은 실행하지 않았다. GT 오기·정규화·형식 표기 차이도 오류에 영향을 줄 수 있으므로 수동 재검수 보류를 자동으로 해제하지 않았다.

이전 CTC 실험과 정확히 같은 **199개 GT crop 부분집합**으로 제한하면 VL 원본 CER **20.6519% → LoRA14.0548%**다. 앞선 CTC40.21→34.50%와 이 차이를 LoRA 자체의 효과라고 설명하면 안 된다. LoRA 효과는 위 동일 VL Before/After 비교로 판단해야 한다.

## 고정 검출기 전체 페이지 추론

앞선 CTC 실험에서 저장한 **동일 원본의 동일 검출 crop**을 그대로 재사용했다. 이 비교는 사전 미지원 영역을 제외한 **이전의 199개 GT 전사 범위**를 유지한다. 따라서 위 VL GT crop235와 표본 범위가 다르다. 그 범위를 넓히거나 다른 검출 결과로 바꿔 점수를 올리지 않았다. 양쪽 모두 같은 bbox·축 정렬 crop·y/x 순서·제외 정책을 적용하고 score 필터를 사용하지 않았다. 포함된 모든 예측을 이어 붙여 채점하므로 매칭된 영역만의 CER이 아니다.

| 같은 고정 검출기 + VL 인식 | Before | After LoRA | After − Before |
|---|---:|---:|---:|
| validation 페이지 CER, 17페이지 | 59.0538% | 45.8585% | −13.1953%p |
| test 페이지 CER, 17페이지 | 97.2865% | 84.7647% | −12.5218%p |
| test 공백 제외 CER | 104.3436% | 87.8511% | −16.4925%p |
| test 치환 / 삭제 / 삽입 | 532 / 153 / 3,223 | 400 / 90 / 2,915 | −132 / −63 / −308 |
| test 검출 P/R/F1 | 49.3703 / 98.4925 / 65.7718% | 동일 | 변화 없음 |

test 페이지 정답 문자는 줄 구분 공백 포함4,017자이며 인식 입력은397개 검출 crop이다. Before/After의 추론 실패는 각각0/397, 페이지 실패0/17이다. 검출 TP196 / FP201 / FN3, 읽기 순서 역전20건이 동일하다. 검출기까지 개선된 것은 아니다. test 인식 시간은18.471→27.375초로 늘었고, 저장된 동일 검출 실행 시간11.700초는 별도다. validation 인식은22.895→30.623초였다. 모델 로딩은 제외하며 단일 실행이다.

**LoRA는 GT crop과 고정 검출기 페이지 양쪽에서 개선됐지만, 페이지 CER84.76%는 여전히 높다.** 오탐201영역과 잔여 삽입2,915문자, 순서 오류 때문에 crop 개선이 운영 수준의 문서 품질로 이어지지 않았다. GT 밖의 실제 미라벨 손글씨가 있으면 FP 집계에도 영향을 준다. 이전의 별도 운영 PaddleOCR-VL1.6 direct spotting validation CER40.60%와도 파이프라인·읽기 순서가 다르므로 같은 모델의 학습 전후처럼 비교하지 않는다. 이번 LoRA에 `Spotting:`을 적용한 전체 페이지 bbox 생성 능력이나 수식·표·다른 언어 보존 능력은 평가하지 않았다.

적용 가치는 **자연어 bbox crop 인식 어댑터의 후속 검증 후보 확보**다. 기존 운영 경로에 바로 적용하거나 배포하지 않았다. 필자·노트가 구분된 새 손글씨와 도형/수식 오탐 검증, 실제 운영 spotting 경로의 별도 평가가 남아 있다.

## 보존된 산출물

- 결과 요약·전체 strata·오류 사례: [summary.json](ocr_vl_lora/summary.json), [비교 CSV](ocr_vl_lora/comparison.csv).
- 실제 학습 로그: [train.log](ocr_vl_lora/train.log), [epoch 기록](ocr_vl_lora/train-history.json), [선택 기록](ocr_vl_lora/train-selection.json), [가중치 갱신·재로딩 검증](ocr_vl_lora/train-verification.json).
- 시작 provenance·환경·실행 소스: `docs/ocr_vl_lora/train-run-receipt.json`, `environment.freeze.txt`, `training_source.executed.py`, `code.sha256.json`.
- 실제 어댑터: `data/experiments/ocr_v2_vl_lora/train/best-adapter/adapter_model.safetensors`(약4.9MiB), 동반 `adapter_config.json`.
- 마지막 어댑터/optimizer: `train/latest-adapter/`, `train/optimizer.pt`. 완전한 resume CLI는 제공하지 않는다.
- 정답 crop 실제 예측: `final/{before,after}.test.predictions.json`, 최종 비교·잠금: `final/comparison.test.json`, `final/final-lock.json`.
- 고정 검출 crop 기반 전체 페이지 비교: `pages/{before,after}.{validation,test}.{predictions,metrics}.json`.
- 전체 로컬 파일 SHA256 목록: [artifacts.sha256.json](ocr_vl_lora/artifacts.sha256.json).

[원본 crop과 실제 예측 비교 이미지](ocr_vl_lora/error-crops.jpg), [사례 JSON](ocr_vl_lora/error-examples.json)을 함께 보존했다. 관련 데이터·마스킹·LoRA·기존 평가 단위 테스트 **32개가 통과**했다([실제 로그](ocr_vl_lora/tests.log)). 초기 전용 환경에 rapidfuzz가 없어 smoke 진입 전에 실패한 뒤, 그 패키지만 설치하고 재실행했다. 기존 GPU 학습 종료, 원격 라벨 쓰기, push, 배포, 유료 자원 사용은 없었다.
