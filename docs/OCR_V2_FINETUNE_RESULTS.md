# V2 한국어·영어 OCR 실제 미세 조정 결과

실행일: 2026-11-09~10, Asia/Seoul. **실제 전체 train 학습과 최종 test 평가를 완료했다.** 모델은 `korean_PP-OCRv5_mobile_rec`이며, 2,348 crop으로 8 epoch / 1,176 optimizer step을 실행했다. GT crop CER은 개선됐지만 공백 제외 CER·영문 성능·전체 페이지 OCR은 악화됐다. **현재 가중치를 운영 OCR 교체용으로 권장하지 않는다.** 기존 PaddleOCR-VL 1.6/Qwen 구조 경로는 수정하지 않았다.

이 문서의 수치는 [기계 판독 요약](ocr_v2_finetune/summary.json), [비교 CSV](ocr_v2_finetune/comparison.csv), 실제 로그·예측을 근거로 한다. test 확인 후 모델·사전·전처리·임계값·학습 설정을 재튜닝하지 않았다. 이후 코드 변경은 입력 무결성 검사, 실행 증거, 보고서, 새 프로세스 재개 검증이다.

## 동일 모델 Before / After

Before는 공식 한국어 학습 가중치 전체를 그대로 로드한 모델이다. After는 그 가중치에서 실제 미세 조정한 모델이다. 사전 확장이나 출력층 재초기화는 없었다. 동일 정답 bbox crop, BGR / 높이 48 / 폭 832 / 종횡비 유지 resize 후 0 padding, 동일 CTC greedy decode, score 필터 없음, 동일 NFC·공백 병합 정규화를 적용했다. crop 평가에서 검출기는 호출하지 않았다.

| 최종 test 지표 | Before | After | After − Before |
|---|---:|---:|---:|
| GT crop micro CER | 40.2086% | 34.4980% | −5.7106%p |
| GT crop 공백 제외 CER | 35.5650% | 38.6349% | +3.0699%p |
| 줄 완전 일치율 | 4.5226% (9/199) | 4.0201% (8/199) | −0.5025%p |
| 치환 | 458 | 907 | +449 |
| 삭제 | 1,052 | 302 | −750 |
| 삽입 | 32 | 114 | +82 |
| 총 편집 오류 / 정답 문자 | 1,542 / 3,835 | 1,323 / 3,835 | −219 |
| 고정 검출기 페이지 CER | 89.7685% | 93.7267% | +3.9582%p |
| 페이지 공백 제외 CER | 93.2724% | 100.9798% | +7.7074%p |
| 페이지 치환 / 삭제 / 삽입 | 632 / 1,004 / 1,970 | 901 / 331 / 2,533 | +269 / −673 / +563 |

GT crop CER 상대 감소율은 `(Before−After)/Before = 14.2023%`이다. Before=0이면 코드에서 N/A(null)로 처리한다. 17페이지 / 16개의 고정 페이지·유사 이미지 그룹 / 199 crop / 정규화 후 3,835문자다. 페이지 문자열에는 줄 사이 구분 공백이 추가되므로 페이지 평가 분모는 4,017문자다. 매칭 성공 영역만을 분모로 사용하지 않았다.

페이지 paired bootstrap 2,000회(seed 20261109)의 CER 개선량(Before−After) 95% 구간은 **−1.2072~+12.5792%p**이다. 유사 이미지 연결 성분 16그룹으로 재표집한 결과(seed 42)는 **−1.1913~+12.7993%p**다. 둘 다 0을 포함하므로 통계적으로 안정적인 개선이라고 주장할 수 없다. 상세 근거: 로컬 `data/experiments/ocr_v2_korean/final/comparison.test.json`, `group-bootstrap.json`.

| test 전사 기반 휴리스틱 언어 구분 | crop / 문자 | Before CER | After CER |
|---|---:|---:|---:|
| 한글 포함·ASCII 영문 없음 | 115 / 2,314 | 44.8142% | 34.2697% |
| 한글 + ASCII 영문 | 34 / 800 | 40.2500% | 31.8750% |
| 나머지 영문·숫자 등 | 50 / 721 | 25.3814% | 38.1415% |

이 구분은 필자·언어 전문가의 확정 라벨이 아니다. 길이별 CER은 1~10자 36.05→40.41%, 11~30자 41.21→34.15%, 31자 이상 39.40→33.42%다. 모든 페이지·언어·길이별 오류 수는 summary의 `recognition_test.strata`에 있다.

## 전체 페이지 평가와 운영 기준선

동일 원본 이미지에 로컬 `PP-OCRv6_medium_det`를 **한 번만** 실행하고 검출 crop·좌표·해시를 동결했다. Before/After는 이 crop을 공유한다. 검출 설정은 저장된 `inference.yml` 기본값이며 test로 조정하지 않았다. 축 정렬 bbox crop을 사용했고 perspective rectification은 양쪽 모두 적용하지 않았다. 읽기 순서는 고정된 y→x 정렬이다. 두 인식기 모두 검출 397영역을 처리했고 실패는 0/397, 페이지 실패도 0/17이다.

IoU ≥0.5 최대 개수 일대일 매칭에서 두 모델 모두 TP=196 / FP=201 / FN=3, **Precision 49.3703% / Recall 98.4925% / F1 65.7718%**다. 검출 성능은 변하지 않았다. GT 대응 순서 역전은 20건으로 동일하다. 누락·오탐·분할/병합 후보와 전체 페이지 예측은 `page-test/*.json`에 있다.

평가 범위는 **보수적으로 선별한 일반 텍스트의 전체 페이지 OCR**이다. 수식·단독 기호·사전 미지원 GT 영역은 제외한다. 예측 면적의 50% 이상이 제외 GT에 포함되면 두 모델에서 똑같이 무시한다. 유효 GT와 제외 GT에 각각 50% 이상 겹치면 페이지 실패로 처리하고 정답을 CER 분모에 남긴다. 실제 본 평가에서 이런 실패는 없었다. 제외 영역에 주로 속하면서 일부 일반 텍스트를 포함한 crop도 제외될 수 있으므로 무제한 문서 OCR 성능으로 해석하면 안 된다. GT에 없는 실제 손글씨가 일부 남았다면 FP 집계도 영향을 받는다.

GT crop 처리 시간은 Before 1.324초 / After 1.140초(199건 batch=16), test 검출 11.700초 + 인식 Before 2.241초 / After 2.087초(397건)다. 인식 시간은 crop 읽기·resize·배치 실행을 포함하고 모델 로딩은 제외한다. 검출 시간은 페이지 읽기·검출·crop 저장을 포함한다. 단일 실행이며 실행 순서·warm-up이 달라 속도 향상 근거로 사용하지 않는다. 예측 문자열이 비어도 성공한 추론이면 삭제로 채점했다.

validation에서 crop CER은 **42.0035→34.1249%**(273 crop), 고정 검출기 페이지 CER은 **61.3966→59.2352%**(17페이지)였다. validation CER 최솟값으로 epoch 8을 선택했다. train loss나 test 성능으로 선택하지 않았다.

현재 운영 경로의 **PaddleOCR-VL 1.6 direct spotting**도 원본 validation 17페이지에서 실제 실행했다. 페이지당 worker timeout 90초, 기존 `paddleocr_vl_worker.py`의 max_pixels=602112 / max_new_tokens=2048을 유지했다. 동일한 일반 텍스트 제외 정책에서 페이지 CER **40.5985%**, 공백 제외 CER **44.1883%**, 검출 P/R/F1 **91.3725 / 85.3480 / 88.2576%**, 실패 **0/17**, worker 로딩 포함 합계 **209.878초**다. 이는 **별도 운영 기준선**이다. VL과 PP-OCRv5 차이를 같은 모델의 학습 효과라고 부르지 않는다. VL은 자체 spotting·읽기 순서를 사용하므로 고정 검출기 Before/After와 조건도 다르다. VL test 평가는 실행하지 않았다.

## 데이터 고정·정제

- 원격 최신 확인: `RADU/labeling-data` = `50aca983da9885d9e7a965c1ef4b05055d47b252`.
- 라벨 SHA256: `dc178e259b69180507176b80fd17696e6d99a2ebc91beef007af6a117e4f5299`.
- 이미지 카탈로그 revision: `93f3a5c0cf545268eb258119fcb92a0ca410b69f`; 기존 parquet 원본 122장은 `7f97f382d9d82aa6b21e9b617b46218f1ec4b0aa`. 페이지별 실제 revision·해시는 `data/processed/ocr_v2/originals/restore_receipts.json`, 크기·PNG 해시는 `pilot-v2/pages.jsonl`과 snapshot catalog에 고정돼 있다.
- 내부 `items/bbox_annotation`을 읽었고 `text`만 정답으로 사용했다. `prediction`을 정답에 복사하지 않았다. GitHub V1/V2 라벨은 수정하지 않았다.
- 최신 수량 163페이지·3,908 bbox, verified 163 / reviewed 3,908. 상태만 믿지 않고 좌표·중복 ID·읽기 순서·중복 영역·uncertain·빈 전사·개행·LaTeX·길이·원본 해시/크기를 검사했다. 전체 세부 감사는 `pilot-v2/quality_report.json`, `review_queue.jsonl`, `docs/ocr_v2/audit_details.json`에 있다.
- raw 유형 제안: text 3,344 / formula 434 / symbol 130. 미분류 처리 경로도 있지만 이번 raw 자동 집계는 0이다. 자동 유형 구분은 휴리스틱이며 수식·도형 속 일반 문자까지 완벽히 구분하지 못한다.
- 16페이지·339 bbox를 전체 보류했다. `hf-0104`는 여러 줄 손글씨가 보여 no_text=true와 모순된다. `hf-0011`은 5+6·8+9·10+11 병합 메모와 분수 구조가 남아 있다. `hf-0000`의 미박스 손글씨도 보류했다. 임의 전사나 자동 병합은 하지 않았다. 원본 이미지에서 다시 육안 확인했다.
- 유효 147페이지에서 원본 좌표 crop 3,569개를 생성한 기존 산출물을 재검증했다. 보류 후 수식·기호 등 524개를 일반 텍스트 인식 대상에서 제외했고, 3,045개 후보 중 사전 미지원 문자가 있는 225개를 추가 보류했다. 총 1,088 bbox가 이번 인식 실험 대상에서 제외됐다(339+524+225). 대표 미지원 문자는 `◦` 127영역, `├` 38영역, `└` 29영역이다. 문자별 집계는 서로 중복될 수 있다. 개행·탭·길이를 조용히 삭제하거나 치환하지 않았다.
- 원본 SHA·RGB 픽셀 동일 이미지는 0쌍, dHash 거리≤4 유사 후보 4쌍은 같은 연결 그룹에 배치했다. crop 전에 seed 42로 raw 페이지 train/validation/test=129/17/17, 그룹=127/16/16을 고정했다. 16개 보류 페이지는 모두 train에 속했다. 필자·노트·동일 도안 메타데이터는 없다. annotator를 필자로 간주하지 않았다. 따라서 **필자 독립성이 보장되지 않는 페이지 그룹 파일럿**이다.

| 인식 실험 | crop | 실제 참여 페이지 | 실제 참여 그룹 | NFC 정답 문자 |
|---|---:|---:|---:|---:|
| train | 2,348 | 111 | 109 | 45,826 |
| validation | 273 | 16 | 16 | 6,359 |
| test | 199 | 17 | 16 | 3,835 |

train/validation/test 사전 추가 보류는 165/24/36개다. 정상 no_text 음성 페이지는 **0개**다. `hf-0104`를 정상 음성으로 평가하지 않았다. no_text 오탐 처리는 합성 단위 테스트로 검증했지만 실제 정상 음성 성능은 이번 데이터로 주장할 수 없다.

## 모델 선택과 실제 학습

공식 [한국어 모델 설명·학습 가중치](https://github.com/PaddlePaddle/PaddleOCR/blob/dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf/docs/version3.x/module_usage/text_recognition.en.md)와 [한국어 학습 설정](https://github.com/PaddlePaddle/PaddleOCR/blob/dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf/configs/rec/PP-OCRv5/multi_language/korean_PP-OCRv5_mobile_rec.yml)을 확인했다. 한국어·영어·숫자를 지원하며 공식 문자 사전의 실제 codepoint로 지원 여부를 검사했다. PP-OCRv6 인식기의 한국어 지원을 가정하지 않았다. PaddleOCR-VL 1.6 공식 문서에서 post-training 설명·추론 가중치를 확인했지만, 이번 upstream checkout에서 이 bbox crop 과제에 바로 대응하는 VL 1.6 학습 recipe를 확인하지 못했다. 검출기를 고정한 소규모 인식기 실험을 위해 공식 한국어 PP-OCRv5 가중치/설정을 선택했다. VL 자체의 미세 조정이 불가능하다고 주장하지 않는다.

공식 코드 commit `dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf`, PPLCNetV3 + SVTR + CTC/NRTR MultiHead다. 968개 tensor key/shape가 모두 일치했다. CTC 출력 11,947 클래스(공식 사전+공백+blank)를 유지했다. NRTR 보조 head 가중치는 보존·동결하고, 실제 배포 인식에 쓰는 backbone/CTC 경로만 CTC loss로 학습했다. 공식 MultiLoss trainer 전체를 그대로 사용한 실험은 아니며 custom recognition adapter다.

공식 백본은 train 모드에서 `[1,40]` adaptive pooling을 사용해 긴 원문 전사와 충돌했다. 첫 smoke는 이를 감지해 중단했고 로그를 남겼다. 전사를 자르지 않고 **추론과 같은 `[3,2]` average pooling**을 train forward에도 적용했다. 입력 폭 832로 104 timestep을 확보했다. train의 최대 94자, 인접 반복 문자 포함 최대 CTC 필요량 99를 처리했다. Before의 추론 구조나 가중치는 바꾸지 않았다.

설정: Adam lr=5e-5, batch=16, 최대 8 epoch, seed=42, global grad clip=5, train에만 대비 배율 0.9~1.1 증강, 매 epoch 전체 validation CER 평가·latest 저장·best 갱신, 개선 없는 3 epoch면 종료. 스케줄러·사전 확장·test 튜닝은 없다. OOM은 없었다. 상세 설정은 [configs/ocr_v2_korean_ctc.json](../configs/ocr_v2_korean_ctc.json), 실제 loss는 [train.log](ocr_v2_finetune/train.log), [epoch 기록](ocr_v2_finetune/train-history.json)에 있다.

smoke는 train 32 / validation 8에서 4 step을 실제 실행했다. 유한 loss·CTC weight 변화 0.00020056·저장/재로딩·export 일치를 확인한 뒤 전체 train을 실행했다. 본 학습의 CTC weight 최대 변화는 **0.05641136**이다. 마지막 train loss=1.73928. epoch별 학습·validation 시간은 약 7.6~8.4초이며 초기 로딩·캐싱·저장 시간은 별도다. export 파일은 실제 `paddle.jit.load`로 재로딩했고 validation 2 crop에서 최대 출력 차이 0 / decode 동일이었다. 임의 길이 입력을 받는 범용 export가 아니라 batch 가변·3×48×832 입력의 Paddle PIR export다.

새 프로세스에서 `latest.pdckpt`의 model/optimizer/Python·NumPy·Paddle RNG를 복원해 step 1176→1177을 실제 수행했다. loss=2.25006, weight 최대 변화=0.00012918. 이 검증 step은 별도 파일에 저장했으며 선택 모델에는 반영하지 않았다.

환경은 RTX4090 24GB / driver580.178.04 / Paddle CUDA runtime12.8 / Python3.10.12 / Paddle GPU3.2.1 / PaddleX3.7.2다. 기존 Qwen 학습은 실행 중이 아니었으며 기존 프로세스·가중치·미추적 작업은 종료·삭제·reset하지 않았다. 전용 `.cache/venvs/ocr-v2`와 별도 프로세스를 사용했으나, **의존성은 기존 HybridOCR site-packages를 읽기 전용 `.pth`로 공유**했다. 독립 패키지 설치까지 완료한 환경이라고 표현하지 않는다. 기존 환경 패키지는 변경하지 않았으며 정확한 버전은 [환경 기록](ocr_v2_finetune/environment.json), [전체 freeze](ocr_v2_finetune/environment.freeze.txt)에 있다.

실행 provenance의 한계: 초기 prepare 기록은 폭 768이었고, 긴 전사 smoke 수정 후 실제 train/test는 폭 832였다. 최종 lock·export graph·실행 중 변경되지 않은 config로 이를 확인했다. train 시작 시 source/config receipt를 저장하지 못해 일부 실행 바인딩은 **사후 재구성**했다. [execution-audit.json](ocr_v2_finetune/execution-audit.json)에 명시했다. 학습 당시 기록한 SHA `dfe77f5803e939389bc6e001533b3b4e9e4c9747aef44d99c1e86aef8db9ca33`와 정확히 일치하는 [실행 코드 사본](ocr_v2_finetune/training_source.executed.py)도 복원·보존했다. 준비 목록·원본 후보·crop 해시를 사후 재검증했고, 새 코드에서는 실행 시작 receipt와 train/eval config·가중치·목록 해시 검사를 강제한다. 사후 기록을 동시 수집 증거로 포장하지 않는다.

## 오류 사례와 해석

[원본 crop 비교 이미지](ocr_v2_finetune/error-crops.jpg), [정답·예측 사례 JSON](ocr_v2_finetune/error-examples.json). 전체 199 crop 중 오류 수 감소 98 / 증가 66 / 동일 35개다. 동일 오류 수가 동일 문자열이라는 뜻은 아니다.

| 변화 | 정답 | Before | After |
|---|---|---|---|
| 개선, hf-0013 | `- 상태가 변하는` | `-상가하는` | `- 상태가 변하는` |
| 개선, hf-0111 | `float: left;` | `floati left;` | `float: left;` |
| 악화, hf-0008 | `GPtoss Model` | `GPtoss Model` | `GPFOsS M.D` |
| 오류 수 동일, hf-0005 | `output Projection` | `ouat Prajection` | ` oupe Prajection` |

근거가 있는 관찰은 삭제 오류가 크게 줄었지만 치환·삽입이 늘었다는 것이다. 한글과 혼합 전사에서는 공백 포함 CER이 내려갔으나, 한글 공백 제외 CER은 40.9014→40.7887%, 혼합은 35.4688→34.8438%로 변화가 작았다. 전체 공백 제외 CER 악화와 영문 영역 퇴행은 인식 품질이 고르게 나아지지 않았음을 보여 준다. 한글 손글씨·공백 전사 쪽으로 적응하면서 영문 구별을 잃은 가능성은 **추론**이며 원인을 분리한 대조 실험은 하지 않았다.

페이지 결과에서는 GT crop보다 훨씬 많은 오탐 영역(201개), 미세 조정 후 +563개의 삽입 오류, y/x 순서 역전 20건이 확인됐다. 실제 이미지에는 도형·수식·부분 전사·다단 구성이 있고, crop 전사만으로 학습한 모델은 검출 오탐에서 더 긴 잘못된 문자열을 출력할 수 있다. 따라서 이 실험의 가치는 운영 대체 모델 확보보다 **인식·검출·읽기 순서 문제를 분리하고 재현 가능한 학습/평가 경로를 확보한 것**에 있다. 다음 실험은 새 개발 분할에서 라벨 완전성 재검수, 필자·노트 그룹 확보, 영문 보존과 FP 억제를 검토해야 한다. 이번 test는 추가 튜닝에 쓰지 않았다.

## 산출물과 재현

로컬 대용량 산출물 루트: `data/experiments/ocr_v2_korean/`.

- `prepared/{train,validation,test}.json`, `.txt`, `held.json`, `provenance.json`: 원문·crop 경로·분할과 사전 보류.
- `train/best.pdparams`: 실제 선택된 106MB multi-head 체크포인트. SHA256 `8aa946c398506f90786a63a3e0f660e1782ef0abfbccd8b6208411f90ed54299`.
- `train/latest.pdckpt`: optimizer·RNG 포함 실제 재개 체크포인트. `resume-fresh/resumed-step.pdckpt`는 별도 검증 결과.
- `train/inference.json`, `inference.pdiparams`: 실제 PIR export(가중치 약 27MB), `verification.json`은 실제 호환성 결과.
- `train/{before,after}.validation.json`, `history.json`, `selection.json`: validation 선택 근거.
- `final/{before,after}.test.json`, `comparison.test.json`: 동일 GT crop 실제 예측·지표·strata·bootstrap.
- `det-{validation,test}/`: 원본에서 동결한 검출 결과와 crop. `page-{validation,test}/`: 실제 전체 페이지 예측·매칭·누락·오탐·순서 오류.
- `vl-validation/`: 별도 운영 VL baseline의 페이지별 출력·로그·지표.
- `data/processed/ocr_v2/{snapshot-50aca98,originals,pilot-v2}/`: 고정 라벨·원본 복원 영수증·crop·overlay·보류 큐·split manifest. 원본 WebP가 아닌 PNG를 crop했다. 좌표는 원본 xyxy이고 fractional 경계의 floor/ceil은 `crop_bbox`에 별도 기록돼 있다.
- [artifact SHA256 목록](ocr_v2_finetune/artifacts.sha256.json): 실제 파일 존재·해시. 공식 학습 가중치와 사전·설정 해시는 실행 audit에 있다.

현재 PC에서 재실행할 때 기존 경로를 덮어쓰지 않고 새 출력 루트를 지정한다. 아래 명령은 또 하나의 실제 전체 실험을 실행하므로 GPU 자원을 확인하고 실행한다.

```bash
# 기존 전용 venv: read-only runtime dependency 경로를 공유한 실제 사용 환경
.cache/venvs/ocr-v2/bin/python -m src.training.run_ocr_v2_experiment \
  --config configs/ocr_v2_korean_ctc.json \
  --output data/experiments/ocr_v2_korean_repeat --with-vl

# 데이터와 지표 검증 (GPU 불필요)
.venv/bin/python -m src.ocr_data.validate --dataset data/processed/ocr_v2/pilot-v2
.venv/bin/python -m unittest tests.test_ocr_v2 tests.test_ocr_finetune tests.test_ocr_finetune_page -v

# 저장된 실제 결과에서 요약 재생성
.venv/bin/python -m src.evaluation.summarize_ocr_finetune
```

새 PC에서는 위 공식 PaddleOCR commit과 공식 URL의 `korean_PP-OCRv5_mobile_rec_pretrained.pdparams`를 준비하고, audit의 SHA를 검증한다. `.cache/venvs/ocr-v2` 의존성은 제공한 freeze 및 CUDA 호환 Paddle wheel로 설치한다. 원본·데이터 생성 명령은 [기존 데이터 파이프라인 문서](OCR_V2_PIPELINE.md)에 있다. tokenizer/학습 가중치 다운로드 없이 데이터·평가 단위 테스트를 실행할 수 있다. Hugging Face 원본은 각 receipt의 고정 revision으로 복원해야 하며, 다른 PNG encoder를 썼다면 해시가 달라질 수 있다.

실제 단위 테스트 **28개가 통과**했다. 데이터 검증 결과와 테스트 출력은 `ocr_v2_finetune/data-validation.json`, [tests.log](ocr_v2_finetune/tests.log)에 있다. 독립 코드 검토에서 지적된 설정·입력 provenance 연결 문제는 hash/config guard로 보강했고, 이번 실행의 사후 기록 한계는 위에 공개했다.

Git checkout/reset, 원격 라벨 수정, 원격 push, 배포, 유료 자원 임대는 실행하지 않았다. 결과 코드와 문서는 현재 작업 트리에 남겼으며 commit하지 않았다.
