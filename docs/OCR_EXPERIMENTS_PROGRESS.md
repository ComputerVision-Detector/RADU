# RADU V2 OCR 데이터 정제·미세 조정·평가 종합 기록

작성 기준: 2026-11-10, Asia/Seoul. 이 문서는 지금까지 실제 실행한 작업을 통합한 기록이다. 개별 실험의 원본 로그·설정·예측·해시는 아래 상세 보고서에 연결한다.

## 1. 현재 결론

**PaddleOCR-VL 1.6의 최초 언어 LoRA에서 인식 CER 개선을 확인했다. 이후 추가 학습·고해상도·가중치 보간·train 정제·시각 LoRA 실험에서는 추가 CER 개선을 확인하지 못했다.**

현재 유지한 후보는 최초 VL 언어 LoRA다. 공식 VL과 동일한 test235 crop에서 CER이 **21.6087% → 15.2238%**, 절대 **6.3849%p**, 상대 **29.5478%** 감소했다. 공백 제외 CER과 치환·삭제·삽입도 함께 감소했다. 다만 test 재사용과 필자 독립성 미확보로, 미사용 독립 필자에 대한 최종 일반화 성능이라고 주장하지 않는다.

고정 검출기를 사용한 전체 페이지 CER은 **97.2865% → 84.7647%**로 개선됐지만 여전히 높다. 인식기 개선만으로 운영 문서 품질이 충분해졌다는 결론은 내리지 않았다. 운영 OCR/Qwen 경로는 변경하지 않았고 배포하지 않았다.

## 2. 요청 범위와 보존 원칙

수행 범위는 V2 라벨 스냅샷 고정, 원본 복원, 품질 감사, 분할·crop 생성, 실제 인식기 학습, 동일 조건 평가, 오류 분석 및 재현 코드 작성이다.

- 기존 수정사항·가중치·체크포인트를 보존했다. 강제 checkout/reset과 기존 학습 프로세스 종료를 하지 않았다.
- 원격 V1/V2 라벨 수정, push, 운영 배포, 유료 서버 신규 임대는 하지 않았다.
- PaddleOCR-VL의 자연어 OCR과 Qwen3-VL의 구조 추출은 기존 경로를 유지했다.
- GT crop 평가는 검출기를 다시 호출하지 않는 인식 전용 경로로 실행했다.
- 전사는 `text`만 사용했다. `prediction`을 정답으로 대체하거나, 불확실한 정답을 OCR로 채우지 않았다.

## 3. 데이터 고정과 기본 품질 감사

| 항목 | 실제 사용 값 |
|---|---|
| 저장소 | ComputerVision-Detector/RADU |
| 라벨 브랜치 | RADU/labeling-data |
| 라벨 파일 | labeling/data/live-v2.json |
| 사용 커밋 | `50aca983da9885d9e7a965c1ef4b05055d47b252` |
| 라벨 SHA256 | `dc178e259b69180507176b80fd17696e6d99a2ebc91beef007af6a117e4f5299` |
| 원본 라벨 규모 | 163페이지 / 3,908 bbox |
| dataset manifest SHA256 | `c8adbdb2abd9e9a2d775bc36bd40866f6592f658a5c6e8fc0d1a0247d1eb094a` |

원본 픽셀 좌표 `[x1,y1,x2,y2]`를 사용했다. 축소된 웹 WebP를 원본 bbox로 자르지 않았다. 원본별 revision·이미지 해시·크기는 복원 receipt와 manifest에 기록했다. 주 원본 revision은 `7f97f382d9d82aa6b21e9b617b46218f1ec4b0aa`, 추가 원본은 카탈로그에 고정된 개별 revision을 따른다.

상태·reviewed·uncertain·빈 전사·ID·읽기 순서·좌표·중복 영역·이미지 무결성·긴 전사·수식/기호 후보 등을 감사했다. reviewed/verified 표시만으로 실제 정답 품질이 보장된다고 보지 않았다.

`hf-0104`는 no_text 표시와 실제 손글씨가 충돌하고, `hf-0011`은 bbox 병합 메모가 해결되지 않아 보류했다. 임의로 전사나 좌표를 고치지 않았다. 초기에는 총16페이지·339 bbox를 보류했고, 나머지147페이지·3,569 crop을 파이프라인에서 관리했다. 일반 텍스트 후보와 수식·기호·불확실 영역은 분리했으며 자동 유형 구분은 휴리스틱이다.

### 분할과 표본 수

crop 생성 전에 페이지·확인 가능한 동일/유사 이미지 그룹으로 seed42 분할을 고정했다. 같은 페이지와 확인된 중복·변형 그룹이 분할을 넘지 않게 했다. annotator를 필자로 간주하지 않았다. 필자·노트 정보가 충분하지 않아 **페이지 단위 파일럿**이다.

| 모델 조건 | train crop | validation crop | test crop |
|---|---:|---:|---:|
| VL 텍스트 후보 | 2,513 | 297 | 235 |
| CTC 사전·길이 조건 통과 후보 | 2,348 | 273 | 199 |
| 후속 LoRA 선택용 validation | — | 244 | 기존235 유지 |
| 최신 train 정제 후 | 2,341 | 기존244 유지 | 기존235 유지 |

후속 라운드의 validation244는 `hf-0019`, `hf-0095`의 재검수 필요53 crop을 제외한14페이지·5,613문자다. 전체297 결과도 진단용으로 계속 보존했다. **297개 점수와244개 점수를 직접 비교해 개선이라고 표현하지 않는다.**

VL test는17페이지·16그룹·235 crop·NFC 정규화 후4,401문자다. 전체 페이지 평가의199 GT 범위는 CTC 조건에서 고정한 별도 부분집합이므로 표본 범위를 혼동하지 않는다.

## 4. 실험 A: 한국어 PP-OCRv5 인식기 미세 조정

모델은 공식 `korean_PP-OCRv5_mobile_rec` 학습 가중치에서 시작했다. 사전 확장이나 출력층 재초기화 없이 실제 train2,348개로 **8epoch·1,176step** 학습했다. Before/After에서 동일 crop·전처리·CTC greedy decode를 사용했다.

| test199 지표 | Before | After | 차이 |
|---|---:|---:|---:|
| CER | 40.2086% | 34.4980% | −5.7106%p |
| 공백 제외 CER | 35.5650% | 38.6349% | +3.0699%p |
| 완전 일치율 | 4.5226% | 4.0201% | −0.5025%p |
| 고정 검출기 페이지 CER | 89.7685% | 93.7267% | +3.9582%p |

삭제 오류는 줄었지만 치환·삽입과 영문 영역 오류가 늘었다. 페이지/그룹 bootstrap 개선 구간도0을 포함했다. **운영 교체 후보로 권장하지 않았다.**

실제 checkpoint 저장·새 프로세스에서 optimizer/RNG 복원 후 추가1step·Paddle export 재로딩을 검증했다. 긴 전사 때문에 train pooling을 추론과 일치시키고 입력 폭832를 사용했다. 일부 최초 실행 provenance는 사후 재구성했으며 그 한계를 상세 기록에 공개했다.

상세: [OCR_V2_FINETUNE_RESULTS.md](OCR_V2_FINETUNE_RESULTS.md).

## 5. 실험 B: PaddleOCR-VL 1.6 언어 LoRA

공식 native VL 가중치를 엄격하게 로드한 뒤 언어부18layer의 q/k/v/o에 rank8·alpha16·dropout0.05 LoRA를 적용했다. tokenizer 표현 가능성을 검사했으며 CTC 사전 때문에 제외됐던 문자도 문자열 치환 없이 처리했다.

train2,513개 전체로 **3epoch·945step**, AdamW lr5e-5·batch2·누적4·seed42로 실제 학습했다. validation297 생성 CER은 **24.2243% → 14.1733%**, 최선은 epoch3였다.

| 동일 test235 지표 | 공식 VL Before | VL + 언어 LoRA | 차이 |
|---|---:|---:|---:|
| CER | 21.6087% | 15.2238% | −6.3849%p |
| 공백 제외 CER | 22.1532% | 15.5129% | −6.6403%p |
| 완전 일치율 | 14.4681% | 19.5745% | +5.1064%p |
| 치환 / 삭제 / 삽입 | 585 / 190 / 176 | 447 / 102 / 121 | 모두 감소 |

상대 CER 감소율은29.5478%다. 페이지 paired bootstrap2,000회의 개선량95% 구간은 **+4.2956~+9.0026%p**였다. 개별 crop 오류 수는112개에서 감소,40개에서 증가,83개에서 동일했다.

동일 고정 검출기의 전체 페이지 test CER은97.2865%→84.7647%였다. 별도 운영 VL direct spotting validation CER40.60%는 다른 경로·범위의 기준선이며 위 수치를 같은 모델 학습 효과처럼 섞어 비교하지 않았다.

현재 유지한 adapter:

```text
data/experiments/ocr_v2_vl_lora/train/best-adapter
SHA256: ab1bfcfba5c0711ebfc82e83432eca13b47fbb48ca0800d9e7b4d590e15421e5
```

상세: [OCR_V2_VL_LORA_RESULTS.md](OCR_V2_VL_LORA_RESULTS.md).

## 6. 실험 C: 추가 학습·고해상도·가중치 보간

이전 언어 LoRA에서 시작해 추가 학습과 픽셀 예산2배 학습을 각각 train2,513개·3epoch·945step 실행했다. 추가 학습률은2e-5였다. test 접근 전 부모와 추가 학습 adapter의 실제 delta를50:50으로 합친 후보도 하나 평가했다. 비율 탐색은 하지 않았다.

| 고정 validation244 조건 | CER |
|---|---:|
| 기존 부모 LoRA | **12.7917%** |
| 해상도만 증가, 재학습 없음 | 13.7894% |
| 추가 학습 최선 | 12.8095% |
| 고해상도 학습 최선 | 13.3262% |
| 50:50 delta 보간 | 12.9521% |

총1,890개의 새 optimizer step을 완료했지만 추가 CER 개선은 없었다. 부모와 추가 학습 최선의 차이는 단1편집 오류로, 의미 있는 악화라고 단정하지 않았다. train loss 감소가 validation 개선으로 이어지지 않았다. **부모를 유지**했고 선택 모델 test CER은15.2238%로 동일했다.

상세: [OCR_LORA_ROUND2_RESULTS.md](OCR_LORA_ROUND2_RESULTS.md).

## 7. 실험 D: train 품질 감사 후 언어부·시각부 LoRA

### 실제 정제 범위

train2,513개 전체의 hash·이미지 읽기·크기·위험 표시를 검사했다. 모든111페이지를 대표하도록260 crop·17 contact sheet를 AI가 시각 검토했다. 전체 전사를 사람이 재검수한 것은 아니다.

3페이지 전체128 crop와 개별44 crop, 합계 **172개를 보류**했다. 사유는 흐름도·진리표·미적분 등 유형 재검수128개, 이웃 줄 포함26개, 전사 불일치/불확실6개, 수식·기호 등8개, 경계 잘림2개, 취소선·덮어쓰기2개였다.

정제 train은 **2,341 crop·108페이지·47,164문자**다. 원문 정답은 수정하지 않았고 예측 기반 선별도 하지 않았다. 짧은 숫자·표 라벨 제외 비중이 높아 구성 변화가 있다. 남은 데이터의 완전한 정답 품질을 보증하지 않는다.

### 실제 학습 결과

동일 부모·정제 train·lr2e-5·3epoch·879step으로 두 조건을 실행했다.

- 언어부: 기존 언어 LoRA 추가 학습, 학습 파라미터1,253,376개.
- 시각부: 부모 언어 LoRA 동결 + 시각 encoder 마지막4block q/v에 새 rank8 LoRA, 학습 파라미터147,456개. projector는 동결.

| validation244 | 선택 epoch | CER | 공백 제외 CER | 완전 일치율 |
|---|---:|---:|---:|---:|
| 기존 부모 | 기존 | **12.7917%** | **13.0444%** | 21.3115% |
| 정제 언어부 | 1 | **12.7917%** | 13.1533% | **23.7705%** |
| 정제 시각부 | 2 | 12.8630% | 13.1315% | 22.5410% |

총1,758개의 새 optimizer step을 실행했다. 언어부의 완전 일치율은52/244→58/244로 늘었지만 CER은 동률이고 공백 제외 CER은 악화했다. 시각부는 부모보다4편집 오류 많았다. CER 기준으로 추가 개선을 입증하지 못해 **기존 부모 유지**를 선택했다.

모든 비학습 파라미터의 byte hash 불변, 의도한 LoRA의 실제 갱신, 유한 loss, 새 base에 adapter를 재로딩한4개 생성 일치를 검증했다. 시각 checkpoint는 부모 언어 adapter와 함께 활성화해야 하며 공식 base에 시각 adapter만 적용하면 실험 조건이 아니다.

단독 crop CLI에서도 두 adapter 동시 활성화와 실제 추론을 확인했다. 다만 CONV2 사례는 batch4 평가와 단독 추론 출력이 달랐다. 개별 demo가 전체 batch 평가를 그대로 재현한다고 주장하지 않으며 비교 수치는 동일 batch 조건을 유지했다.

이 실험은 이미 정제 전 데이터를 배운 부모의 추가 학습이다. 처음부터 정제 데이터로 학습한 결과가 아니고 이전보다 step 수도 달라, 정제의 독립적인 인과 효과를 분리하지 못했다.

상세: [OCR_CLEAN_VISION_RESULTS.md](OCR_CLEAN_VISION_RESULTS.md).

## 8. 전체 페이지 평가와 남은 병목

최신 선택 모델로 고정 검출397 crop/199 GT·17페이지를 실제 다시 추론했다.

| 항목 | 결과 |
|---|---:|
| 페이지 CER | 84.7647% |
| 공백 제외 CER | 87.8511% |
| 치환 / 삭제 / 삽입 | 400 / 90 / 2,915 |
| IoU0.5 검출 TP / FP / FN | 196 / 201 / 3 |
| 검출 Precision / Recall / F1 | 49.3703% / 98.4925% / 65.7718% |
| 읽기 순서 역전 | 20건 |
| 최신 detector crop 인식 실행 시간 | 24.745초 |
| 추론 실패 | 0건 |

위 시간은 모델 로딩·검출 실행을 제외한 단일 인식 실행 시간이다. 매칭 성공 영역만 채점한 CER이 아니다. 검출기는 학습하지 않았으므로 검출 성능 개선을 주장하지 않는다. 누락 GT가 apparent FP에 영향을 줄 수 있고, 많은 삽입·영역 범위 불일치·읽기 순서 문제가 남아 있다. test에 정상 no_text 페이지가 없어 no_text 오탐 성능은 검증되지 않았다.

과적합·정답 잡음·작은 표본·입력 분포·적응 범위는 가능한 원인이다. 현재 실험으로 각 원인을 분리해 확정하지는 못했다.

## 9. 환경·검증·재현

주 GPU는 RTX4090 24GB, driver580.178.04다. CTC는 Python3.10/Paddle GPU3.2.1, VL은 Python3.11/PyTorch2.14.1+cu126/Transformers5.22.1/PEFT0.21.2를 사용했다. 전용 venv를 사용했지만 일부 의존성은 기존 runtime site-packages를 읽기 전용으로 공유했다. 패키지 설치까지 완전 독립된 환경이라고 표현하지 않는다.

최신 회귀 테스트 **41개 통과**. 데이터/분할 누수·prediction 혼입 방지·보류 처리·NFC·편집 거리·bbox 일대일 매칭·빈 결과/실패·LoRA 활성화/동결·설정 drift를 포함한다. GPU smoke·본 학습의 저장/재로딩 검증은 별도로 완료했다. CTC는 export·실제 재개 step도 검증했고, 최신 LoRA CLI는 optimizer를 저장하지만 학습 재개 entrypoint는 제공하지 않는다.

```bash
# 최신 회귀 테스트
.cache/venvs/ocr-vl-lora/bin/python -m unittest \
  tests.test_ocr_clean_vision tests.test_ocr_train_audit \
  tests.test_ocr_lora_round2 tests.test_ocr_vl_lora \
  tests.test_ocr_v2 tests.test_ocr_finetune tests.test_ocr_finetune_page -v

# 최신 실제 결과 요약 재생성
.venv/bin/python -m src.evaluation.summarize_ocr_clean_vision

# 새 output 경로에 crop 비교 추론
.cache/venvs/ocr-vl-lora/bin/python -m src.cv.ocr_clean_vision_infer \
  --image /absolute/path/to/crop.png --output /path/to/new-output --arm selected
```

전체 학습 재실행은 각 보고서의 설정 파일과 명령을 사용하되 기존 산출물을 보존하도록 새 output 경로를 지정한다. 본 기록 작성 과정에서 학습이나 평가를 다시 수행하지는 않았다.

## 10. 산출물 위치

| 산출물 | 위치 |
|---|---|
| 데이터 스냅샷·원본·crop·분할 | `data/processed/ocr_v2/` |
| 최신 train 감사·보류·contact sheet | `data/processed/ocr_v2/train-audit-round3/` |
| CTC checkpoint·export·예측 | `data/experiments/ocr_v2_korean/` |
| 현재 유지한 최초 VL LoRA | `data/experiments/ocr_v2_vl_lora/` |
| 추가 학습·고해상도·보간 | `data/experiments/ocr_v2_vl_lora_round2/` |
| 정제 언어·시각 checkpoint·예측 | `data/experiments/ocr_v2_vl_lora_round3/` |
| 최신 정량 요약·CSV·hash·로그 | [docs/ocr_clean_vision/](ocr_clean_vision/) |
| 최신 테스트 실제 로그 | [tests.log](ocr_clean_vision/tests.log) |
| 데이터 파이프라인 설명 | [OCR_V2_PIPELINE.md](OCR_V2_PIPELINE.md) |

데이터 정제/분할은 `src/ocr_data/`, 학습은 `src/training/ocr_finetune.py`, `ocr_vl_lora.py`, `ocr_lora_round2.py`, `ocr_clean_vision.py`, 평가는 `src/evaluation/`, bundle 추론은 `src/cv/ocr_clean_vision_infer.py`에 구현했다. 대용량 모델과 데이터는 로컬 산출물이며 문서만으로 가중치가 배포되는 것은 아니다.

현재 판단은 **최초 VL 언어 LoRA를 로컬 인식 후보로 유지하고, 운영 적용은 보류**다. 후속 CER 개선을 주장하려면 새로운 방법의 동일 조건 validation 개선과 더 신뢰할 수 있는 정답·미사용 평가셋이 필요하다. 이 문서의 후속 필요 사항은 아직 실행한 실험으로 집계하지 않는다.
