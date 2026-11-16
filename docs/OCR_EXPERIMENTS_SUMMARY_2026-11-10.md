# RADU V2 OCR 누적 실험 종합 기록 — 2026-11-10 최신본

작성 기준: 2026-11-10, Asia/Seoul. 최초 데이터 정제부터 인식기 미세 조정, LoRA 후속 학습, 페이지 GT/FP 감사, CTC 범위 필터 복원, 최근 GT 경계·crop 실험까지 실제 실행 내용을 통합했다. 이전 종합 기록과 개별 보고서는 보존했다. 이번 문서 작성에서는 학습·추론을 다시 실행하지 않았다.

**읽는 순서:** 1절의 현재 판단 → 11~13절의 최근 진단 → 14절의 평가 조건별 수치 구분. 최초 학습과 데이터 상세는 3~7절에 있다.

## 1. 현재 결론

**PaddleOCR-VL 1.6의 최초 언어 LoRA에서 인식 CER 개선을 확인했다. 이후 추가 학습·고해상도·가중치 보간·train 정제·시각 LoRA 실험에서는 추가 CER 개선을 확인하지 못했다.**

현재 유지한 후보는 **최초 VL 언어 LoRA와 기존 crop 처리**다. 공식 VL과 동일한 test235 crop에서 CER이 **21.6087% → 15.2238%**, 절대 **6.3849%p**, 상대 **29.5478%** 감소했다. 공백 제외 CER과 치환·삭제·삽입도 함께 감소했다. 다만 test 재사용과 필자 독립성 미확보로, 미사용 독립 필자에 대한 최종 일반화 성능이라고 주장하지 않는다.

최초 고정 검출기·legacy199 GT 기준 페이지 CER은 **97.2865% → 84.7647%**였다. 이후 감사에서 GT 누락과 CTC 문자 사전에 의한 정상 줄 제외를 확인했다. 같은 GT v2에서 필터 복원은 **30.0615% → 28.9025%**였으나 삽입 증가와 넓은 신뢰구간이 남았다.

가장 최근의 이미지 기반 여백 축소는 validation CER **12.9877% → 13.7003%**로 악화해 채택하지 않았다. 어두운 배경에서 문장을 잘라낸 실패를 실제 확인했다. GT bbox11개 수정 후 페이지 CER **25.0084% → 23.6041%**는 **평가 경계 변경의 효과**이며 모델 개선으로 집계하지 않는다.

현재까지 운영 OCR/Qwen 경로는 변경하지 않았고 배포하지 않았다. 독립 사람 GT 검수는0건이며, 기존 진단 집합을 독립 test나 최종 일반화 벤치마크라고 부르지 않는다.

실제로 완료한 본 학습은 CTC1회, 최초 VL LoRA1회, 추가/고해상도2회, 정제 언어/시각2회로 **총6회**다. 이후 페이지 감사·범위 복원·경계 실험은 기존 모델을 고정한 추론/재채점이며 새 학습 횟수에 넣지 않는다.

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

## 8. 최초 LoRA의 legacy 전체 페이지 평가와 병목

최초 LoRA와 후속 선택 유지 확인에서, 고정 검출397 crop/legacy199 GT·17페이지를 실제 추론했다. 이 절의 수치는 이후 GT v2~v4와 범위가 다르다.

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

최신 관련 회귀 테스트는 **68개 통과**다. 초기41개에서 페이지 감사13개, 범위 복원6개, 경계 실험8개를 추가했다. 데이터/분할 누수·prediction 혼입 방지·보류 처리·NFC·편집 거리·bbox 일대일 매칭·빈 결과/실패·LoRA 활성화/동결·설정 drift를 포함한다. GPU smoke·본 학습의 저장/재로딩 검증은 별도로 완료했다. CTC는 export·실제 재개 step도 검증했고, 최신 LoRA CLI는 optimizer를 저장하지만 학습 재개 entrypoint는 제공하지 않는다.

```bash
# 최신 회귀 테스트
.cache/venvs/ocr-vl-lora/bin/python -m unittest \
  tests.test_ocr_boundary tests.test_ocr_scope_recovery tests.test_ocr_page_audit \
  tests.test_ocr_clean_vision tests.test_ocr_train_audit tests.test_ocr_lora_round2 tests.test_ocr_vl_lora \
  tests.test_ocr_v2 tests.test_ocr_finetune tests.test_ocr_finetune_page -v

# 정제 언어/시각 LoRA 라운드의 결과 요약 재생성
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

위10절까지는 초기 학습 라운드의 기록이다. 이후 실행한 페이지 감사·범위 복원·경계 실험은 다음 절에 이어진다.


## 11. 후속 진단 E: 페이지 GT 범위와 apparent FP201 전수 감사

최초 페이지 점수가 높은 이유를 분리하기 위해 기존 LoRA와397개 예측을 고정했다. 기존 정규화·조립·IoU0.5 일대일 매칭을 재실행해 **TP196/FP201/FN3, CER84.7647%, S400/D90/I2,915, 순서 역전20건**을 정확히 재현했다. GT crop235의15.2238%도 저장 예측에서 재현했다.

17개 원본 전체와 FP201개를 모두 AI가 시각 검토했다. 최종 주 분류는 다음과 같다.

| apparent FP201의 최종 주 분류 | 개수 |
|---|---:|
| 정상 텍스트의 legacy GT 누락 | 94 |
| 수식·기호·표 숫자·취소선 등 평가 제외 영역 | 82 |
| 분할·병합/IoU 문제 | 11 |
| 실제 비텍스트 오탐 | 1 |
| 판단 보류 | 13 |
| 확정된 전체 영역 중복 | 0 |
| 합계 | **201** |

부분 반복3개는 확정 중복이 아니라 분할·병합 범주로 기록했다. 이웃 줄 유입은 별도 보조 태그로 남겼다. unmatched라는 이유만으로 모두 검출기 오탐이라고 판단하지 않았다.

새 페이지 GT는 legacy199개 중19개 보류, CTC 제외 정상 텍스트32개 복원, 이미지에서 확인한 텍스트94개 추가로 **306개**가 됐다. 원본 V2와는 별도 버전으로 만들었고, AI 판정·검출 도움 geometry라는 한계를 기록했다.

| 같은397개 예측의 재채점 | GT 영역 | 정답 문자 | 페이지 CER |
|---|---:|---:|---:|
| legacy 기준 | 199 | 4,017 | 84.7647% |
| 새 GT v2, 추가 ignore 미적용 | 306 | 5,695 | 46.9008% |
| 새 GT v2, 고정 ignore 적용 | 306 | 5,695 | 30.0615% |

이 변화는 GT/정책/범위 변경이며 모델 개선이 아니다. ignore는 예측 면적50% 이상이 ignore bbox에 있고 유효 GT와 전혀 교차하지 않을 때만 적용한다. 유효 GT와 교차하는 mixed 예측은 유지했다.

검출·crop·순서를 분리한 추가 진단도 실행했다.

| 비교 | 고정 분모 | 결과 |
|---|---|---|
| 정답 bbox C0 vs 대응 검출 bbox C1 | 16페이지·138영역·2,501문자 | CER13.8745% →15.3139% |
| 같은 인식문, 기존 순서 O0 vs 감사 순서 O1 | 16페이지·138영역·구분 공백 포함2,623문자 | CER14.3729% →14.2204%, 역전1/813쌍→0 |
| 실제 고정 검출 P0 vs 정답 bbox/순서 P1 | 17페이지·306GT·5,695문자 | 페이지 CER30.0615% →14.1001% |

C0/C1은 매칭 가능한 영역만의 경계 영향이고 전체 페이지 성능이 아니다. O0/O1도 공통 영역만의 순서 진단이다. P1은 GT를 참조하는 비배포 조건이며, P0와의 차이에 검출·crop·범위·순서·batch 등이 함께 작용한다. 차이를 합산해 원인별 기여율을 만들지 않았다.

이번 감사에서 **raw 검출500개 중 CTC 범위 필터로103개가 제거되고397개만 저장됐음**을 확인했다. 이 때문에 새 FN을 모두 실제 검출 누락으로 해석할 수 없었다.

상세: [OCR_PAGE_AUDIT_RESULTS.md](OCR_PAGE_AUDIT_RESULTS.md).

## 12. 후속 실험 F: CTC 범위 필터 복원과 GT v3

같은 검출기·같은 설정으로17개 원본을 실제 다시 실행했다. **이전397개의 ID·bbox·crop hash를 모두 일치시키고 추가103개를 정확히 복원**했다. 검출/추론 단계에서는 GT나 CTC 사전으로 영역을 제거하지 않고500개 전부를 저장했다.

추가103개만 고정 LoRA로 인식했다. 기존397개 인식문과 원래 batch 문맥은 보존했으며, 공통 예측의 문자열이 바뀌지 않았음을 검증했다. 대표4개를2회 반복해 tensor/출력과 기존 캐시의 일치도 확인했다.

| 같은 GT v2:17페이지·306GT·5,695문자 | 기존397 bank | 복원500 bank |
|---|---:|---:|
| ignore 후 채점 예측 | 304 | 346 |
| TP / FP / FN | 271 /33 /35 | 299 /47 /7 |
| Precision / Recall / F1 | 89.1447 /88.5621 /88.8525% | 86.4162 /97.7124 /91.7178% |
| 페이지 CER | 30.0615% | **28.9025%** |
| 공백 제외 CER | 31.2230% | 29.3783% |
| S / D / I | 540 /542 /630 | 528 /147 /971 |

삭제는395자 줄었지만 삽입은341자 늘어 전체 CER 개선은1.1589%p에 그쳤다. 페이지 bootstrap95% 개선 구간은 **−8.0551~+10.1283%p**로0을 포함한다. 안정적인 전체 개선이라고 단정하지 않았다.

CTC에서 제외했던 정상 GT32개는 `├`11개, `└`6개, `─`2개, `◦`12개, `中`1개 때문에 사전 검사에서 탈락했다. 그중28개는 복원 검출과 기본 IoU로 매칭됐다. 검출기 학습보다 **인식 모델과 맞는 범위 계약**이 먼저라는 근거다.

추가103개는 AI 시각 검토상 일반 본문/코드34개, 수식/행렬60개, 단독 화살표/중괄호7개, 덮어쓴 제목1개, 분리 글자1개였다. 고정 GT v2 정책에서42개가 채점에 남았고, 그중 수식10개도 작은 GT 교차 때문에 유지됐다. 수식 유입을 피하려고 임계값을 사후 변경하지 않았다.

### 별도 GT v3 재검토

확대 원본37항목을 재검토했다. 원본 GT13개를 포함으로 복원했고, 이 중9개 전사는 이미지에 근거해 새 버전에 수정했다. 잘린 웹사이트1개는 보이지 않는 접미사를 추정하지 않고 보류했다. 유효 GT는 **306→318개**, 페이지 정답은 **5,695→5,982문자**가 됐다. geometry와 order는 그대로다.

| 별도 GT v3:17페이지·318GT·5,982문자 | 기존397 bank | 복원500 bank |
|---|---:|---:|
| 페이지 CER | 28.0843% | 25.0084% |
| 공백 제외 CER | 29.2518% | 25.6851% |
| S / D / I | 517 /631 /532 | 541 /159 /796 |

GT v3 내부의 bank 비교와 GT v2 내부 비교를 구분했다. GT v2의30.0615%와 v3의25.0084%를 직접 모델 개선량으로 계산하지 않았다. v3도 AI 초안이며 독립 사람 검수는0건이다. 기존13개 FP 보류도 해결됐다고 표시하지 않았다.

실제 GPU 입력은 새103+반복8=111 crop이었다. 새 학습은 없다. 사람 검수용37항목 화면과 미검수 template을 만들었지만, 그 자체를 검수 완료로 집계하지 않았다.

상세: [OCR_SCOPE_RECOVERY_RESULTS.md](OCR_SCOPE_RECOVERY_RESULTS.md).

## 13. 후속 실험 G: 이미지 기반 crop과 GT 경계 v4

### 실제 crop 비교와 채택 실패

단일 이미지 기반 방법을 사전에 고정했다. grayscale<160 mask에서 긴 연속 run을 노트 선 후보로 제거해 bbox를 추정하고3px 여백을 붙였다. 픽셀을 지우지는 않지만, crop으로 글자를 잘라낼 수 있다는 한계가 있다. 밝기·여백 sweep이나 진단 결과 기반 재튜닝은 하지 않았다.

validation 대표21 crop을 확인한 후, 동일244영역 전체를 원본/처리 조건에서 실제 추론했다. CER이 엄격히 낮을 때만 채택하고 동률/악화면 기존 crop을 유지하도록 고정했다.

| 비교 | 페이지·영역·정답 문자 | 원본 CER | 축소 후보 CER | 판단 |
|---|---|---:|---:|---|
| validation GT crop | 14페이지·244개·5,613문자 | **12.9877%** | 13.7003% | 후보 탈락 |
| 사전 선언한 진단 검출 crop | 16페이지·138개·2,501문자 | **15.3139%** | 15.5138% | 재선택 없이 진단만 |

validation 오류는729→769개, S505→507/D101→159/I123→103이었다. 완전 일치율은20.9016→23.3607%로 올랐지만 CER 기준을 사후 바꾸지 않았다. validation23영역 개선·22영역 악화·199영역 오류 수 동일, 진단10개 개선·11개 악화·117개 동일이었다.

**어두운 배경에서 실제 글자 잘림을 확인했다.** hf-0157-line-0026/0027에서 문장 앞부분이 mask/ruling 처리로 bbox에서 탈락해 전체 폭의22.8%,37.5%만 남았다. 오류는5→33,4→29로 늘어 두 줄에서53오류가 추가됐다. 이는 단순 수치 차이로 설명할 수 없는 물리적 정보 손실이다. 해당 방법은 운영에 적용하지 않았다.

과거 validation244의12.7917%는297개를 생성한 뒤 일부를 제외한 값이다. 이번에는244개 자체를 batch4로 묶어13개 문자열이 달랐다. 따라서 새 원본/처리 조건끼리 비교했으며 과거 캐시를 Before에 섞지 않았다. 같은 M138 원본 출력은 앞선 C1 캐시와 모두 일치했다.

### 별도 GT 경계 감사

mixed_keep30개와 교차하는 GT23개·7페이지를 모두 AI 시각 검토했다. bbox11개·4페이지를 수정했다. 자동 여백 제안5개와 원본을 보고 지정한 수동 경계6개이며, 머리말2개의 추가 축소 제안은 보류하고10개는 변경하지 않았다. 보류 제안은 기존 GT를 평가에서 제외했다는 뜻이 아니다.

GT v4는 **479개 전체 ID·전사·include·order를 유지하고11개 bbox만 변경**했다. 독립 사람 인증본이나 전체 inline 수식/본문을 새로 분할한 완전 GT는 아니다.

| 같은500개 인식문 /17페이지·318GT·5,982문자 | GT v3 | bbox 수정 GT v4 |
|---|---:|---:|
| ignore 후 채점 예측 | 351 | 346 |
| mixed_keep | 30 | 25 |
| 페이지 CER | 25.0084% | **23.6041%** |
| 공백 제외 CER | 25.6851% | 24.3149% |
| S / D / I | 541 /159 /796 | 541 /160 /711 |
| TP / FP / FN | 311 /40 /7 | 310 /36 /8 |

같은 인식문인데 GT 경계/ignore 판정으로 CER이1.4042%p 달라졌다. 취소선·행렬·불확실 영역5개가 불필요한 GT 교차에서 벗어났고, 타이트한 GT 때문에 IoU TP가1개 줄었다. **이 수치 하락은 모델/검출기 개선이 아닌 평가 geometry 민감도**다.

실제 GPU 입력은 validation496개와 진단276개, 합계772개였다. GT 경계 재채점은 기존500개 캐시를 재사용했다. 원본 모델·기존 crop을 유지했고 새 학습·운영 변경은 없다.

상세: [OCR_BOUNDARY_RESULTS.md](OCR_BOUNDARY_RESULTS.md).

## 14. 수치를 읽을 때 지켜야 할 구분

| 수치 변화 | 실제 달라진 것 | 해석 |
|---|---|---|
| GT crop235:21.6087→15.2238% | 같은 VL의 학습 가중치 | 최초 언어 LoRA의 실제 인식 개선 |
| legacy 페이지97.2865→84.7647% | 같은 검출/GT 범위에서 인식기 | 동일 VL 학습 전후 페이지 비교 |
| legacy84.7647→감사 GT v2 30.0615% | GT·범위·순서/ignore 정책 | 평가 기준 변경, 모델 개선 아님 |
| GT v2 30.0615→28.9025% | 기존 필터에서 제거된103개 복원 | 같은 모델의 coverage tradeoff; 삽입 증가/CI0포함 |
| GT v3 28.0843→25.0084% | 같은 v3 안에서397→500 bank | 별도 GT 조건의 coverage 비교 |
| GT v3 25.0084→v4 23.6041% | 같은500개 문자열,GT bbox11개 | 평가 경계 수정 효과, 모델 개선 아님 |
| validation12.9877→13.7003% | 같은244개에서 crop 변경 | 최신 이미지 기반 후보 실패; 기존 crop 유지 |

이 표의 모든 CER을 한 축의 ‘연속 모델 개선 그래프’로 연결하면 안 된다. GT235/199/306/318, 문자 수, 페이지 조립 방식, batch 구성과 정책이 다르다. 특히 **최종23.6041%를 최초21.6087%와 직접 비교하거나, 최신 모델의 GT crop CER이라고 표현하면 안 된다.**

검출·crop·범위·순서는 상호작용한다. 각 차이를 더해 원인별 정확한 기여율을 만들지 않았다. 예측문 길이와 Levenshtein 삽입 수, 동일 오류 수와 동일 출력 문자열도 구분했다.

## 15. 최신 검증·산출물·재현

최신 관련 회귀 테스트는 **68개 통과**다. 검증 대상은 데이터/분할·NFC·S/D/I·bbox 일대일 매칭·FP ID coverage·보류/ignore·crop 분모·순서 불변·adapter/설정/row drift·덮어쓰기 방지·validation 선택 등이다. 실제 GPU 재로딩/반복 추론·tensor/출력 일치도 별도 검증했다.

다만 전체 프로젝트 discovery는172개 집계 중7개 모듈 import 오류였다. 전용 LoRA 환경에서 `test_hybrid`, `test_hybrid_vector`, `test_image_upload`는 fastapi, `test_ocr_first_analysis`, `test_structure_rules`, `test_text_box_candidates`는 cv2, `test_relation_analysis`는 defusedxml이 없다. 전체 프로젝트가 통과했다고 주장하지 않는다. 테스트 통과가 모든 촬영 조건의 crop 글자 보존이나 GT 정답성을 보증하지도 않는다.

| 추가 산출물 | 위치 |
|---|---|
| 페이지 GT/FP201 감사·C0/C1/O0/O1/P0/P1 | `data/experiments/ocr_page_audit_v1/` |
| raw500 복원·GT v3·추가103 인식 | `data/experiments/ocr_page_audit_scope_recovery_v1/` |
| validation/진단 crop 비교·GT v4 | `data/experiments/ocr_page_audit_boundary_v1/` |
| 최신 결과·CSV·해시·로그 | [docs/ocr_boundary/](ocr_boundary/) |
| FP201 전수 화면 | [review.html](../data/experiments/ocr_page_audit_v1/review.html) |
| 복원103 전수 화면 | [restored103-gallery.html](../data/experiments/ocr_page_audit_scope_recovery_v1/restored103-gallery.html) |
| GT 경계23개 전체 화면 | [geometry-review.html](../data/experiments/ocr_page_audit_boundary_v1/geometry-review.html) |
| 최신 crop 실패 사례 | [crop-errors.html](../data/experiments/ocr_page_audit_boundary_v1/crop-errors.html) |
| 사람 검수37항목 화면 | [human-review.html](../data/experiments/ocr_page_audit_scope_recovery_v1/human-review.html) |

새 평가 코드는 `src/evaluation/ocr_page_audit*.py`, `ocr_scope_recovery*.py`, `ocr_boundary*.py`다. 원본/모델/processor/config/예측 hash, 실제 실행 소스 사본과 환경은 각 실험의 `completion-manifest.json`, `input-lock.json`, phase receipt에 보존했다. 새 경로에서 캐시 재채점·보고서 생성도 재현했다. 구체적인 단계별 명령은 각 상세 보고서에 있다.

```bash
# 최신 관련 회귀 테스트
.cache/venvs/ocr-vl-lora/bin/python -m unittest \
  tests.test_ocr_boundary tests.test_ocr_scope_recovery tests.test_ocr_page_audit \
  tests.test_ocr_clean_vision tests.test_ocr_train_audit tests.test_ocr_lora_round2 \
  tests.test_ocr_vl_lora tests.test_ocr_v2 tests.test_ocr_finetune \
  tests.test_ocr_finetune_page -v
```

## 16. 현재 유지 결정과 남은 작업

**유지:** 최초 VL 언어 LoRA, 기존 crop, 기존 운영 OCR/Qwen 경로. 추가 학습·고해상도·보간·정제 추가 학습·시각 LoRA·최신 밝기 기반 trim은 교체 근거를 확보하지 못했다. 모든 실제 후보 checkpoint와 실패 로그는 보존했다.

**확인한 문제:** CTC 사전 범위가 VL 페이지 평가에 잔존해 정상 줄을 제거했고, GT 누락/거친 bbox가 apparent FP와 삽입 평가에 영향을 줬다. 고정 밝기 trim은 어두운 배경에서 글자를 잘랐다. 정답 영역에서도 잔여 인식 오류가 있어 범위 정리만으로 충분한 운영 품질이 되지 않았다.

**미완료/불확실:** 독립 사람 GT 검수, 페이지 완전성 인증,13개 FP 보류,inline 수식/본문의 세밀한 분할,흐름도 의미 순서,어두운 배경을 포함한 crop 글자 보존,미사용·독립 필자 평가셋이다. AI 검토와 사람 검수를 혼동하지 않는다.

후속 투자에서 먼저 확인할 것은 **독립 사람이 검증한 전사·geometry와 혼합 영역 기준**, 그리고 **글자 보존을 보장할 수 있는 crop 처리 조건**이다. 이 필요 사항은 아직 실행 완료한 성과로 집계하지 않는다. 기존 진단 집합에서 점수가 좋아지도록 사후 임계값을 탐색하거나 새 학습을 추가하지 않았다.
