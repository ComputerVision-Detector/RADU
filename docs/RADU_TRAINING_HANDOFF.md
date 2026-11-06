# RADU Qwen3-VL-4B 구조 추출 학습 인수인계

마지막 갱신: 2026-11-07 (Asia/Seoul, 최신 내용은 아래 현재 작업 현황). 이전 기록: 이미지 전용 학습(A안) 결정, FC_A·hdBPMN 분할·normalized·partial prepare, PaddleOCR-VL 8GB 시험, 수식·표 데이터 조사까지 반영. 이 문서는 다른 AI/개발자가 작업을 이어받기 위한 현재 상태와 실행 순서를 기록한다. 최종 수량은 추적 가능한 요약인 [`RADU_DATA_ACQUISITION.json`](RADU_DATA_ACQUISITION.json), 로컬 `data/processed/radu_staging/acquisition_summary.json`, `verified_assets.jsonl`, `inventory.json`을 근거로 한다. RST 별도 증거는 `data/processed/radu_staging/ai2d_rst/rst_verification.json`이다. 과거 다운로드 도중 기록보다 이 최종 감사 결과를 우선한다.

## 사용자가 정한 목표와 역할

- 학습 목표(사용자 최신 지시): LoRA 튜닝한 Qwen3-VL-4B가 자연어를 제외한 **화살표, 도식, 수식, 표**를 잘 인식하게 한다. 이 중 수식·표는 현재 데이터로 학습할 수 없다(아래 조사 결과).
- 자연어 문자 전사와 글자 bbox는 **PaddleOCR-VL 1.6**이 담당한다(사용자 지정, `main` 39567dc의 spotting 경로).
- **A안 확정(사용자 결정):** Qwen 학습 입력은 이미지뿐이다. OCR 결과는 학습 입력이 아니며 추론 후 위치 기반 후처리로 병합한다. Qwen이 글자를 전사·교정하는 것은 학습 목표가 아니다. 병합 규칙은 아직 구현하지 않았다.
- **학습 순서(사용자 결정):** 화살표·도식 파일럿(FC_A, hdBPMN)을 먼저 진행한다. 파일럿은 8GB GPU, 메모리, 토큰, 저해상도 화살표 인식 검증용이며 **결과 adapter는 최종 모델로 쓰지 않는다.**
- **최종 학습(사용자 결정):** 화살표·도식·수식·표 전체 데이터를 한 번에 넣어 Qwen3-VL-4B 원본에서 새로 시작한다. 파일럿 adapter 위에 이어서 순차 학습하지 않는다(사용자가 망각 문제를 지적). 
- **학습 장비(사용자 확정):** 최종 학습은 RTX 4090 24GB에서 진행한다. 이 PC(RTX 4060 8GB)는 파이프라인 검증 전용이다. 8GB 제약 때문에 정답 형식이나 해상도를 줄이지 않는다.
- 기반 모델은 `Qwen/Qwen3-VL-4B-Instruct`, 튜닝 방식은 LoRA다. 현재 로컬 GPU가 8GB라서 별도의 4비트 QLoRA 프로필도 만들었다.
- 현재 대상 6개 데이터셋은 `hdbpmn`, `fc_a`, `ai2d_rst`, `didi`, `casia_ohfc`, `flowlearn`이다.
- 사용자의 최신 실행 지시는 “실제 데이터를 다운해서 학습을 위해 적재”하는 것이다. 단순 추천이나 빈 설정 생성에서 멈추지 말고 실제 파일·정답 구조·다운로드 검증 증거를 남긴다.
- 별도 감시 에이전트 1개가 구현/적재 검토를 맡았고, 이 문서는 사용자가 추가 요청한 문서 담당 에이전트가 작성했다.
- 이번 세션에도 읽기 전용 독립 감시 에이전트를 추가했다. 상시 실행이 아니라 단계별로 검토를 요청하는 방식이다. 1차 검토 결과는 아래 '독립 리뷰 기록'에 있다.

## 현재 작업 현황 (2026-11-07, 가장 최신)

### 현재 하는 작업

**학습 준비 완료, 본 학습 미시작.** 사용자가 직접 실행한다. 코드는 미커밋.

**데이터.** `data/processed/radu_final`, 16종, train 1,866,927(토큰 초과 IBEM 3장 제외), validation 2,541, test 2,547. 예상 사용은 37.5만 회, 서로 다른 이미지 약 16.6만 장.

| 분류 | 소스별 train |
|---|---|
| 도식 | hdBPMN 308, FC_A 194, FC_B 280, FC_Bscan 280, FA 132, hdFlowmind 619 |
| 수식 | CROHME 10,903, MathWriting 229,864, UniMER 922,064, HME100K 93,160, IBEM 3,718 |
| 표 | PubTabNet 498,172, FinTabNet.c 77,522, TabRecSet 29,132, cTDaR 351, READ ABP 228 |

**검증.**
- 전수 토큰 검사 실패 0, 최대 5,495토큰(상한 5,500).
- 최장 16개 1 step 통과(최대 6,016토큰 포함, 예약 peak 22,526 MiB).
- 실제 설정 60 step 통과(예약 peak 20,460 MiB, 1.435장/초 평가 포함, RSS 12.7GB).
- lora_B 144개 전부 갱신. 재개 L2 차이 0.0539, 노름 1.9534.
- 테스트 89개 통과. 감시 2차 검토에서 실행 차단·라벨 결함 없음.
- 수정 전 5종 정답 오버레이 육안 확인: 위치 일치. 재변환 후 다시 육안 확인했다는 기록은 없다.

**구현 완료.** hdFlowmind EXIF 회전 이미지 복구. 끝점 없는 화살표 bbox 채점은 단위 테스트만 통과했고 실제 평가 실행 전이다.

**독립 감시 1차 검토에서 나온 결함.** 직접 확인 후 코드를 수정했다.

1. 소스 간 동일 이미지: READ ABP와 cTDaR 100쪽은 READ를 남기고 cTDaR에서 제외. HME100K와 UniMER hwe 3,021장은 UniMER를 남기고 HME100K에서 제외.
2. IBEM 줄바꿈 수식은 조각마다 전체 LaTeX가 붙어 있어 해당 페이지 제외(전체 7,447쪽 중 1,097쪽).
3. hdFlowmind 같은 그림 변형이 split을 넘어감: 접두 동일, 번호 차 3 이내, dHash 6 이하를 같은 그룹으로 묶음.
4. 빈 셀을 생략한 표는 정답 불완전이라 제외(새 표 소스에 적용). cTDaR modern은 대부분 빠짐.
5. cTDaR 같은 이름 116건은 다른 이미지였음: id에 트랙 포함, 이름 기준 제외 삭제.
6. IBEM `\begin{equation*}` 제거, `align*`는 `aligned`로 변경.

추가 수정: HME100K 한자·번호 문장부호 포함 618행 제외. IBEM held-out을 적은 논문에서 뽑아 train 손실 축소.

**코드 변경.** `token_scan.py` 전수 검사 저장, 중복 없는 샘플러, 소스별 eval loss, 체크포인트 전체 보존. 설정은 `configs/qwen3_vl_4b_lora_final.json`. `infer_image.py`에 `--tasks`와 수식·표 표시 추가. VRAM 단편화 OOM이 있어 `train.py`에 expandable_segments를 넣었다.

### 해야 할 작업

순서대로 진행한다.

1. 사용자 학습 실행
2. 체크포인트별 소스 평가
3. 최종 평가는 `--max-new-tokens 6000`으로 실행

### 결정 사항

사용자 결정이다. 소스별 비중 값만 에이전트 제안이다.

- 최종 학습은 16종 소스, 37.5만 회, 도식 비중 40%. 예상 시간 약 2.5~3일.
- 해상도는 1MP 유지. 2MP 시험은 60 step에서 VRAM 부족으로 중단되어 해상도 가설은 미검증이다.
- 시작은 사용자가 직접 명령어로 실행한다(실시간 로그 확인 목적). **에이전트는 학습을 시작하지 않는다.**
- 소스별 5,000장 상한은 에이전트가 임의로 정한 값이었고 사용자가 거부했다. train은 전체를 후보로 넣고 샘플러가 비율대로 뽑는다. 검증·테스트는 소스별 최대 200장.
- 비중 제안(사용자가 도식 40%만 확정, 소스별 값은 에이전트 제안):
  - 도식: hdBPMN 13, FC_A 5, FC_B 5, FC_Bscan 4, FA 4, hdFlowmind 9
  - 수식: CROHME 5, MathWriting 6, UniMER 5, HME100K 8, IBEM 6
  - 표: PubTabNet 8, FinTabNet.c 5, TabRecSet 8, cTDaR 5, READ ABP 4
- 제외: WTW(사용자 결정, 손글씨 아님), PARES(샘플 라벨에 표 격자·수식·화살표 없음).
- CASIA는 신청 중이며 승인 전에는 제외한다. 비밀번호가 학습 초반 12시간 이내에 오면 재시작, 이후면 결과를 보고 결정한다.
- 사용 조건상 비상업이며 CASIA는 학술 연구 목적, 재배포 금지.
- 표 상한 변경(사용자 원칙: 답변 형식 축소 금지): 셀 좌표 유지 150셀, 격자 유지 400셀. PubTabNet과 FinTabNet.c도 같은 규칙을 적용한다.
- 토큰 상한: 최종 설정의 실제 상한은 5,500이다. 처음 전수 검사는 이전 설정 값인 6,900 기준으로 돌렸고, 최장 샘플 리허설에서 VRAM 여유가 작아 5,500으로 낮췄다. 그 결과 IBEM 3장만 제외됐고 정답 형식은 바꾸지 않았다.

### 미확인·위험

- hdFlowmind는 relations 없음, 끝점 대부분 없음. 분할 간 같은 그림 변형이 일부 남아 그 점수는 낙관적이므로 체크포인트 선택에 단독으로 쓰지 않는다.
- HME100K는 공식 split이 없어 같은 수식 기준으로 분할했다. 작성자 누수는 판단 불가, 출처와 사용 조건 미확인.
- cTDaR·READ ABP는 같은 책의 다른 쪽이 split을 넘는지 판단 불가(메타데이터 없음).
- PubTabNet에 격자 빈칸 표 1.3%가 남음.
- hdBPMN 등 도식은 장당 50~160회 반복되어 과적합 위험. 체크포인트별 평가로 최종 지점을 고를 계획.
- 평가 시 정답이 4,096토큰을 넘는 행이 있어 최종 평가는 `--max-new-tokens 6000`으로 실행한다.
- 장기 실행 메모리 증가는 미검증.
- 인식 성능은 미보장.

## 대량 데이터 확장 기록 (2026-11-06, 아래 내용보다 우선)

사용자 결정: 상업적 사용 계획이 없어 비상업 라이선스 데이터를 쓸 수 있다. 이 절의 수치는 RTX 4090 장비에서 측정했다. **전체 학습은 아직 시작하지 않았고, 수식·표 인식 성능도 아직 측정하지 않았다.**

**hdBPMN epoch 실험.** FC_A + hdBPMN으로 12 epoch를 돌렸다(`models/radu/exp_hd_epochs12`, git 제외). 판정은 부분 개선이다. 파싱 90% 이상, 노드 F1 0.5 이상 기준을 hdBPMN에서 충족하지 못했다. 해상도 영향은 아직 미검증 가설이다.

| validation | 모델 | 파싱 | 노드 | 화살표 | 관계 |
|---|---|---|---|---|---|
| FC_A 57 | 파일럿 | 56 | 0.949 | 0.694 | 0.757 |
| FC_A 57 | 12 epoch | 56 | 0.988 | 0.899 | 0.936 |
| hdBPMN 101 | 파일럿 | 39 | 0.113 | 0.117 | 0.017 |
| hdBPMN 101 | 12 epoch | 77 | 0.438 | 0.436 | 0.166 |

**추가 기능.**
- 스키마에 `formulas`(LaTeX), `tables`(행·열·span 격자, 셀 글자 제외) 과제를 추가했다. 도식 프롬프트는 바이트 단위로 그대로다.
- `normalize_formula_table.py`: CROHME 2023, UniMER, MathWriting, PubTabNet, FinTabNet.c 변환. 소스별 기본 상한 5000/200/200.
- `normalize_coco.py`, `build_coco_splits.py`: FC_B, FC_Bscan, FA 추가. FC_B와 FC_Bscan은 같은 그림이라 writer 그룹을 공유한다.
- `train.py`: `sampling_shares`, `samples_per_epoch`, `max_eval_samples`. `evaluate.py`: `--sample-per-dataset`, 수식·표 채점.
- `infer_image.py`: 이미지 1장을 주석 PNG와 JSON으로 출력. `--data` 없이 실행하면 도식 과제만 요청한다. `visualize_predictions.py`: 예측 HTML 갤러리.

**합본 `data/processed/radu_full`(git 제외, partial).** 설정은 `configs/qwen3_vl_4b_lora_full.json`이다.

| 소스 | train | validation | test | 학습 비중 |
|---|---|---|---|---|
| hdBPMN | 308 | 101 | 92 | 20% |
| FC_A | 194 | 57 | 92 | 8% |
| FC_B | 280 | 196 | 196 | 8% |
| FC_Bscan | 280 | 196 | 196 | 7% |
| FA | 132 | 84 | 84 | 7% |
| CROHME 2023 | 5,000 | 200 | 200 | 10% |
| UniMER | 5,000 | 200 | 200 | 7% |
| MathWriting | 5,000 | 200 | 200 | 8% |
| PubTabNet | 5,000 | 200 | 200 | 15% |
| FinTabNet.c | 5,000 | 200 | 200 | 10% |
| 합계 | 26,194 | 1,634 | 1,660 | 100% |

**전수 검사(29,488장).** 이미지 디코딩, sha256, split 간 바이트·픽셀·그룹 중복, 데이터셋 간 픽셀 중복, 정답 JSON과 빈 정답, 텍스트 필드 모두 문제 0건. 최대 5,316토큰으로 상한 6,900 초과 0건. 원본 감사는 28개 소스, 42,054파일, 문제 0건이며 일부 파일은 크기만 확인했다.

**1 step 검증.** `models/radu/smoke-full-1840`에서 rc=0, eval_loss 0.6293(300건, 45초), peak 예약 21,966 MiB. 16개 샘플만 봤으므로 긴 표의 메모리 peak는 증명되지 않았다. 전체 학습(26,000장 × 3 epoch, 4,875 step)은 1.45장/초 기준 약 15시간으로 추정한다.

**시작 전 알려진 부족한 점.**
- 수식 15,000장과 PubTabNet 5,000장은 객체가 이미지당 1개이고 bbox가 이미지의 90% 이상이라 페이지 내 위치 학습이 안 된다. 표 전체도 이미지당 1개다.
- 프롬프트가 도식, `formulas`, `tables` 세 조합뿐이어서 혼합 페이지 요청은 학습되지 않는다.
- 도식은 비중 50%에 이미지 1,194장(고유 914종)으로, 평균 33회, hdBPMN은 약 51회 반복된다.
- 표 소스 2개는 논문·재무 문서라 촬영·손글씨 표가 없다.

**미변환 소스.** 사용자가 A, B, C안 중 결정하기 전이다. 권장 A안은 기존 수식·표 상한을 올리고 hdFlowmind, IBEM, TabRecSet을 변환하며 추론은 요소별로 나눠 요청한다.
- 주석 파일 확인, 내용은 변환 전 확인 필요: hdFlowmind(VOC XML 1,776), IBEM, TabRecSet, cTDaR 2019, READ ABP, SciTSR, CISOL, AI2D-RST, CGHD.
- 위치 좌표 없음: DIDI(그래프만, 자동 정렬 미검증), SketikZ(TikZ), FlowVQA(Mermaid), FlowLearn(SciFlowchart는 그림 영역만, SimFlowchart 라벨은 미확인). 현재 형식으로는 학습 불가.
- 미확인: FC-Detection, HASYv2, MathWriting+.
- PARES: tar 해제에 9.35GiB 필요, 미해제. 라벨 내용 미확인.
- 사용자 제공 필요: WTW(`data/raw/radu/wtw/archives/`), HME100K(`data/raw/radu/hme100k/archives/`), CASIA 비밀번호. 필수는 아니다.
- 라이선스 명시 없음: FC_B, FC_Bscan, FA, cTDaR, SketikZ.

**재현 명령.** 합본을 다시 만들면 절대경로가 현재 장비 기준으로 기록된다.

```bash
python -m src.training.normalize_formula_table --dataset <crohme_2023|unimer|mathwriting|pubtabnet|fintabnet_c>
python -m src.training.prepare --output-dir data/processed/radu_full --allow-partial
HF_HOME=$PWD/.cache/huggingface HF_HUB_OFFLINE=1 .cache/venvs/train/bin/python -m src.training.train --config configs/qwen3_vl_4b_lora_full.json
```

## RTX 4090 이전 후 기록 (2026-11-05, 아래 내용보다 우선)

장비: Linux, RTX 4090 24GB. 이 절의 수치는 모두 이 장비에서 측정했다. 파일럿 adapter는 검증용이며 최종 모델이 아니다.

**환경.** `.cache/venvs/train`을 새로 설치했다(Python 3.11, torch 2.14.1+cu126, torchvision 0.29.1+cu126, transformers 5.22.1, peft 0.21.2). 기존 requirements에 없던 `torchvision`을 추가했고, `requirements-training.lock.txt`를 생성했다. torchvision은 반드시 cu126 빌드를 쓴다(PyPI 0.29.2는 임포트 실패).

**데이터.** 공개 소스를 다시 받았고 수량이 기존 기록과 일치한다: FC_A 419, hdBPMN 502, native 2,118파일, AI2D 4,903, DIDI 58,655, FlowLearn 23,857행, CASIA 2,957. 분할 매니페스트를 재생성하면 커밋본과 바이트까지 같다. 계획 파일은 `data/raw/radu/*.plan.jsonl`이다.
- 없음: RST ZIP(사용자 제공, 재전달 필요, `rst_annotations` 0), 상류 코드 ZIP(불필요).
- 기록보다 FC_A 2개(약 1.3MB), DIDI 2개(약 0.4MB) 적다. 4060 계획 파일이 없어 정체 미확인이며, 이미지와 샘플 수는 일치한다.
- PaddleOCR 환경 `.cache/venvs/ocr`는 없다.

**VRAM (최종 설정 BF16, `max_pixels` 1003520, 고정 샘플 1 step).** 토큰당 약 1.9 MiB. 상한을 8192에서 6900으로 낮췄다. 새 소스를 넣으면 다시 측정한다.

| 토큰 | 결과 | peak 할당 | peak 예약 |
|---|---|---|---|
| 5,083 (train 실제 최장) | 통과 | 18,405 MiB | 20,874 MiB |
| 5,983 | 통과 | 20,143 MiB | 20,776 MiB |
| 6,973 | 통과 | 22,053 MiB | 22,618 MiB |
| 8,188 | OOM | - | - |

**검증 완료.** 1 step 후 `lora_B` 144개 전부 갱신. 강제 종료 후 재개한 결과는 중단 없는 실행과 잡음 수준으로 일치(`lora_B` L2: 재개 대 전체 0.0375, 같은 설정 2회 0.0424, 이동량 0.136). 학습은 비트 단위로 재현되지 않는다. 재개할 때는 처음과 같은 `--max-steps`를 써야 한다(다르면 학습률 스케줄이 어긋남).

**파일럿.** `models/radu/pilot_fc_a_hdbpmn_v1`, 96 step, OOM 없음, peak 예약 22,580 MiB, validation loss 0.525→0.411. test 184건(`eval_test/`) 결과 F1:

| 구간 | 모델 | 파싱 | 노드 | 화살표 | 관계 |
|---|---|---|---|---|---|
| FC_A 92 | adapter | 90 | 0.931 | 0.709 | 0.755 |
| FC_A 구조 학습 73 | adapter | 71 | 0.940 | 0.721 | 0.772 |
| FC_A 구조 미학습 19 | adapter | 19 | 0.888 | 0.641 | 0.660 |
| hdBPMN 92 | adapter | 39 | 0.123 | 0.107 | 0.036 |
| FC_A 92 | 원본 | 82 | 0.011 | 0.003 | 0.000 |
| hdBPMN 92 | 원본 | 80 | 0.000 | 0.001 | 0.000 |

원본 점수는 해석에 주의한다. 프롬프트가 허용 `kind`를 알려 주지 않고, 원본은 0-1000 척도 좌표를 자주 출력한다. 파싱된 샘플만 사후 보정하면(종류 무시, 좌표 재척도) 원본 노드 F1은 FC_A 0.313, hdBPMN 0.322다. adapter의 종류 무시 노드 F1은 FC_A 0.983, hdBPMN 0.257이다. 즉 **hdBPMN에서는 adapter가 원본보다 위치를 잘 찾는다고 말할 수 없다.**

**hdBPMN 실패 분석.**
- 53/92건이 4096토큰까지 화살표를 반복 생성해 잘렸다(중앙값 정답 21개, 예측 84개). 39건은 `nodes`에 도달하지 못했다. 좌표가 120px 간격 격자로 이어지는 환각이다.
- 학습 이미지 12건도 파싱 5건, 노드 0.129, 화살표 0.133, 관계 0.036으로 test와 같다. 표본은 작지만 일반화보다 학습 부족 쪽이다.
- 미검증 가설: 축소(hdBPMN 중앙값 3.77MP, FC_A 1.14MP, `max_pixels` 약 1MP), 학습량 부족(도식당 요소가 FC_A의 2배 이상인데 hdBPMN 표본은 약 750회), 언어 LoRA만 학습. 해상도를 올리면 이미지 토큰과 VRAM이 늘어 상한을 재측정해야 한다.

**`evaluate.py` 수정.** 모델 출력의 리스트형 관계 endpoint로 평가가 중단되던 문제와, 정수 ID를 버려 원본 모델이 예측 0개로 채점되던 문제를 고쳤다. 테스트 52개 통과.

## 현재 상태

아래 표는 4060 시점 기록이다. 학습 코드 미검증, 실제 학습 미실행 항목은 위 절로 대체됐다.

| 구분 | 확인된 상태 |
|---|---|
| 공통 학습 계약 | `src/training/schema.py` 구현. 이미지 전용으로 변경: `ocr`·`text_links` 제거 |
| 6개 소스 레지스트리 | `configs/radu_datasets.json` 구현 |
| 구조 정답 검증 및 SFT 준비 | `src/training/prepare.py` 구현 |
| FC_A/hdBPMN COCO 변환 | `import_coco.py`, `normalize_coco.py` 구현. 실제 배포본 변환 완료, 수치 검사와 오버레이 검수(샘플 12장) 완료 |
| FC_A/hdBPMN 분할 | `build_coco_splits.py`, `configs/splits/fc_a_v1.json`, `hdbpmn_v1.json` 생성 |
| 나머지 4개 native 변환기 (AI2D-RST, DIDI, FlowLearn, CASIA) | 미구현 |
| 수식·표 과제 | 미구현. 현재 데이터로 학습 불가. 파일럿에서 제외하고 최종 학습 전에 데이터 조사·추가와 schema 확장 |
| 소스별 균형 sampling | `train.py`의 `balance_sources` 구현. 소스 2개로 과적합 시험에서 오류 없이 동작했고 6개 소스 효과는 미검증 |
| LoRA/QLoRA 학습 코드 | `src/training/train.py` 구현, 실제 GPU 학습 미검증 |
| 다운로드·해시·안전 압축 해제 | `src/training/acquire.py` 구현 |
| 원본·정답 staging 인덱스 | `src/training/stage_sources.py` 구현. normalized 학습 데이터와 별도 |
| 실제 파일 감사 | 원본 26,921개 파일, 6,400,059,348 bytes (약 6.400GB) 검증. 계획상 누락 0개 |
| 단위·통합 검사 | `tests/` 합계 50개 모두 통과 (GPU 학습 검증 아님) |
| OCR 환경 | `.cache/venvs/ocr` (Python 3.11, Paddle GPU 3.2.1, PaddleOCR 3.7.0, PaddleX 3.7.2) 설치·GPU 인식 확인. 가중치 `models/` 확보, 리비전은 `models/ocr_revisions.json` |
| 학습 환경·Qwen 가중치 | `.cache/venvs/train` (torch 2.14.1+cu126, transformers 5.22.1, peft 0.21.2, bitsandbytes 0.50.2). transformers는 `<5.23`으로 제한. 모델 revision `ebb281ec70b05090aa6165b016eac8ec08e71b17` 고정, 가중치는 `.cache/huggingface`(HF_HOME) |
| 토큰 실측 (844건) | 8GB 설정: 중앙값 1,568 / p95 3,003 / p99 3,850 / 최대 4,530. 프로세서 prefix 약 326토큰 포함. 4,096 초과 4건이라 상한을 4,608로 올림. 고해상도(`max_pixels` 1003520, 감시 측정): p99 4,636 / 최대 5,316, 8192 초과 0 |
| 실제 데이터 다운로드 | 6개 소스의 선택 원본 파일 모두 확보. AI2D-RST 코퍼스 1,000개는 원본 이미지/segmentation과 연결 완료. CASIA는 암호화 파일 확보 상태 |
| 학습용 normalized JSONL | FC_A 343건, hdBPMN 501건 생성. 나머지 4개 없음 |
| prepare 결과 | `data/processed/radu`: **partial**, train 502 / validation 158 / test 184. 2개 소스뿐이며 완전한 학습 데이터가 아님 |
| GPU 가중치 갱신 | **과적합 테스트로 첫 확인.** 학습 샘플 4건, 60 step에서 `lora_B` 144개 전부 0이 아님. 일반화 성능 증거가 아니다 (아래 기록). 최장 16건 1 step은 8GB에서 통과와 OOM이 갈려 신뢰할 수 없으므로 4090에서 검증한다 |
| 전체 preflight | 8GB 설정 통과: train 502, validation 158, LoRA 모듈 144. 최종 설정은 미실행 |
| 구조 평가 | `src/training/evaluate.py` 구현 (노드 IoU, 화살표 끝점, 관계 F1) |
| 실제 학습 | 실행하지 않음. 학습 성공 아님 |
| 진행 중 | 체크포인트 재개 확인, adapter와 원본 모델 추론 비교, 이후 BF16 실행 시도와 Linux 설치 검사 |
| 학습 파라미터 | 11,796,480개 (0.27%) |

**현재 결과는 “원본 확보·감사 완료 + FC_A·hdBPMN 2개 소스 파일럿 입력 생성”이다. 4개 소스 변환, 수식·표 데이터, GPU 검증이 남아 있으므로 “6개 데이터셋 학습 준비 완료” 또는 “학습 성공”으로 보고하면 안 된다.**

### 최종 확보·적재 상태

아래는 2026-11-04 최종 통합 무결성 감사와 6개 소스 staging 인벤토리 결과다. 26,921개는 감사한 원본 자산 파일 수이지 학습 샘플 수가 아니다. 압축 파일과 별도 배포 버전이 포함되며, 압축 해제·파생 이미지·캐시를 더한 전체 디스크 사용량과도 다르다.

| 소스 | 실제 확보·검증 결과 | 아직 완료되지 않은 일 |
|---|---|---|
| FC_A | 이미지 419개와 공식 정답 다운로드·staging 완료. 공식 train 248 / test 171. 노드 3,035 / 화살표 3,092 / 관계 3,092, 이미지 문제 0. `fc_a/source/` | `fc_a_v1` 분할·normalized 343건 완료. GPU 검증 |
| hdBPMN | COCO ICDAR2021 이미지 502개 (train 308 / validation 102 / test 92)와 정답 staging 완료. 노드 8,355 / 화살표 8,737 / 관계 8,737, 이미지 문제 0. 별도 native 배포 이미지 704개도 확보 | `hdbpmn_v1` 분할·normalized 501건 완료. native에만 있는 이미지는 미사용 |
| DIDI | 두 NDJSON 합계 2,263,317,106 bytes 및 프롬프트 TGZ 81,997,586 bytes 확보. 실집계 58,655개 (text 22,287 / no_text 36,368), train 43,995 / validation 7,330 / test 7,330, 원본 인덱스 완료 | 렌더링, 손그림-기호 정답 대응 검증. 프롬프트는 실제 localization GT가 아님 |
| FlowLearn | 선택 실이미지 23,848개 모두 기대 size/SHA 및 이미지 decode 검사 통과: Sci 3,848 unique + 합성 char 10,000 + word 10,000. 메타 포함 23,864개 파일, 1,872,692,555 bytes. catalog는 중복 참조를 보존해 23,857개 행 | 과학 원본 split 중복과 cross-split 이미지 1개 해결, native normalized 변환 |
| AI2D-RST | AI2D 이미지 4,903 + annotation 4,903 검증. 원본 arrows 34,991 / relationships 73,079. 사용자 제공 RST ZIP 1,302,664 bytes 검증·해제, 주석 1,000개가 원본 이미지/segmentation 1,000개와 모두 연결. 이미지 문제·누락 0 | RST 그룹/연결 native 변환, split 검토. connectivity null 222개는 미주석이며 빈 정답으로 학습 금지 |
| CASIA-OHFC | release ZIP 224,060,550 bytes 확보. 암호화 파일 2,961개 중 InkML 2,957개 인덱스 생성. template ZIP 암호화 601개와 신청서 PDF 확보 | 신청서 미제출, 비밀번호 미확보. 사용자 측 연구실 신청 절차 후 해제·렌더링·정답 검증 가능 |

초기 다운로드 중 작은 파일 7개 실패는 재다운로드로 복구했다. FlowLearn 고정 commit은 `205b467eaf541846cf36469d17d62e8f81be7be1`이다. 선택 실이미지는 Git LFS 원본 CAS에서 검증·복사까지 완료했고 `data/raw/radu/flowlearn/source/`가 원본 적재 위치다. 남아 있던 캐시 checkout의 불필요한 LFS pull만 중단했으며, 검증된 raw 데이터는 정상이다. 작업 재개 시 `.cache` checkout 완료 여부에 의존하지 말고 **raw 파일 + `verified_assets.jsonl` + 최종 summary**를 확인한다.

**`GIT_LFS_SKIP_SMUDGE=1`로 받은 clone이나 sparse checkout 성공만으로 실이미지를 받았다고 판단하지 않는다.** 파일 앞부분이 `version https://git-lfs.github.com/spec/`이고 크기가 약 130 bytes이면 LFS pointer다. 선택 바이너리를 실제 받거나 원본 LFS CAS의 기대 SHA256 객체를 검증·복사해야 한다. 이번 raw 원본에는 이 검증·복사가 완료됐다. 감사 코드는 pointer를 실패로 처리한다. Git LFS 버전은 3.8.1로 확인했다. 429를 피하려고 서버 제한을 우회하지 말고, 개별 요청 중단·공식 bulk 경로·재시도 간격을 사용한다.

### FlowLearn 실제 레코드 수와 분할 문제

- 확보한 고유 이미지: 23,848개. 과학 3,848개, 합성 char/word 각 10,000개다.
- 최종 catalog는 source train/test의 참조를 보존해 23,857개 행이다. 합성 20,000개 + 과학 참조 3,857개이며, 전체 split 참조 수는 train 19,337 / test 4,520이다.
- 과학 train은 3,337개 참조에 3,329개 unique 이미지, test는 520개 참조다. 원 배포의 `all_w_ocr` 3,858개 레코드를 고유 이미지 수 3,848로 잘못 표현하지 않는다.
- 과학 이미지 `214727773-Figure4-1.png`는 공식 train과 test 양쪽에 존재한다. 그 밖의 split 내부 중복 참조도 있다. 전체 목록은 `inventory.json`의 `scientific_duplicate_references`와 `scientific_cross_split_images`에 보존했다.
- 원본을 임의로 조용히 deduplicate하지 않았다. normalized/SFT 준비 전 이미지/원본 논문 그룹 단위 dedup와 split 정책을 명시하고 새 split 버전을 만든다.

### AI2D-RST 사용자 제공 파일 검증·적재 완료

- 원본 ZIP: `data/raw/radu/ai2d_rst/archives/ai2d-rst-v1-1.zip`, 1,302,664 bytes.
- 로컬 SHA256: `eb11d67507e08eb9bfd0f5944da7ca32cfcffa13e119b04ac5054effa65a759a`.
- 해제본: `data/raw/radu/ai2d_rst/ai2d-rst-v1-1/json/ai2d-rst/`.
- ZIP CRC 검사 정상, 해제 파일 1,002개 모두 ZIP 내부 내용과 SHA 일치. 배포처가 제공한 별도 digest는 없으므로 이 해시는 로컬 보관·내용 일치 검증 증거다.
- RST 주석 1,000개 ↔ 원본 PNG/segmentation 1,000개 join 완료, missing 0. 그룹/연결의 요소 참조도 검증했다.
- 연결 주석 present 778개, null 222개. null은 원본 값을 유지했고 `[]`로 바꾸지 않았다. **미주석과 완전한 빈 그래프를 구분해야 한다.**
- RST connectivity edge 6,585개: directional 3,362 / undirectional 3,197 / bidirectional 26. 원본 AI2D relationships 73,079와 별도 계층이므로 합쳐서 같은 정답 수로 세지 않는다.
- `data/processed/radu_staging/ai2d_rst/rst_catalog.jsonl` 1,000개 행 생성. 기존 원본 catalog 4,903개 중 1,000개에 RST 주석 경로를 연결했다.
- 검증 증거: `data/processed/radu_staging/ai2d_rst/rst_verification.json`.
- 실물 v1.1 배포 README는 **CC BY-SA 4.0**을 명시하며 GitHub README의 CC BY 4.0과 다르다. 이 불일치를 레지스트리·source provenance에 기록했다. 실제 사용 시 실물 배포 조건을 확인한다.
- 이전에 받은 `upstream-repository-v1.1.zip` 18,968,348 bytes는 코드 48개 파일(디렉터리 포함 62개 entry)이며 이번 사용자 제공 RST 코퍼스와 다른 자료다.

### FC_A·hdBPMN 검수 결과와 분할 v1

확인된 사실:

- **FC_A의 `N`과 `Nb`는 구조가 같다.** 파일명 `b` 템플릿은 문구가 다르지만(문구 유사도 N 내부 0.80, N↔Nb 0.11) 14쌍 중 13쌍은 도형·연결 구성이 일치하고 7번은 도형 구성만 일치한다. 또한 family 1과 6은 같은 그래프라 별칭으로 처리한다. 공식 test의 그래프가 train에 있는 셈이라 템플릿 분리 평가로 볼 수 없다. 일치 판단은 템플릿별 최빈 구성 비교이며 그래프 동형 검사는 아니다.
- FC_A 공식 train/test는 작성자 10명이 겹친다. hdBPMN 공식 분할은 작성자가 겹치지 않고, 연습문제 10개는 모든 분할에 있다.
- **COCO keypoint 이름이 실제와 반대다.** 메타데이터는 `['head','tail']`이지만 첫 점이 꼬리, 둘째 점이 화살촉이다. `arrow_prev`/`arrow_next` 대조 11,829개 중 11,828개 일치·1개 동률, 샘플 12장 오버레이도 일치. 따라서 `start=keypoints[:2]`, `end=keypoints[3:5]`가 맞다.
- 바이트 동일 중복 2쌍: FC_A `writer12_6.png`/`writer12_6_.png`, hdBPMN `ex00_writer0067.jpg`/`ex01_writer0067.jpg`. 각각 뒤쪽 파일을 제외했다. 바이트는 다르지만 거의 같은 이미지인 FC_A `writer8_1`도 제외했다.
- hdBPMN native 704장과 COCO 502장은 파일명 501개가 겹치지만 해시는 전부 다르다. 재인코딩본으로 보이며(픽셀 비교는 안 함) 같은 이미지가 중복 학습되지 않도록 COCO만 사용했다.
- bbox 범위 이탈과 면적 0 annotation은 두 소스 모두 없다.

| 매니페스트 | train | validation | test | 제외 |
|---|---|---|---|---|
| `configs/splits/fc_a_v1.json` | 194 | 57 | 92 (구조 미학습 19 / 학습 73) | 76 |
| `configs/splits/hdbpmn_v1.json` | 308 | 101 | 92 | 1 (중복) |

- FC_A 정책: 작성자 단위로 분할(seed 42), `N`+`Nb`를 한 family로 묶고 family 2, 3, 12를 test 전용으로 보류한다. 결과는 test의 구조 미학습/학습 구간을 나눠 보고한다. 보류 family를 train·validation 작성자가 그린 이미지는 제외했다.
- 제한: 보류 family 2는 도형 종류를 빼면 family 10과 그래프가 같다(구조 미학습 19장 중 6장). 라벨을 포함하면 다르다. 구조 미학습 결과를 해석할 때 고려한다.
- hdBPMN 정책: 공식 작성자 분할 유지. 작성자 누수가 없고 FC_A처럼 데이터를 버릴 근거가 확인되지 않았다는 판단이며, 연습문제별 구조 일관성은 측정하지 않았다. 연습문제 번호는 평가용으로 기록했다.
- 알려진 손실: hdBPMN의 `sequenceFlow`/`messageFlow`/`dataAssociation`은 모두 `arrows`로 합쳐지고 `directed: true`로 기록된다. 화살표 종류 구분은 학습하지 않는다.

### PaddleOCR-VL 1.6 시험 (추론 후 병합용, 학습 입력 아님)

- 대상: `src/training/run_ocr.py`로 FC_A 6장 + hdBPMN 6장. RTX 4060 8GB에서 **12/12 성공**. 장당 FC_A 2~6초, hdBPMN 6~10초. 846장 약 1.5시간은 이 표본으로 계산한 추정치다.
- 기존 worker의 여유 VRAM 10GB 조건 때문에 `RADU_VL_MIN_FREE_MB=6000` 설정이 필요하다(기본값 10000은 유지). 시험 시 여유 VRAM 약 6.8GB.
- **최대 VRAM과 정확도는 측정하지 않았다.** 결과는 `data/processed/radu_ocr/paddleocr_vl16.jsonl`.
- 관찰된 출력 문제(병합 단계에서 처리 필요): 짧은 글자의 수식 감싸기(`\( A \geq B \)`, `\( N \)`), HTML 이스케이프(`&lt;`), 손글씨 오인식(`shout`, `lnd`로 읽음), hdBPMN의 도형 내 글자 줄 분리(`Take`, `order`).
- `text_box_candidates.py`는 라벨링 카탈로그 전용이라 재사용하지 않았다. `run_ocr.py`는 같은 worker 로더·추론과 좌표 변환 함수를 쓴다.

### 수식·표 데이터 조사 결과

- 확인됨: FC_A(도형 5종 + text + arrow), hdBPMN(BPMN 12종 + flow 3종), AI2D 주석 키(`arrowHeads`, `arrows`, `blobs`, `containers`, `imageConsts`, `meaningfulSpaces`, `relationships`, `text`) 어디에도 수식·표 구조 정답이 없다. FlowLearn 과학 부분은 완전한 구조 정답이 없다.
- FC_A 글자 영역 3,194개 중 1,079개에 수학 기호(`=`, `<`, `>`, `+`, `*`, `^`, `sqrt`)가 있으나 글자 박스일 뿐 수식 구조 정답이 아니다.
- 미확인: CASIA(암호화), FlowLearn 합성본 내부 라벨, DIDI 라벨 내용.
- **결론: 사용자 목표 중 수식·표는 현재 데이터로 학습할 수 없다. 사용자 결정에 따라 파일럿에서는 제외하고, 최종 학습 전에 데이터 조사·추가와 과제(schema) 확장을 진행한다.**

`stage_sources.py`는 원본/이미지/소스 구조 정답을 인덱싱하는 별도 적재 단계다. 출력은 `data/processed/radu_staging/`이고 `training_ready: false`를 명시한다. 이 단계 산출물의 `ocr: null`과 “Run actual OCR” 안내는 A안 이전 기록이다. 이것은 `prepare.py`가 받는 `normalized.jsonl`이 아니다. DIDI DOT 프롬프트나 과학 논문 문맥을 손그림/화살표의 완전한 GT라고 주장하지 않는다.

## 코드 위치와 기존 파이프라인의 차이

| 경로 | 역할 |
|---|---|
| `src/training/schema.py` | 데이터셋 ID, 원본 좌표, 과제별 complete coverage, 참조 무결성 검증, 이미지 전용 프롬프트 |
| `src/training/import_coco.py` | COCO 데이터와 분할 sidecar를 normalized 레코드로 변환. 좌표는 감싸는 정수 bbox |
| `src/training/build_coco_splits.py` | FC_A·hdBPMN 버전별 분할 매니페스트 생성, 중복 제외, 그룹 누수 검사 |
| `src/training/normalize_coco.py` | COCO + 분할 매니페스트로 `data/raw/radu/<source>/normalized.jsonl` 생성 |
| `src/training/run_ocr.py` | PaddleOCR-VL 1.6 일괄 실행. 재개 가능, 실패는 오류로 기록. 학습 입력이 아니라 추론 후 병합용 |
| `src/training/evaluate.py` | 구조 평가: 노드 IoU, 화살표 끝점, 관계 F1 |
| `scripts/train_rtx4090.sh` | RTX 4090 학습 실행 스크립트. 실행 미검증 |
| `configs/splits/` | 분할 매니페스트 (`fc_a_v1.json`, `hdbpmn_v1.json`) |
| `configs/category_maps/` | 원본 category를 node/arrow/text로 대응 |
| `src/training/prepare.py` | 원본 이미지·분할·그룹·중복 검증 후 prompt/completion JSONL 생성 |
| `src/training/train.py` | Qwen3-VL 로드, 언어 attention LoRA, 응답 부분 loss, preflight, 학습/체크포인트 저장 |
| `src/training/acquire.py` | JSONL 다운로드 계획 실행, 재실행 시 검증, SHA256/MD5/Git blob SHA1 기록, 안전 압축 해제 |
| `src/training/stage_sources.py` | 실제 원본 자산과 source 정답의 staging/catalog 생성. 학습 준비 미완료를 명시 |
| `src/training/audit_acquisition.py` | 다운로드 계획과 실제 파일을 대조해 size/hash·이미지 verify를 검사하고 LFS pointer를 거부 |
| `configs/radu_datasets.json` | 소스 URL·라이선스·실제 status·raw_root·staging_catalog·normalized 예정 위치·인벤토리 링크 |
| `docs/RADU_DATA_ACQUISITION.json` | Git 추적 가능한 compact 최종 증거: 다운로드 수량·bytes·누락·staging·24개 검사 결과 |
| `configs/qwen3_vl_4b_lora.json` | **최종 학습 설정** (RTX 4090 24GB, BF16 LoRA). 아직 실행 검증 안 됨 |
| `configs/qwen3_vl_4b_qlora_8gb.json` | RTX 4060 8GB **smoke test 전용** 설정. 최종 학습에 쓰지 않음 |
| `requirements-training.txt` | Paddle와 분리한 학습 의존성 범위, 검증된 lockfile은 아님 |
| `requirements-training-qlora.txt` | 학습 의존성 + bitsandbytes |

기존 `src/hybrid/pipeline.py`는 Qwen2.5-VL OCR 교정 경로다. 새 도형·좌표 생성 금지, 일반 문자 수식 출력 등의 기존 지시가 있으므로 새 구조 추출 학습 경로와 혼동하지 않는다. 기존 웹 검수·OCR 경로는 이번 구현으로 교체하지 않았다.

기존 웹 문서의 허용 타입과 연결 제약도 다르다. 학습 결과를 웹에 연결하려면 normalized 구조에서 편집기 문서로 변환하는 adapter가 별도로 필요하다.

## 실제 실행 환경

- OS/셸: Windows, PowerShell.
- 로컬 GPU: NVIDIA GeForce RTX 4060, 약 8GB (`nvidia-smi`에서 8188 MiB 확인).
- `python`/`git` 명령은 원래 PATH에서 찾지 못했다. 실제 사용 가능한 Python 실행 파일은 사용자 홈의 `.local/bin/python3.15.exe`다. 이후 기존 Git 설치의 `$env:ProgramFiles/Git/cmd/git.exe`를 발견해 FlowLearn bulk clone에 사용했다.
- 테스트용 Pillow는 저장소의 `.cache/training-test-site`에 `pip --target`으로 설치했다. 이 테스트 설치가 학습 가상환경 설치를 의미하지 않는다.
- OCR과 학습은 환경을 분리한다: `.cache/venvs/ocr`, `.cache/venvs/train` (git 제외, Python 3.11). `HYBRIDOCR_WORK_LOG.md`의 RTX 4090 24GB 환경과 이 PC는 다르다.
- 학습은 Python 3.11 별도 가상환경을 권장한다. 현재 Python 3.15 테스트 성공이 PyTorch/Transformers/bitsandbytes GPU 조합의 호환성을 증명하지 않는다.
- 학습 환경 구축 후 CUDA 인식, BF16 지원, bitsandbytes 4bit 로드, 실제 1 step forward/backward를 확인한다. 패키지 버전을 고정하고 `pip freeze`를 보존한다.

현재 테스트 재실행 (가상환경에 Pillow가 설치되어 있으면 첫 번째 명령으로 충분):

```powershell
python -m unittest tests.test_training tests.test_acquire tests.test_source_staging tests.test_coco_normalize -v
```

현재 임시 테스트 환경을 그대로 쓰는 경우:

```powershell
& "$env:USERPROFILE/.local/bin/python3.15.exe" -c "import sys,unittest; sys.path.insert(0,'.cache/training-test-site'); result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(['tests.test_training','tests.test_acquire','tests.test_source_staging','tests.test_coco_normalize'])); sys.exit(not result.wasSuccessful())"
```

## 데이터 적재 경로와 확인할 증거

다운로드 데이터는 `.gitignore`에 의해 Git에서 제외되는 `data/raw/radu/`에 둔다. 모델·체크포인트는 `models/`에 둔다. 데이터 원본이나 가중치를 소스 저장소에 커밋하지 않는다.

```text
data/raw/radu/
  coco.discovery.json         GitHub COCO 변환 저장소 tree 탐색 결과
  hdbpmn.discovery.json       hdBPMN 원본 저장소 tree 탐색 결과
  flowlearn.discovery.json    Hugging Face tree 탐색 결과 (초기 응답은 페이지 일부일 수 있음)
  didi.discovery.json         GCS 상위 객체 목록
  casia.source.html           CASIA 공식 안내 페이지 보관본
  <download-plan>.jsonl       실제 다운로드할 URL·상대경로·예상 크기/해시
  acquisition_receipts.jsonl  실제 파일별 다운로드/검증 성공·실패 기록
  hdbpmn/
  fc_a/
  ai2d_rst/
  didi/
  casia_ohfc/
  flowlearn/

data/processed/radu_staging/
  inventory.json              실제 6개 원본/소스 정답 적재 인벤토리 (생성 완료)
  acquisition_summary.json    소스별 기대/존재/검증 파일 수·문제 집계 (최종 감사 완료)
  verified_assets.jsonl       개별 원본의 감사 상태·해시·이미지 크기
  fc_a/staged.jsonl
  hdbpmn/staged.jsonl
  didi/catalog.jsonl
  flowlearn/catalog.jsonl
  ai2d_rst/catalog.jsonl
  casia_ohfc/catalog.jsonl
```

위 탐색 메타데이터가 존재하는 것은 원본 데이터 다운로드 완료 증거가 아니다. `acquisition_receipts.jsonl`은 append-only라 재실행 시 같은 경로의 여러 기록이 있을 수 있다. 최종 경로별 최신 상태와 실제 파일 해시로 완료 수를 계산한다. `.part`는 완료 파일로 세지 않는다.

최종 감사한 원본 자산 합계는 `docs/RADU_DATA_ACQUISITION.json`의 `downloaded_files: 26921`, `downloaded_bytes: 6400059348`이고 `missing_assets`는 비어 있다. RST ZIP은 확보·연결 완료다. CASIA는 missing이 아니라 **파일 확보 완료/암호 해제 불가** 상태다.

다운로드 유틸리티 사용법 (실제 생성된 plan 파일명을 넣는다):

```powershell
python -m src.training.acquire download data/raw/radu/<download-plan>.jsonl --root data/raw/radu --workers 12
python -m src.training.acquire extract data/raw/radu/<archive>.zip data/raw/radu/<new-extraction-directory>
python -m src.training.stage_sources --root data/raw/radu --output data/processed/radu_staging
python -m src.training.audit_acquisition data/raw/radu/<download-plan>.jsonl --root data/raw/radu --output data/processed/radu_staging
```

- 계획의 `path`는 `--root` 기준 상대경로다.
- GitHub tree SHA와 commit SHA는 서로 다르다. 재현 가능한 raw URL은 실제 commit revision으로 고정하고 각 blob SHA 검증을 병행한다.
- 압축 해제 대상 폴더가 이미 있으면 중단한다. 기존 데이터 디렉터리를 임의로 삭제해 다시 풀지 않는다.
- 암호화 ZIP은 공개 다운로드와 학습 사용 가능 여부를 구분한다. 비밀번호를 추측하거나 우회하지 않는다.

## 데이터셋별 확보·변환 시 주의점

| 소스 | 공식 배포/선택 형식 | 확인된 원본 단서 | 적재 후 해야 할 일 |
|---|---|---|---|
| hdBPMN | [원본](https://github.com/dwslab/hdBPMN), [COCO 변환본](https://github.com/bernhardschaefer/handwritten-diagram-datasets) | 원본 현재 tree의 이미지 704개, COCO `hdBPMN-icdar2021`은 별도 이전 배포. 서로 같은 버전으로 간주하지 말 것 | COCO 경로 완료. native에만 있는 이미지는 BPMN 좌표 대응 검증 전까지 미사용 |
| FC_A | [COCO 변환본](https://github.com/bernhardschaefer/handwritten-diagram-datasets) | 실제 디렉터리는 `datasets/fca/`; train/test JSON만 있고 val JSON 없음. 공식 분할 작성자 10명 중복 실확인 | `fc_a_v1` 완료. test는 구조 미학습/학습 구간을 나눠 평가 |
| AI2D-RST | [RST 안내](https://github.com/thiippal/AI2D-RST), [원본 AI2D](https://registry.opendata.aws/allenai-diagrams/) | AI2D 4,903개 이미지/정답 검증 완료. RST 코퍼스는 사용자 제공 ZIP으로 확보. Zenodo 18MB는 upstream 코드 48개 파일(디렉터리 포함 62개 entry)뿐 | RST ZIP 확보·join 완료. directed/undirected 구분, 그룹 노드 표현 결정, 변환기 작성. RST 담화 관계를 흐름 관계로 섞지 않기 |
| DIDI | [공식 코드·데이터 안내](https://github.com/google-research/google-research/tree/master/didi_dataset) | GCS에 `diagrams_20200131.ndjson` (1,222,429,136 bytes), `diagrams_wo_text_20200131.ndjson` (1,040,887,970 bytes), `diagrams_20200131_prompts.tgz` (81,997,586 bytes) 공개 객체 확인 | 실제 레코드/split 집계, 궤적 렌더링, 프롬프트 label_id 대응 확인. DOT/XDOT 프롬프트 좌표를 실제 손그림 bbox GT로 사용하지 않기 |
| CASIA-OHFC | [공식 안내](https://nlpr.ia.ac.cn/databases/CASIA-OHFC/) | 공개 release ZIP·template ZIP·영문 신청서 링크 존재. 본 데이터 암호는 연구실 신청 절차와 연관 | 아카이브·신청서 확보와 암호 해제/학습 허가를 구분. 그룹 책임자 서명·제출은 사용자 측 절차. 에이전트는 서명·메일 발송 안 함. 승인 후 InkML 좌표변환/기호 인스턴스 검증 |
| FlowLearn | [Hugging Face](https://huggingface.co/datasets/jopan/FlowLearn), [공식 코드](https://github.com/Jo-Pan/FlowLearn) | repo 전체 tree는 확인했고 선택 실이미지 23,848개와 메타데이터 16개를 최종 검증. 과학 unique 3,848와 참조 3,857를 구분 | 실제 image/parsed/Mermaid 대응 확인, 중복·공식 cross-split 이미지 처리, 문서·template 그룹 분리. Mermaid만으로 화살표 픽셀 경로 GT를 만들어내지 않기 |

라이선스는 레지스트리와 원 배포본을 함께 보존한다. hdBPMN/DIDI는 CC BY 계열, FC_A/FlowLearn은 비상업 조건, AI2D 원본과 RST 주석은 별도 조건, CASIA는 연구용 동의 절차가 있다. 다운로드가 가능하다는 이유로 모든 소스의 사용 조건이 같다고 처리하지 않는다.

## normalized 학습 계약

현재 지원 과제는 `nodes`, `arrows`, `relations`다. 입력은 이미지뿐이며 `ocr`와 `text_links`는 A안에 따라 제거했다. **수식·표 과제는 아직 구현하지 않았다.** CROHME/MathWriting은 현재 선택한 6개에 포함하지 않았다.

각 normalized 레코드 필수 정보:

- `id`, `dataset`, `image` (해당 normalized JSONL 부모 폴더 기준 상대경로)
- 원본 픽셀 `width`, `height`
- `source_revision`, `annotation_origin` (`human`, `source_annotation`, `synthetic_program` 중 하나)
- 명시적 `split`: `train`, `validation`, `test`
- 비어 있지 않은 전역 `groups` 키 목록: 작성자, 원본 문서, template 등 적용한 누수 방지 기준을 기록
- `tasks`: 이 샘플에서 **완전한 정답이 있는 과제만** 선택
- `coverage`: 모든 선택 과제는 `complete`
- `target`: 선택한 과제의 키만 포함

출력 필드:

```json
{
  "nodes": [{"id": "n1", "bbox": [10, 10, 100, 80], "kind": "process"}],
  "arrows": [{"id": "a1", "bbox": [100, 40, 150, 45], "start": [100, 42], "end": [150, 42]}],
  "relations": [{"id": "r1", "source": "n1", "target": null, "directed": true, "arrow_id": "a1"}]
}
```

이 예시는 형식 설명이며 실제 정답이 아니다. ID는 이미지별 읽기 순서 `n1`/`a1`/`r1`이고, 원본 annotation은 `source_ids`로 추적한다. `target`에 글자 전사 문자열을 넣지 않는다. 글자와 도형의 연결은 추론 후 병합 단계에서 계산한다. 관계 endpoint는 포함된 노드/화살표를 참조하며, unresolved이면 `null`이다.

**미주석 관계는 `relations: []` 정답으로 학습시키지 않는다.** 해당 과제를 `tasks`, `coverage`, `target`에서 모두 제외한다. 기하 추정 결과나 다른 모델 예측은 원본 정답과 구분하며 검증 없이 `source_annotation`으로 둔갑시키지 않는다.

COCO importer는 `category` 문자열과 표준 `category_id`를 모두 지원한다. `arrow_prev`/`arrow_next`는 annotation ID다. `arrow_geometry`가 sidecar에 없으면 start/end는 null이다. FC_A·hdBPMN은 `normalize_coco.py`가 검증된 keypoint 순서로 채운다.

배포 COCO는 train/val/test JSON별로 `image_id`를 다시 시작한다. importer는 source partition을 포함하는 기본 sample ID (`<source_partition 또는 split>:<image_id>`) 또는 명시적 sidecar `sample_id`를 사용하도록 수정했고 회귀 검사를 추가했다. 별도 파일을 합칠 때 숫자 image ID만 전역 ID로 사용하지 않는다.

## 학습 준비 및 실행 순서

완료:

- 다운로드 감사·staging. CASIA는 사용자 연구실 신청/암호 해제 대기
- FC_A·hdBPMN 분할 정책, 시각·수치 검수, normalized, partial prepare

남은 순서:

1. Python 3.11 GPU 학습 환경과 Qwen3-VL-4B 모델 준비, commit SHA 고정: 완료.
2. 토큰 길이 실측: 완료 (상한 4,608). 정답을 자르지 않는다.
3. 가중치 갱신 확인: 과적합 테스트로 완료. 체크포인트 재개와 adapter·원본 모델 추론 비교는 진행 중이다.
4. FC_A·hdBPMN 파일럿으로 8GB GPU, 메모리, 토큰, 저해상도 화살표 인식을 검증하고 원본 모델과 구조 지표를 비교한다. 결과 adapter는 검증용이며 최종 모델로 쓰지 않는다.
5. RTX 4090으로 이전한다(아래 '이전 시 주의'). `configs/qwen3_vl_4b_lora.json`으로 preflight와 1 step을 먼저 실행해 이 PC에서 증명할 수 없는 항목을 확인한다.
6. 수식·표 데이터를 조사·추가하고 과제 스키마를 확장한다.
7. AI2D-RST, FlowLearn 합성본 변환기를 작성한다. DIDI는 정답 대응 검증부터, CASIA는 비밀번호 확보 후 진행한다. 새 소스마다 누수 그룹과 분할을 정한다.
8. 소스별 균형 sampling이 6개 소스에서 의도대로 동작하는지 검증한다. (구현 완료)
9. 최종 학습: RTX 4090에서 화살표·도식·수식·표 전체 데이터를 한 번에 넣어 Qwen3-VL-4B 원본에서 새로 시작한다. 파일럿 adapter에 이어서 학습하지 않는다.
10. 추론 후 PaddleOCR-VL 결과와 구조를 병합하는 규칙을 구현하고 평가한다.

normalized 준비 후 명령:

```powershell
# 이미 실행함. 분할·normalized·prepare 출력은 덮어쓰지 않으므로 재생성 시 새 버전/경로를 사용한다.
python -m src.training.build_coco_splits
python -m src.training.normalize_coco --dataset fc_a --split-manifest configs/splits/fc_a_v1.json --category-map configs/category_maps/fc_a.json --revision ecf548411d87d2ac78c8fd9f2e65ee1c961778ca
python -m src.training.normalize_coco --dataset hdbpmn --split-manifest configs/splits/hdbpmn_v1.json --category-map configs/category_maps/hdbpmn.json --revision ecf548411d87d2ac78c8fd9f2e65ee1c961778ca
python -m src.training.prepare --registry configs/radu_datasets.json --output-dir data/processed/radu --allow-partial
# 아직 실행하지 않음
python -m src.training.train --config configs/qwen3_vl_4b_qlora_8gb.json --preflight-only
python -m src.training.train --config configs/qwen3_vl_4b_qlora_8gb.json --max-steps 1
```

위 명령은 필요한 데이터/학습 환경이 준비된 이후에 실행한다. `prepare` 기본은 6개 소스를 모두 요구한다. 일부 데이터 파일만 있는 파일럿에는 `--allow-partial`을 명시하며 partial 학습임을 보고한다. 기존 출력 디렉터리를 덮어쓰지 않으므로 새 버전 디렉터리를 사용하고 config의 train/validation 경로를 함께 바꾼다. smoke test 출력과 본 학습 출력은 다른 `output_dir`로 둔다.

이 명령은 이 PC의 smoke test용이다. RTX 4090에서는 `--config configs/qwen3_vl_4b_lora.json`을 쓴다.

### GPU smoke test 기록 (8GB QLoRA)

- 조건: 4-bit, `max_pixels` 200704, 임시 폴더 출력, 최장 샘플 16건, `--max-steps 1`.
- 결과: 코드는 끝까지 돌았고 adapter 47MB도 저장됐지만 **`lora_B` 144개가 전부 0(max 0.0)으로 가중치가 갱신되지 않았다. 학습 미검증이다.**
- 기록값: `train_loss` 0.6374, `eval_loss` 0.7855, 125.5초/step, 전체 245초, 최대 할당 12,268 MiB. 메모리 수치는 유효하다. **loss 값은 학습 증거가 아니다.** 1 step뿐이고 샘플은 test split을 포함한 임시 구성이다. 결과 adapter는 폐기 대상이다.
- **교훈: 완주와 저장만으로 학습 성공을 판단하지 않는다. 가중치 변화까지 확인한다.**
- 메모리 측정(샘플 1건): 양자화 준비 뒤 float32가 된 파라미터를 bf16으로 유지하면 최대 약 2.5GB 줄었다. 1,602토큰은 9,062에서 6,497 MiB로, 4,530토큰은 OOM에서 12,156 MiB로 바뀌었다. 그래서 `train.py`가 prepare 후 float32 파라미터를 bf16으로 되돌린다.
- 첫 시도는 backward에서 OOM(2.19GiB 할당 실패)이었다. bf16 수정 뒤의 실행은 backward를 지나 끝까지 돌았지만, 가중치가 갱신되지 않았으므로 실제 학습의 메모리 안정성 증거로 보기에는 부족하다.

### 과적합 테스트 기록 (8GB QLoRA)

- 조건: 학습 샘플 4건, lr 1e-3, 60 step, 임시 폴더 출력.
- 결과: loss 0.696→0.031, `lora_B` 144개 전부 0이 아님(최대 0.0256), 체크포인트와 학습 중 평가 동작, 균형 샘플링 오류 없음, 최대 할당 5,874 MiB.
- **첫 가중치 갱신 증거이며 일반화 성능 증거가 아니다.** validation loss 상승은 과적합 시험에서 예상된 현상이다.

### RTX 4090 이전 시 주의

- 이 PC에서 증명할 수 없는 항목: BF16 설정의 메모리와 속도, 고해상도 동작, 최장 샘플 backward. 4090에서 preflight와 1 step을 먼저 실행해 확인한다.
- prepare 결과에 절대경로가 있으므로 4090에서 `prepare`를 다시 실행한다.
- 가상환경은 복사하지 말고 새로 설치한다. `requirements-training.lock.txt`는 아직 없다. 실행 스크립트는 `scripts/train_rtx4090.sh`이며 실행 권한은 커밋 때 부여해야 한다.
- 고해상도 토큰 측정은 끝났지만(최대 5,316, 상한 8192) BF16 경로는 코드 추적만 했고 실행하지 않았다. 4090 preflight와 1 step에서 확인한다.

`--preflight-only`는 전체 샘플 토큰화, 길이, prefix 일치 및 LoRA 대상 구성을 확인한다. **GPU forward/backward 성공을 의미하지 않는다.**

## 알려진 잔여 작업과 위험

- **수식·표는 사용자 목표에 포함되지만 현재 데이터로 학습할 수 없다.** 파일럿에서는 제외하고 최종 학습 전에 데이터 추가와 스키마 확장을 진행한다(사용자 결정). CASIA, FlowLearn 합성본 내부 라벨, DIDI 라벨 내용은 미확인이다.
- AI2D-RST, DIDI, FlowLearn, CASIA 변환기 미구현. DIDI는 검증된 정답 대응이 없고, FlowLearn 과학 도표는 완전한 그래프 정답이 없으며, CASIA는 비밀번호가 없다.
- **토큰 상한:** 실측 최대 4,530토큰에 맞춰 4,608로 올렸다. 상한을 올리면 메모리 부담도 커진다. 새 소스를 추가하면 다시 측정한다.
- **1 step의 가중치 미갱신:** 초기 1 step 실행은 `lora_B` 144개가 전부 0이었다. 유력 원인은 warmup 때문에 첫 step 학습률이 0이었던 것이다. warmup 계산을 내림으로 바꾼 뒤 과적합 테스트에서 가중치가 갱신되어 이 가설이 뒷받침되지만, 1 step 재실행 자체는 최장 샘플 OOM으로 확인하지 못했다. 학습 후 `lora_B`가 전부 0이면 오류로 중단하는 검사를 `train.py`에 추가했다.
- **8GB 메모리 초과:** 1 step 실행의 최대 할당이 12,268 MiB로 8GB를 넘었고 할당 실패 경고 후 진행됐다. 초과분은 Windows가 시스템 메모리로 넘기는 것으로 추정하며 확인하지 않았다. **이 PC의 속도를 4090 예상치로 쓰지 않는다.**
- 양자화 제외 패턴은 `visual`이 아니라 `model.visual`이어야 한다. transformers 5.22는 이름 앞/끝 일치로 비교하며, `train.py` 안전 검사가 이 오류를 잡았다. 여유 VRAM 약 6.8GB는 이 PC의 다른 프로세스 사용량에 따라 변한다.
- A안의 약점: 글자-도형 연결을 위치로 계산하므로 도형 밖 글자와 화살표 라벨에서 틀리기 쉽다(예상이며 미측정). A안과 OCR 입력 방식의 정확도 비교는 하지 않았다.
- `stage_sources.py`는 코드 문구만 수정했다. 기존 staging 산출물과 `docs/RADU_DATA_ACQUISITION.json`은 재생성하지 않아 A안 이전 표현이 남아 있다.
- DIDI/FlowLearn 규모가 크면 작은 손그림 데이터가 묻힐 수 있어 `balance_sources`를 켰다. 큰 소스의 양 제한(cap)은 구현되어 있지 않다.
- 중복 검사는 이미지 파일 bytes SHA256 기준이다. 같은 이미지 재인코딩·크롭·버전 변경은 그룹 메타데이터와 추가 이미지 유사도 검사로 관리해야 한다.
- 학습 run은 model commit·환경·입력 manifest 해시를 기록한다.
- RTX 4060 8GB 프로필은 낮은 해상도·4,608 token의 smoke test 전용 설정이다. 작은 화살표 주변 정보 손실은 이 설정으로 판단하지 말고 최종 설정에서 검증한다.
- 토큰 상한 초과는 실패시키며 이미지 토큰이나 JSON 정답을 자르지 않는다. 필요하면 해상도/크롭/예산을 조정하고 좌표 변환을 기록한다.
- 구조 평가는 `src/training/evaluate.py`를 쓴다. 기존 `src/evaluation/evaluate.py`는 이 스키마용이 아니다. 실제 모델 예측에 대한 평가 결과는 아직 없다.
- 다운로드·파싱/학습 성공, 과제별 정확도, 웹 편집기 연동 성공은 각각 별도 검증 사항이다.

## 독립 리뷰 기록

감시 에이전트가 프롬프트 literal brace `.format` 오류와 화살표–관계 대응 필드 누락을 지적했고 수정 확인했다. 이후 집중 리뷰에서 추가 차단급 문제는 발견하지 못했다고 보고했다. 실데이터 적재 중 추가로 발견한 source partition별 COCO image ID 충돌은 namespace로 수정했고, 다운로드/안전 압축 해제/LFS materialization/FlowLearn split 중복 보고 검사와 함께 총 24개 검사 통과를 부모 에이전트가 보고했다.

확인한 보호 장치는 미주석 과제 제외, complete coverage 요구, 그룹/이미지 해시 분할 검사, 프롬프트/패딩 loss 제외, EOS 학습 유지, truncation 금지, 언어 attention q/k/v/o만 LoRA, 예상 trainable parameter 검사, QLoRA kbit 준비다. 실제 processor·GPU forward/backward 검증을 대체하는 리뷰는 아니다.

이번 세션 감시 에이전트 1차 검토에서 나온 결함과 수정:

- `warmup_ratio` 오류: `warmup_steps`로 변경 (`training_schedule`).
- eval/save 주기: 100에서 25로 변경. 도달할 수 없는 설정은 거부한다.
- 원본 ID를 그대로 학습하던 문제: 이미지별 읽기 순서 `n1`/`a1`/`r1`로 변경, 원본은 `source_ids`에 보존.
- FC_A family 1과 6이 같은 그래프인 문제: 별칭 처리.
- FC_A `writer8_1` 유사 중복: 제외.
- OCR 잔여 문구 정리.

이 수정은 테스트 통과까지 확인했으며 GPU 학습 검증을 대체하지 않는다.

감시 에이전트 2차 검토에서 나온 수정:

- `use_cache`를 text config까지 끔.
- 읽기 순서 동률 처리에 x2, y2 추가, 데이터 재생성.
- 학습 step 계산을 ceil로 변경(96).
- 학습 후 최종 평가 추가.
- `__init__` 문구 수정.

감시 결과: 그래프 844건이 원본과 일치. BF16 경로는 코드 추적상 문제 없음(실행은 안 함).

## 다음 AI가 시작할 때

1. 이 파일, `docs/RADU_DATA_ACQUISITION.json`, `configs/radu_datasets.json`을 읽고 실제 `data/raw/radu/` 파일·최종 `verified_assets.jsonl`·인벤토리를 확인한다.
2. FlowLearn raw는 검증 완료다. 중단된 캐시 Git/LFS checkout을 근거로 raw가 불완전하다고 판단하지 않는다. 추가 다운로드 재개가 필요하면 실행 중 프로세스부터 확인하고 같은 출력 파일에 중복 프로세스를 띄우지 않는다.
3. 원본 다운로드 완료, 압축 해제 완료, 정답 변환 완료, SFT 준비 완료를 별도로 집계한다. OCR은 학습 입력이 아니다(A안).
4. 부모 에이전트의 최신 메시지와 실제 파일 상태가 이 문서보다 최신이면 근거를 확인해 이 문서를 갱신한다.
5. 학습 전에 테스트와 실제 데이터 시각 검수를 수행한다. GPU smoke test는 완주와 저장이 아니라 **가중치 변화까지 확인**한 뒤에만 통과로 보고한다.
