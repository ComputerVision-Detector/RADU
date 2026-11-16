# OCR 인식 모델만 추가 튜닝한 결과 — 2026-11-12

**결론: 실제 추가 학습을 완료했지만 채택 가능한 CER 개선은 확인하지 못했다. 최초 VL 언어 LoRA를 유지한다.** 검출기·crop·전처리·읽기 순서·GT·운영 선택은 변경하지 않았다. 실패 후보도 checkpoint·optimizer·예측·로그와 함께 별도 경로에 보존했다.

## 범위와 고정 조건

이번 사용자 지시로 새 optimizer step을 수행했다. 직전까지의 ‘새 학습 없음’ 감사 범위와 별개인 신규 모델 튜닝 실험이다. 로컬 RTX4090을 사용했고 유료 자원 추가·모델 다운로드·배포·원격 push·라벨 수정은 없다. 적용할 AGENTS.md는 저장소와 상위 검색에서 발견되지 않았다. 기존 Git 수정/미추적 산출물은 보존했다.

부모는 `data/experiments/ocr_v2_vl_lora/train/best-adapter`이고 `adapter_model.safetensors` SHA256은 `ab1bfcfba5c0711ebfc82e83432eca13b47fbb48ca0800d9e7b4d590e15421e5`로 유지됐다. 디렉터리 해시가 아니다. 같은 공식 base·processor/tokenizer·이미지112896~200704pixels·OCR:·BF16/SDPA·left-padding·greedy/beams1·max_new_tokens128을 사용했다.

학습 데이터는 이전 라운드에 이미 존재하던 clean train2,341crop/108페이지다. 기존 데이터·전사·보류·분할을 바꾸지 않았다. 별도 인간 검수를 수행하거나 정답을 OCR로 채우지 않았다. validation244crop/14페이지/5,613문자도 이전 범위를 고정했다. train/validation 페이지 겹침0을 확인했다. 이것은 반복 사용된 연구용 validation이고 독립 필자 일반화 벤치마크가 아니다.

추론은 이미 확정한 singleton-v1(batch1, 입력마다seed42)으로 통일했다. 부모 기준선도 이번에 새로 실행했다. 과거 batch4의12.7917%/12.9877%를 before로 섞지 않았다. 검출·crop·전처리 코드를 수정하지 않고 인식 모델의 동일 입력 출력을 비교한 것이다.

## 실행 전 고정한 두 후보

두 후보 모두 같은 원래 부모에서 독립적으로 출발했다. 첫 후보 결과로 다음 후보의 설정을 바꾸거나 첫 후보 checkpoint에서 이어 학습하지 않았다.

- **A: 영문 토큰 가중 CE.** 정답 completion token을 tokenizer로 디코딩했을 때 ASCII 영문자가 있으면 가중치1.5, 그 외1.0이다. 가중합을 전체 가중치로 나눈 CE를 사용했다. 출력 철자/대소문자를 교정하거나 바꾸는 후처리는 없다.
- **B: 같은 가중 CE + 부모 분포 보존.** 동일 train 입력·동일 정답 completion 위치에서 고정 부모의 soft distribution과 student 사이 KL(parent||student)을0.1배 더했다. 온도1, teacher는 eval/no_grad이며 optimizer에 포함하지 않았다. 부모 출력을 새로운 hard GT로 만들지는 않았다.

각2epoch·586step, batch2/누적4, AdamW lr1e-5/weight_decay0.01, warmup10step 뒤 cosine으로최초LR의10%까지, clip1, seed42다. 기존18개 언어 attention layer q/k/v/o LoRA의1,253,376파라미터만 학습했다. 기본 모델·시각 encoder·projector는 동결했다. 이미지 증강·새 crop·해상도/검출 threshold 탐색은 없다.

validation 선택 규칙은 plan.json에 학습 전에 기록했다. 부모를 포함하고 CER이 엄격히 낮으며, 공백 제외 CER 및 영문 포함 영역 CER이 나빠지지 않고, 한글 전용 영역 CER이 부모보다0.5%p 이상 나빠지지 않는 후보 중 최저CER를 고른다. 동률/미통과는 부모 유지다. test는 선택에 사용하지 않는다.

## 실제 결과

모든 행은 같은 validation244개·14페이지·정답5,613문자·batch1이다. 공백 제외 분모는4,592문자다.

| 모델 | micro CER | 공백 제외 CER | 완전 일치율 | S / D / I | 총오류 |
|---|---:|---:|---:|---|---:|
| **기존 부모** | **13.1659%** | **13.4582%** |21.7213%|516/107/116|**739**|
| A epoch1 |13.3975%|13.7413%|22.1311%|528/103/121|752|
| A epoch2 |13.6291%|14.0897%|22.5410%|531/101/133|765|
| B epoch1 |13.3262%|13.7413%|22.1311%|527/96/125|748|
| B epoch2 |13.3262%|13.7195%|22.5410%|520/95/133|748|

A의 최선은 부모보다13오류, B의 최선은9오류 더 많았다. 완전 일치는 최대53/244→55/244로 늘었지만 CER 기준을 바꾸어 성공으로 집계하지 않았다. 삭제 감소와 삽입/치환 증가가 상쇄되어 전체 오류가 늘었다.

| 영역 구분 | 부모 | A epoch1 | A epoch2 | B epoch1 | B epoch2 |
|---|---:|---:|---:|---:|---:|
| 영문 포함(혼합 포함)157영역/3,879문자 CER |12.7868%|12.7868%|13.1219%|12.7610%|12.8126%|
| 한글 전용82영역/1,695문자 CER |14.1003%|14.8673%|14.8673%|14.6903%|14.5723%|

이 분류는 GT의 문자 존재 여부에 따른 휴리스틱이며, 영문 포함 CER은 영문 문자만의 오류율이 아니라 해당 영역 전체 전사의 CER이다. B epoch1에서 영문 포함 영역은1오류 줄었지만 한글 오류가10개 늘었다. 한글·영문 혼용에서 균일한 이점을 확인하지 못했다.

영역별로 A1은23개 개선/34악화/187오류수 동일, A2는24/36/184, B1은24/33/187, B2는27/31/186이었다. 후보마다69~73개 문자열이 달라졌다. 동일 오류 수가 동일 출력이라는 뜻은 아니다. 원문·출력·오류 변화는 `summary/candidate-error-cases.json`에 있다.

페이지 bootstrap2,000회/seed42의 CER 개선량95% 구간은 A1 −0.7135~+0.1403%p, A2 −1.0941~−0.0165%p, B1 −0.7218~+0.2407%p, B2 −0.7111~+0.2692%p다. 반복 사용 validation에서 본 사후 기술 통계이며 독립 확증·필자 일반화 구간이 아니다. 최종 부모와 부모의 차이[0,0]도 같은 출력 비교의 결과일 뿐 새 방법의 효과가 아니다.

## 검증과 실행량

- 새 모델 튜닝7개+기존68개, **75개 관련 테스트 통과**,1.352초. teacher detach/형상, 정답 completion의 causal shift와 prompt/padding mask, 가중 CE 산식, 선택 guard, 빈 출력 분모, 입력/행 hash drift 및 덮어쓰기 거절을 확인했다. 전체 프로젝트 discovery는 실행하지 않았다.
- GPU smoke는 train8개/1step에서 실제LoRA최대변화0.00000100024, 유한loss/gradient, 동결 불변, fresh base 재로딩 출력 일치를 확인했다. smoke adapter는 본 학습 부모나 선택 후보로 쓰지 않았다.
- A/B 본 학습은 각586step, 합계 **1,172 optimizer step**이다. smoke 포함1,173step이며 혼동하지 않는다. student 학습 입력은2,341×2×2=9,364crop, smoke8개 별도다. B의 고정 teacher는4,682개 train 입력을 추가 forward했다.
- 생성 추론은 부모244+반복4, 후보validation244×4, 두 후보 fresh reload4×2, smokevalidation4+reload4로 **1,240crop**이다. 추론 실패0건이다.
- A 가중치최대변화0.00133397, B0.00132042였다. 두 후보 모두 비학습 **905,601,648파라미터**의 byte digest가 학습 전후 동일했다. 기존 부모 파일도 해시 불변이다.
- A 최대GPU할당약3.12GB, B약4.96GB. 환경RTX4090/Python3.11.15/PyTorch2.14.1+cu126/Transformers5.22.1/PEFT0.21.2다. 학습·epoch별validation 시간 합은 A223.341초, B292.130초로 총515.471초다. 최초 모델 로딩·추가 hash/smoke 등은 이 합에 포함하지 않아 전체 작업 시간으로 표현하지 않는다.
- 부모 첫4입력 반복, 각 후보 마지막epoch를 새 base+adapter로 재로딩한4입력 모두 일치했다. 모델/processor/config/crop/라벨/기존CV평가코드·훈련행/평가행의 hash를 단계 전후 확인했다. 후보checkpoint는 학습 종료 후 별도trained-candidate-lock.json으로 고정했다.
- 부모 및4개epoch 후보의 CER/문자 분모는 평가 helper와 별개의 NFC/공백/동적계획법 구현으로 재계산해 모두 일치했다.

## 최종 선택과 해석

선택은 **parent 유지**다. 새 후보가 validation 기준을 통과하지 못해 test235를 새로 실행하지 않았다. test에서 우연히 좋아 보이는 모델로 재선택하지 않았다. 기존 부모의 역사적 test235 CER15.2238%를 이번 batch1 조건의 새 측정처럼 인용하지 않는다.

이 결과는 ‘동일 기존 데이터에 대한 영문 토큰 가중 및 약한 부모분포보존 추가학습’ 두 방법의 비교다. 모든 OCR 튜닝이 불가능하거나 부모가 최적이라고 증명한 것은 아니다. 모델은 이미 같은 데이터 계열을 학습했고, 전사 잡음·crop 경계·작고 반복된 validation이라는 제약이 있다. 영문 가중이 한글 퇴행의 유일 원인이라고 단정하지도 않는다.

요청대로 OCR 모델만 튜닝했고 다른 파이프라인 부분은 그대로다. 운영 모델 교체는 하지 않는다. 새 후보는 오류 분석/재현용으로 보존한다. 추가 실험을 잘 된 결과가 나올 때까지 무한히 이어가거나 성능 기준을 사후 바꾸지 않았다.

## 산출물과 재현

새 루트: `data/experiments/ocr_model_only_20261112/`.

- `plan.json`, `EXPERIMENT_CONTRACT.md`, `input-lock.json`: 사전계획·원본/환경/설정/코드/입력 고정.
- `parent-validation/`, `parent-repeat/`, `parent.json`: 실제 새 batch1 기준선.
- `latin_weighted/`, `latin_weighted_parent_KL/`: 각2epoch 실제 adapter·optimizer·예측·가중치/동결/재로딩 검증.
- `trained-candidate-lock.json`, `selection.json`, `selected-verification.json`, `selected-model.json`: 모델 파일과 선택의 연결. 부모 유지이며 기존앱의 선택 파일은 변경하지 않았다.
- `summary/validation.csv`, `results.json`, `candidate-error-cases.json`, `independent-validation-CER.json`: 실제 정량·언어 구성별 결과와 재계산.
- `tests.log`, `baseline.log`, `smoke.log`, `train-latin-weighted.log`, `train-parent-KL.log`: 실행 증거.
- `final-not-run.json`: test를 실행하지 않은 이유.
- `executed-code/`, `completion-manifest.json`, `preservation-check.json`: 실행 소스·최종파일 hash·기존산출물 불변 확인.

명령 전체는 run의 README.md에 있다. 핵심 실행 코드는 `src/training/ocr_model_only.py`, 검증/요약은 `src/evaluation/ocr_model_only_verify.py`, `ocr_model_only_report.py`다. 기존root와 다른 새 경로에서만 재실행한다. 학습을 또 수행하는 명령이며 단순 요약 재현과 구분한다.

```bash
.cache/venvs/ocr-vl-lora/bin/python -m src.training.ocr_model_only --root /new/run --stage prepare
.cache/venvs/ocr-vl-lora/bin/python -m src.training.ocr_model_only --root /new/run --stage baseline
.cache/venvs/ocr-vl-lora/bin/python -m src.training.ocr_model_only --root /new/run --stage smoke --arm latin_weighted_parent_KL
.cache/venvs/ocr-vl-lora/bin/python -m src.training.ocr_model_only --root /new/run --stage train --arm latin_weighted
.cache/venvs/ocr-vl-lora/bin/python -m src.training.ocr_model_only --root /new/run --stage train --arm latin_weighted_parent_KL
.cache/venvs/ocr-vl-lora/bin/python -m src.evaluation.ocr_model_only_verify --root /new/run --stage freeze
.cache/venvs/ocr-vl-lora/bin/python -m src.training.ocr_model_only --root /new/run --stage select
.cache/venvs/ocr-vl-lora/bin/python -m src.evaluation.ocr_model_only_verify --root /new/run --stage check
.cache/venvs/ocr-vl-lora/bin/python -m src.training.ocr_model_only --root /new/run --stage final
.cache/venvs/ocr-vl-lora/bin/python -m src.evaluation.ocr_model_only_report --root /new/run
```

목적함수/직렬화 구현 근거: [PyTorch KL divergence](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.kl_div.html), [PEFT checkpoint 형식](https://huggingface.co/docs/peft/developer_guides/checkpoint). 외부 문서가 RADU에서의 개선을 보장한다는 뜻은 아니며 채택 판단은 위 실제 결과만 사용했다.
