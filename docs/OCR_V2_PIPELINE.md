# V2 OCR 데이터 파이프라인

PaddleOCR-VL 1.6의 자연어 OCR과 Qwen3-VL의 도형·화살표·수식·표 경로는 유지한다. 이 도구는 `src/training/schema.py`의 구조 SFT 계약을 변경하지 않는다. 본 학습, 모델 다운로드, 배포, GitHub 라벨 쓰기와 push는 실행하지 않는다.

## 실행 환경과 재현

저장소 루트에서 실행한다. 기존 `.venv`에 필요한 패키지가 있어 이번 실행에서는 의존성을 설치하지 않았다. 새 환경은 `python3 -m venv .venv-ocr-data` 후 `.venv-ocr-data/bin/python -m pip install -r requirements-ocr-data.txt`로 만든다. 원본 PNG 해시는 encoder에도 의존하므로 `Pillow==12.3.0`을 유지한다. 추론은 기존 Paddle 환경에서만 별도로 실행한다. 데이터·평가는 GPU/Paddle을 import하지 않는다. 로컬 디렉터리의 대용량 원본·crop은 Git에서 무시되지만 이 작업의 실제 산출물로 디스크에 남는다. 요약 증거는 `docs/ocr_v2/`에 보존한다.

```bash
# 출력 디렉터리는 새 경로여야 한다. 기존 스냅샷/라벨/산출물을 덮어쓰지 않는다.
.venv/bin/python -m src.ocr_data.snapshot --output data/processed/ocr_v2/snapshot-50aca98
.venv/bin/python -m src.ocr_data.restore --snapshot data/processed/ocr_v2/snapshot-50aca98 --output data/processed/ocr_v2/originals --cache-only
.venv/bin/python -m src.ocr_data.build --snapshot data/processed/ocr_v2/snapshot-50aca98 --originals data/processed/ocr_v2/originals --output data/processed/ocr_v2/pilot-v2 --seed 42
.venv/bin/python -m src.ocr_data.validate --dataset data/processed/ocr_v2/pilot-v2
.venv/bin/python -m src.ocr_data.export_recognition --dataset data/processed/ocr_v2/pilot-v2 --config models/recognition/PP-OCRv6_medium_rec/inference.yml --max-length 256 --output data/processed/ocr_v2/ppocrv6-lists-v2
.venv/bin/python -m src.ocr_data.export_recognition --model paddleocrvl16 --dataset data/processed/ocr_v2/pilot-v2 --tokenizer models/vision_language/PaddleOCR-VL-1.6/tokenizer.json --max-length 256 --output data/processed/ocr_v2/paddleocrvl16-lists-v2-final
.venv/bin/python -m unittest tests.test_ocr_v2 -v
```

`--cache-only`를 빼면 **데이터 원본만** 다운로드할 수 있다. 원본 122장은 기존 `annotation_images.restore`의 parquet revision `7f97f382d9d82aa6b21e9b617b46218f1ec4b0aa`를 사용하고, 추가 이미지는 카탈로그에 고정된 loose-file revision을 사용한다. 현재 카탈로그/V2 revision은 `93f3a5c0cf545268eb258119fcb92a0ca410b69f`이다. 실제 각 페이지 revision은 `originals/restore_receipts.json`에 남긴다. 재현 시 legacy manifest는 snapshot manifest의 `code_commit`에 있는 `data/benchmark/qwen_ablation_pilot/additional_images.manifest.json`을 함께 사용한다. 최신 snapshot을 원하면 `--commit`을 생략하고 새 출력 경로를 사용한다. 예전 라벨을 고정하려면 `--commit 50aca983da9885d9e7a965c1ef4b05055d47b252`를 지정한다. 원격 최신 SHA 확인은 여전히 실행한다.

## 데이터 계약과 보류

원본 `live-v2.json`의 `items[id].bbox_annotation.boxes`를 읽는다. 앱의 `samples` 통합 export를 이 형식으로 오인하면 중단한다. `text`만 정답이며 `prediction`은 변환 레코드에 복사하지 않는다. bbox는 원본 픽셀 xyxy이고, fractional 좌표가 들어오면 crop 경계만 floor/ceil하여 별도 `crop_bbox`에 기록한다. WebP는 crop 입력으로 사용하지 않는다. 원본 해시/크기, V2의 source fingerprint, assets fingerprint가 다르면 정상 데이터에서 보류한다. 원본 실패는 `integrity_failures.json`에 기록하고 CLI exit 2로 끝낸다.

검수 미완료·불확실·빈 전사·좌표 오류·중복 ID/순서·거의 동일한 중복 영역·미해결 메모는 보류한다. 비수식 여러 줄 영역도 줄 crop으로 확정하지 않고 보류한다. 일부 영역에 문제가 있으면 기본 파이프라인은 페이지 전체를 보류해 부분 GT가 전체 페이지 GT처럼 쓰이지 않도록 한다. 이유와 원래 문자열은 `pages.jsonl`, `review_queue.jsonl`에 보존한다. 잘못된 no_text인 hf-0104와 병합 메모가 남은 hf-0011은 고정 보류 사유가 있다. 대표 오버레이에서 제목·포스트잇 손글씨 누락을 확인한 hf-0000도 pilot-v2부터 전체 보류한다. 이 사유는 metadata 양식을 채워도 자동 해제되지 않는다. 수동 검수 후 새 정책 버전/새 산출물을 만든다.

유형 `text/formula/symbol/unclassified`는 모두 `type_suggestion`이며 `type_confirmed=null`이다. LaTeX 흔적 탐지 규칙은 `\\` 다음 영문 명령, escaped 괄호 또는 `$` 존재이다. 넓은 후보 탐지라 일반 달러 기호나 jQuery의 `$()` 코드도 포함될 수 있다. 사전 집계 418과 같은 규칙이라고 가정하지 않으며 이번 규칙의 결과는 434이다. 기호 전용은 한글 음절/자모·영문·숫자가 없는 비어 있지 않은 전사이다. 여러 분류는 중첩된다. 분수·행렬은 bbox 내부 개행까지 원문 그대로 보존하고 자동으로 줄 분할하지 않는다. 긴 문자열 기준은 256 codepoint 초과이며 삭제/잘림은 없다. crop 전체 메타데이터와 별도로 일반 텍스트 **후보** JSONL을 생성한다. 유형은 시각 검수 후 확정해야 한다.

주 경로인 PaddleOCR-VL 1.6에는 `--model paddleocrvl16`으로 원본 crop·`OCR:` query·정답·로컬 tokenizer의 label token 수를 연결한 JSONL 학습 과제 목록을 만든다. 이는 향후 학습 adapter용 목록이며 upstream trainer 통합을 구현했다는 뜻은 아니다. 이번에는 3,045개 후보가 label token 한도 256 이내였고 최대는 70이었다. 이미지/프롬프트 토큰은 향후 학습 preflight에서 별도로 예산 검사를 해야 한다.

별도 인식기 비교용 PP-OCRv6 인식 목록은 로컬 `inference.yml`의 실제 문자 사전을 읽어 `crop_absolute_path<TAB>original_text`를 만든다. 탭·개행, 최대 길이 초과, 미지원 문자는 `rejected.jsonl`에 원문/이유와 함께 남기며 변환하지 않는다. 256은 이번 데이터 검사에 선택한 명시적 한도이며 모델 학습 설정의 기본값이라고 주장하지 않는다. 학습 시 사전·공백 처리와 `max_text_length`를 맞춰야 한다. 목록 생성은 학습 실행이 아니다. 후보 목록은 자동 유형 분류 검수 전의 파일럿용이며 확정 학습셋이 아니다.

## 분할과 검수 산출물

crop 전에 페이지와 그룹의 연결 성분을 계산하고 seed 42로 약 80/10/10을 배정한다. 동일 원본 SHA와 디코딩된 RGB 픽셀 해시를 묶는다. dHash 64-bit Hamming distance <=4인 유사 이미지 **후보**도 보수적으로 같은 그룹에 놓고 검수 목록에 남긴다. 이 탐지는 촬영 각도·회전·부분 크롭을 모두 검출한다는 보장이 없다.

`group_metadata.template.json`의 writer_id/note_id/design_id/source_family_id와 evidence를 사람이 채워 `build --metadata FILE --output NEW_DIRECTORY`에 전달한다. annotator는 필자로 쓰지 않는다. 양식의 `type_overrides`와 `review_resolution`은 검수 기록용 예약 필드이며 현재 build에는 반영하지 않는다. 이 버전의 metadata 입력은 그룹 정보에만 적용한다. 현재 그룹 정보가 없으므로 분할 상태는 `provisional_page_group_pilot`, `writer_independence_verified=false`이다. 그룹을 추가하면 분할이 바뀔 수 있으므로 기존 test와 비교한 최종 성능을 주장하지 않는다. 증강은 현재 없으며 이후 train에만 적용한다. 필자 독립 최종 테스트셋은 아직 없다. 파일럿 test도 임계값/모델 선택/스모크에 사용하지 않는다.

`pages.jsonl`은 모든 페이지의 원본 경로·정답 bbox·전사·읽기 순서·제외 사유를 보존한다. `crops.jsonl`은 실제 생성된 각 crop의 원본/파일 해시와 분할을 연결한다. `splits.json`은 seed, 페이지 배정, 그룹 연결 성분과 입력 해시를 담는다. `overlays/`와 `review_images/`는 원본에서 생성한 점검 이미지이다. `manifest.json`은 데이터 및 보고서 해시를 고정한다. `validate`는 원본·crop 해시/크기·정답 원문·분할 누수를 다시 검사한다.

## 저장 예측 평가

페이지 모드 입력 예:

```json
{"mode":"page","pages":[{"id":"hf-0000","regions":[{"bbox":[765,40,1552,178],"text":"모델 출력","order":1}]}]}
```

두 예측 JSON의 최상위에 `dataset_manifest_sha256`을 반드시 넣는다. 값은 사용한 데이터셋의 `manifest.json` 파일 SHA256이다. 누락 또는 다른 fingerprint는 평가를 거부한다. bbox는 원본 좌표이다. `order`를 모든 영역에 명시하거나 목록 자체를 예측 읽기 순서로 제공한다. 전체 페이지 문자열은 필터링 후 모든 예측 영역을 그 순서로 이어 붙인다. 검출 IoU 0.5에서 최대 매칭 수를 얻는 augmenting-path 이분 매칭을 사용한다. 동일 매칭 수 내 총 IoU 최댓값을 보장하지 않으며, 인접 후보는 IoU 내림차순으로 결정한다. `cases`에 누락·오탐·분할/병합 후보 및 읽기 순서 역전을 남긴다. 전체 OCR CER은 매칭 성공 영역만 계산하지 않는다.

인식 전용 입력 예:

```json
{"mode":"recognition_only","crops":[{"page_id":"hf-0000","region_id":"hf-0000-line-0001","text":"인식기 출력"}]}
```

동일 GT bbox crop을 **인식 모델에 직접** 넣은 결과만 이 모드를 사용한다. mode 선언만으로 외부 추론 과정을 자동 입증하지는 못하므로 모델·가중치·crop provenance도 함께 보존해야 한다. 검출을 다시 실행한 crop 출력은 이 형식으로 표시하면 안 된다. `infer_smoke`의 이 모드는 PaddleX `create_model(PP-OCRv6_medium_rec)` 인식기만 호출한다.

```bash
.venv/bin/python -m src.evaluation.ocr_v2 --dataset data/processed/ocr_v2/pilot-v2 --predictions predictions.page.json --split validation --output evaluation.page.json
.venv/bin/python -m src.evaluation.ocr_v2 --dataset data/processed/ocr_v2/pilot-v2 --predictions predictions.recognition.json --split validation --output evaluation.recognition.json
```

예측이 빠진 GT도 분모에 포함한다. 실패는 `error` 필드로 제공하며 빈 출력으로 채점하고 별도 보고한다. 정상 no_text 페이지의 오탐을 센다. 빈 정답의 CER은 분모 0이므로 null이고 삽입 수와 no_text 오탐으로 보고한다. 기존 `ocr_metrics.normalize`의 NFC/공백 병합을 재사용하며 micro CER·공백 제외 CER·완전 일치율·치환/삭제/삽입 수를 계산한다. `text`와 `formula` 등 제안 유형별 인식 지표를 분리한다. 수식 CER은 LaTeX 문자열 비교이며 수학적 동치성 지표가 아니다. 여러 줄 수식의 완전 일치는 region 단위이다.

기본 build는 보류 페이지 전체를 제외한다. 외부에서 부분 GT를 구성하는 경우 보류 bbox 면적 안에 예측 면적의 50% 이상이 있는 예측은 검출과 전체 문자열 모두에서 제외한다. 유효/보류 영역을 동시에 걸치는 예측은 오류로 중단하여 애매한 가림 정책으로 점수를 올리지 않는다. 알 수 없는 페이지/영역·중복 예측 ID·잘못된 예측 좌표도 중단한다. 데이터셋 split 밖의 예측은 섞어 넣지 않는다.

## 개발 추론과 다음 판단

```bash
# train에서 최대 5건만 선택 가능. 테스트셋 선택 옵션 없음.
.venv/bin/python -m src.ocr_data.infer_smoke --dataset data/processed/ocr_v2/pilot-v2 --mode recognition_only --device gpu:0 --limit 2 --output smoke.recognition.json
.venv/bin/python -m src.ocr_data.infer_smoke --dataset data/processed/ocr_v2/pilot-v2 --mode page --device gpu:0 --limit 2 --output smoke.page.json
```

스모크는 실행 확인용이다. 소수 예측 JSON을 전체 split 평가기에 넣으면 나머지는 누락으로 계산되므로 전체 성능으로 해석하면 안 된다. 전체 validation 추론을 별도로 수행한 뒤 고정 IoU/정규화로 비교한다. GT crop 인식 CER이 낮은데 페이지 누락/분할/순서 오류가 크면 검출·읽기 순서 경로를, 검출 recall이 충분한데 GT crop CER이 크면 인식기를 우선 개선한다. 둘 다 나쁘면 각각 분리해 개선한다. 지금은 기준 성능을 실제 측정하지 않았으므로 어느 모델을 먼저 학습할지 결론 내리지 않는다.
