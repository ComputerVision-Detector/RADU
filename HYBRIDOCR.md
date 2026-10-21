# HybridOCR 벡터 검수 작업실

RADU 전체를 복사한 `/home/user/Desktop/RADU/HybridOCR` 안에서 개발합니다.

```bash
cd /home/user/Desktop/RADU/HybridOCR
.venv/bin/python -m uvicorn src.hybrid.app:app --host 127.0.0.1 --port 8010
```

http://127.0.0.1:8010 에서 이미지를 올리거나 HF 데이터셋 표본을 불러옵니다. 분석이 끝나면 **벡터 시각 검수 작업실**이 열립니다. 이전 결과는 첫 화면의 **벡터 검수 작업실 열기**로 다시 엽니다.

## 시각 검수와 PDF

- **텍스트**: 원본 위의 영역을 선택합니다. 더블클릭하면 해당 위치에서 글자를 수정할 수 있습니다. `Ctrl+Enter` 또는 영역 바깥 클릭으로 적용하고 `Escape`로 취소합니다. 오른쪽 속성 패널에서도 내용을 수정합니다.
- **이동·크기**: 요소를 드래그하거나 선택 테두리 핸들을 움직입니다. Shift/Control 클릭으로 여러 요소를 고르고 함께 움직일 수 있습니다.
- **표**: 상단 **표 추가**에서 행·열을 정하고, 캔버스에서 표 영역을 드래그합니다. 셀마다 사각형과 텍스트를 가지므로 개별 선택·수정·크기 변경이 가능합니다. 여러 셀은 다중 선택과 편집 메뉴의 그룹 기능으로 함께 이동할 수 있습니다.
- **도형·화살표**: 도구로 추가합니다. `V`로 선택 도구로 돌아온 뒤 화살표의 시작점·끝점 핸들을 움직이거나 속성에서 방향을 반전합니다. 연결 도형과 라벨도 편집할 수 있습니다.
- **비교**: 보기 버튼으로 원본, 오버레이, 벡터 복원, 나란히 비교를 전환합니다. 원본만 보는 상태는 편집하지 않습니다. 확대·축소, 화면 맞춤, Space 드래그로 이동합니다.
- **저장**: 편집은 자동 저장되며, **검수 완료 · 저장**은 사용자가 확인한 상태로 저장합니다. 이후 다시 수정하면 확인 상태가 해제됩니다. 여러 창의 저장 충돌은 revision으로 검출합니다.
- **PDF 다운로드**: 현재 편집한 텍스트·표 셀·도형·화살표의 위치·크기·스타일을 벡터 PDF로 출력합니다. 원본 이미지와 선택 테두리는 출력하지 않습니다. 원본 페이지의 좌표와 비율을 보존합니다.

사용자가 JSON을 직접 편집하는 화면은 제공하지 않습니다. 내부 저장·전송에만 JSON을 사용합니다. 기존 RADU 편집기와 공용 SVG 상호작용·출력 코드를 재사용합니다.

## 분석과 좌표

1. 이미지 EXIF 방향을 보정하고 RGB PNG로 저장합니다. 자동 원근 변환은 적용하지 않습니다.
2. PaddleOCR-VL 1.6 spotting으로 텍스트와 원본 좌표를 추출합니다. 각 줄에 고정 id를 붙입니다.
3. 구조 검출기가 도형·화살표의 좌표 후보를 생성합니다. OpenCV 후보에는 오검출이 포함될 수 있습니다. 검수 화면에서 삭제·수정하거나 표 도구로 누락을 보완합니다.
4. Paddle 프로세스 종료 후 Qwen2.5-VL-7B에 **동일 원본 이미지와 OCR·구조 후보 JSON**을 전달합니다. Qwen은 OCR id별 교정문과 불확실 항목을 반환합니다. 임의 좌표 생성을 요청하지 않습니다.
5. OCR 좌표에 Qwen 교정문을 연결하여 SVG 요소로 구성합니다. 이전 Markdown 결과는 기존 OCR 위치를 보존하고, 유사도가 높은 문장만 보수적으로 대응합니다. 이전 작업을 열었다고 새 모델 추론을 수행하지는 않습니다.

새 작업은 `data/hybrid/<작업 ID>/`에 `source.png`, `ocr.json`, `review.json`, `review.raw.txt`, `vector.json`, `state.json`, `stage.txt`, `run.log`를 저장합니다. 원본 OCR·모델 출력은 사용자 벡터 편집으로 덮어쓰지 않습니다. 예전 `final.json`은 호환용으로 보존합니다.

HF 이미지 전용 저장소는 파일을 seed로 선택합니다. 표준 `data/<split>-*.parquet` 저장소는 shard·행을 seed로 선택하고 revision·파일·행 번호를 기록합니다. 전체 데이터셋에 대한 균등 추출을 보장하지 않습니다. 그 외 구조화된 데이터셋에는 기존 `load_random_records`와 `extract_image`를 재사용합니다.

## 실행 환경

새 환경은 Python 3.10 이상에서 `python -m pip install -r requirements-hybrid.txt`로 설치하고, CUDA에 맞는 `paddlepaddle-gpu`를 별도 설치합니다. 복사한 가상환경은 기존 RADU와 같이 `/home/user/Desktop/HybridOCR/.venv`의 패키지를 `.pth`로 참조합니다. 독립 재배포 환경에서는 의존성을 다시 설치해야 합니다. 복사된 CLI shebang 대신 `.venv/bin/python -m ...`으로 실행하세요.

로컬 가중치는 `models/vision_language/PaddleOCR-VL-1.6`, `models/vision_language/Qwen2.5-VL-7B-Instruct`에 있습니다. PDF 출력은 CairoSVG·시스템 Cairo·한글 폰트를 사용합니다.

환경 변수:

- `HYBRIDOCR_QWEN_MODEL`: Qwen 로컬 체크포인트 또는 HF ID
- `HYBRIDOCR_MAX_TOKENS`: Qwen 생성 상한, 기본 6144. 잘린 응답은 성공 처리하지 않습니다.
- `HYBRIDOCR_MAX_PIXELS`: Qwen 이미지 픽셀 상한, 기본 1605632
- `HYBRIDOCR_JOB_TIMEOUT`: 모델 전체 제한, 기본 1200초
- `HYBRIDOCR_REVIEW_TIMEOUT`: Qwen 제한, 기본 900초

**서버 worker는 1개로 실행하세요.** 서버 내 GPU 작업은 순차 처리하며, 실행 도중 서버 재시작 시 작업을 다시 요청해야 합니다. 현재는 한 이미지·한 페이지 검수입니다.

## 검증과 한계

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tests/browser/hybrid_vector_review.py
```

두 번째 명령은 실행 중인 로컬 서버, Playwright, 시스템 Chrome과 검증용 HF 표본을 사용합니다. 실제 SVG 더블클릭, 드래그·크기 변경, 표 셀 수정, 화살표 끝점 변경, 저장·다시 열기, PDF 다운로드를 검사합니다.

인식 결과는 검수 후보입니다. 한국어 손글씨·수식·복잡한 표·화살표 연결의 정확성은 보장하지 않습니다. 병합 셀 자동 복원과 원본 픽셀 수준의 레이아웃 재현은 별도 개선이 필요합니다. 현재 표 셀은 개별 사각형이므로 셀 크기를 바꾸어도 인접 셀의 경계는 자동으로 함께 움직이지 않습니다. Qwen이 불확실 항목을 반환하지 않아도 사용자의 확인이 필요합니다.
