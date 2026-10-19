# RADU

손그림 순서도 이미지에서 텍스트·도형·화살표를 구조화하고 관계를 복원한 뒤, 웹에서 검수·편집해 JSON/SVG/PNG/PDF로 내보내는 캡스톤 시제품입니다.

## 구성

```text
data/                         원본·처리 데이터 및 벤치마크 결과 (기본적으로 Git 제외)
models/                       로컬 가중치 (Git 제외)
src/cv/                       전처리, 구성요소 후보 검출, OCR, 관계 분석, 출력
src/evaluation/               Qwen2.5-VL 기준선 및 요소/관계 평가
src/web/backend/              FastAPI 업로드·분석·저장·내보내기 API
src/web/frontend/             SVG 기반 웹 검수·편집기
```

## 빠른 시작

Python 3.10 이상을 권장합니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-web.txt
uvicorn src.web.backend.app:app --reload
```

브라우저에서 `http://127.0.0.1:8000`을 열어 이미지를 올립니다. SVG 순서도는 CairoSVG로 PNG 변환해 래스터 이미지와 같은 분석 파이프라인에 전달합니다. PDF는 직접 입력할 수 없으므로 SVG 또는 래스터 이미지로 내보낸 뒤 업로드하세요. 웹 서버는 Pillow가 디코딩할 수 있는 래스터 이미지 형식을 받으며, AVIF와 HEIF/HEIC 디코더도 설치합니다. GIF·다중 페이지 이미지에서는 첫 프레임/페이지만 분석합니다. 파일 크기·픽셀 수에 애플리케이션 상한은 두지 않지만, 실제 처리는 사용 가능한 메모리·저장 공간·분석 시간에 따릅니다. 웹 전용 의존성에는 CairoSVG가 포함되어 있으며, SVG 변환과 PNG/PDF 출력에는 시스템 Cairo 라이브러리가 필요합니다. PaddleOCR를 사용하려면 `pip install -r requirements.txt`로 전체 OCR 의존성을 추가하세요. 현재 기능의 검증 범위와 남은 항목은 [웹 UX 점검 기록](docs/WEB_UX_AUDIT_2026-10-19.md)에 정리했습니다.

## 모델 및 외부 OCR 설정

- **구성요소 검출:** 학습된 Ultralytics 가중치가 있다면 `pip install ultralytics` 후 `RADU_DETECTOR_WEIGHTS=/path/to/weights.pt`를 지정합니다. 클래스 이름은 `text`, `arrow`, `box` 또는 `process`, `decision`, `terminator`를 포함해야 합니다. 가중치가 없으면 OpenCV로 후보 상자를 생성합니다. 이 후보는 학습·평가된 객체 검출기가 아니므로 결과에 검수 경고가 붙습니다.
- **현재 기본 비교 대상은 두 로컬 모델입니다:** PaddleOCR v6 (PP-OCRv6), PaddleOCR-VL 1.6. CLOVA 어댑터는 코드에 유지하지만 API 비용 때문에 기본 벤치마크와 웹 선택에서 보류합니다. 나중에 비교할 때만 `engines=("clova", "paddleocrv6", "paddleocrvl16")`로 명시해 실행합니다. CLOVA 요청 오류도 샘플 결과에 기록하고 다음 엔진으로 진행합니다.
- **PaddleOCR v6:** `PaddleOCR(ocr_version="PP-OCRv6")` 로컬 OCR 파이프라인입니다. PaddleOCR/PaddlePaddle 설치와 CPU/GPU 호환성을 맞춰야 합니다.
- **PaddleOCR-VL 1.6:** 현재 worker는 `PaddleOCR-VL-1.6-0.9B` 모델에 `Spotting:` 질의를 보내 텍스트와 위치를 추출합니다. 웹에서 이 엔진을 선택하면 결과만 오버레이하고 OpenCV 구성요소 검출 및 RADU 관계 복원은 실행하지 않습니다. CPU/GPU, 모델 저장공간과 백엔드 구성이 v6보다 더 큰 영향을 주므로 벤치마크 자원 조건과 함께 결과를 검토해야 합니다.
- **CLOVA OCR:** CLOVA OCR의 Invoke URL과 Secret을 `CLOVA_OCR_INVOKE_URL`, `CLOVA_OCR_SECRET` 환경 변수로 설정한 뒤 선택합니다. Secret은 코드나 저장소에 기록하지 마세요.
- **Qwen2.5-VL:** `python -m src.evaluation.run_qwen_vlm IMAGE --model MODEL_ID_OR_LOCAL_PATH --output OUTPUT.json`으로 실행합니다. 모델 다운로드·실행에 적절한 GPU 메모리와 Hugging Face 접근이 필요할 수 있습니다.

현재 개발 환경에는 PaddleOCR 3.7.0과 PaddlePaddle GPU 3.2.1이 설치되어 있고, 필요한 가중치만 이 프로젝트의 `models/` 아래로 복사했습니다. 이 작업공간의 `.venv`는 해당 기기에 미리 설치된 `/home/user/Desktop/HybridOCR/.venv` Paddle 패키지를 `.pth` 경로로 참조하며, 모델 실행 시에는 RADU 내부 가중치 경로를 직접 사용합니다. 다른 기기에서는 빠른 시작대로 의존성을 설치하세요. 이 세션에서는 NVIDIA 드라이버가 감지되지 않아 CPU로 실행했고, PP-OCRv6와 PaddleOCR-VL 1.6 모두 프로젝트 안의 로컬 가중치로 추론 경로가 실행되는 것을 확인했습니다. VL의 CPU 초기화는 약 13초, 샘플 추론까지 약 16초가 걸렸으며, 제공된 도식 PNG는 VL이 텍스트를 추출하지 못했습니다. 실제 정확도 비교에는 손그림 순서도 이미지와 전사 정답 데이터가 필요합니다. PaddleX의 쓰기용 캐시는 기본적으로 `.cache/paddlex/`를 사용합니다. 가중치 폴더는 용량 때문에 Git에서 제외됩니다. 실행 설정 예시는 [.env.example](.env.example)에 있습니다.

## OCR 벤치마크

정답 라벨이 없는 HF 이미지 데이터셋은 정량 CER 대신 표본 추론과 시각 비교를 사용합니다. 다음 명령은 `hakuuung/HybridOCR_CustomNotes`에서 seed 42로 이미지 3장을 스트리밍 무작위 추출하고, PP-OCRv6와 PaddleOCR-VL 1.6 결과를 이미지 위에 표시합니다. split은 `train`이 있으면 자동 선택하고, 없으면 첫 split을 사용합니다.

```bash
python -m src.evaluation.benchmark_ocr_visual \
  --dataset hakuuung/HybridOCR_CustomNotes \
  --samples 3 \
  --seed 42 \
  --output-dir data/benchmark/hf_visual
```

각 표본마다 모델별 어노테이션 PNG와 두 모델 및 원본을 세로로 묶은 `*_comparison.png`가 저장됩니다. 전체 OCR 문자열, 추론 시간, 오류와 이미지 경로는 `benchmark_report.json`에 기록됩니다. CLOVA는 기본 호출하지 않습니다. 데이터셋에 다른 split을 쓰거나 API 비용을 감수해 CLOVA도 실행하려면 `--split` 또는 `--engines paddleocrv6 paddleocrvl16 clova`를 지정하세요. 새 환경에는 `datasets` 패키지가 필요하며 HF 토큰은 private 데이터셋일 때만 설정하면 됩니다.

이미지별 정답 문자열을 파일명 키로 담은 JSON을 준비합니다.

```json
{"sample-01.png": "시작", "sample-02.jpg": "입력값 확인"}
```

다음 Python 호출은 동일한 이미지·정답으로 PP-OCRv6와 PaddleOCR-VL 1.6을 실행합니다. 엔진별 첫 추론 시간(모델 초기화 포함), 이후 평균 추론 시간, CER, 필요한 자원·실행 조건을 `data/benchmark/ocr.json`에 기록합니다. CER가 낮은 후보를 추천값으로 표시하지만 기준 모델 상태는 `pending_team_review`로 남겨 검증 결과를 확인한 뒤 확정합니다. 한 엔진에서 오류가 나도 해당 오류를 결과에 남기고 나머지 엔진을 계속 실행합니다.

```bash
python -c 'from src.cv.ocr_engine import benchmark; print(benchmark("data/raw", "data/processed/ocr_ground_truth.json", "data/benchmark/ocr.json"))'
```

## 평가 데이터 형식

정답과 예측 JSON은 `image`, `elements`, `relations`, `warnings` 필드를 사용합니다. 요소의 `bbox`는 `[x1,y1,x2,y2]` 픽셀 좌표입니다. 흐름 관계의 `source`/`target`은 요소 ID이며, `direction`은 `right|left|up|down|unknown`입니다. `evaluate()`는 타입 일치 IoU 매칭으로 요소를 대응한 뒤 요소별 P/R/F1, 연결 대상·관계 유형 점수, 방향까지 일치하는 연결 점수를 냅니다.

```python
from src.evaluation.evaluate import evaluate
report = evaluate(prediction, ground_truth, iou_threshold=0.5)
```

## 현재 프로토타입의 범위

OpenCV 전처리는 문서 사각형이 두드러지면 원근을 펴고, 약한 기울기와 조명을 보정합니다. PaddleOCR-VL 1.6을 선택하면 현재는 `Spotting:` 경로의 텍스트·위치만 오버레이하며 OpenCV 구성요소 검출과 RADU 관계 복원은 실행하지 않습니다. 비교를 위한 `RADU 복원` 보기는 미실행 상태를 표시합니다. 다른 OCR 엔진이나 OCR 미사용 분석은 기존 검출 경로를 유지합니다. PaddleOCR-VL 분석 시 서버 CMD에 활성 장치, 모델, 입력 크기, 텍스트 줄 수, 유효 박스 수 및 오류 원인을 출력합니다.

웹 작업실은 빈 캔버스에서 순서도를 직접 만들거나, 원본 이미지 위에 분석 결과를 겹쳐 검수하는 두 작업 흐름을 지원합니다. 빈 캔버스에서는 텍스트·처리·판단·시작/종료 도형과 화살표를 배치하고, 두 도형을 연결해 흐름을 구성할 수 있습니다. 이미지 작업은 업로드한 원본을 보존하고 방향 보정·전처리 후 OCR 엔진과 CPU/GPU를 선택해 분석합니다. OCR을 실행하면 `OCR 기준` 보기에서 인식된 텍스트와 위치만, `RADU 복원` 보기에서 같은 OCR 결과에 RADU가 검출한 도형·화살표 및 관계를 더한 결과를 따로 비교할 수 있습니다. 원본·전처리 오버레이·복원·나란히 보기 전환도 유지됩니다. 선택한 요소는 드래그로 이동하고 여덟 핸들로 크기를 조정하며, 도형 이동·크기 변경에 맞춰 연결 화살표의 끝점도 함께 갱신됩니다. Shift/Control 클릭과 빈 영역 드래그로 여러 요소를 선택하고 함께 이동할 수 있습니다. 복사·붙여넣기·복제, 정렬 및 균등 배치, 10픽셀 격자 맞춤도 제공합니다. 텍스트는 더블클릭 또는 속성 패널에서 수정할 수 있고, 종류 필터, 스페이스바 캔버스 이동, 관계의 연결 대상 및 확인·보류 상태 편집, 실행 취소/다시 실행, 화살표 키 미세 이동, 확대/축소와 화면 맞춤도 지원합니다. JSON 파일을 다시 열어 편집할 수 있습니다.

저장된 문서는 웹에서 다시 열고 삭제할 수 있으며 `data/processed/sessions/`에 JSON으로 보관됩니다. 내보내기는 JSON, SVG, PNG, PDF를 지원하며 SVG/PNG/PDF는 요소가 놓인 영역에 맞춰 출력 범위를 계산합니다. 입력 파일 크기와 픽셀 수에는 애플리케이션 상한을 두지 않으며, 시스템 메모리·저장 공간·분석 시간은 실제 자원 한계로 작용합니다. SVG 입력 및 PNG/PDF 출력에는 CairoSVG와 시스템 Cairo 라이브러리가 필요합니다. PDF 원본은 직접 분석 입력으로 지원하지 않습니다. 서버 오류는 화면에 표시되고 분석 실패 시 업로드를 다시 시도할 수 있습니다. API 문서는 `http://127.0.0.1:8000/docs`에서 확인할 수 있습니다.
