# HybridOCR 공동 라벨링 안내

작업 저장소: https://github.com/ComputerVision-Detector/RADU

공유 기준 브랜치: **feat/hybridocr**. 모델·웹·평가 코드, 작업 기록, 정답 JSON 122개와 5장 실험 결과를 이 브랜치에서 관리한다. 모델 가중치·가상환경·원본 이미지 1.7GB는 Git에 넣지 않는다. 이미지는 아래 명령으로 고정된 HF revision에서 복원한다.

## 1. 코드 받기

```bash
git clone --branch feat/hybridocr https://github.com/ComputerVision-Detector/RADU.git
cd RADU
python3 -m venv .venv
```

아래 명령의 `python`은 생성한 가상환경을 활성화한 Python이다. Linux/macOS는 `source .venv/bin/activate`, Windows PowerShell은 `.venv\Scripts\Activate.ps1`로 활성화한다. JSON을 GitHub 웹 편집기로 수정할 때에는 Python 설치가 필요 없지만, 원본 이미지 확인은 별도로 필요하다.

## 2. 원본 이미지 복원 (GPU 불필요)

```bash
python -m pip install -r requirements-annotation.txt
python -m src.evaluation.annotation_images
```

`data/benchmark/qwen_ablation_pilot/images/hf-0000.png` ~ `hf-0121.png`가 생성된다. 모델 추론 없이 HF parquet에서 이미지만 복원한다. dataset revision은 `7f97f382d9d82aa6b21e9b617b46218f1ec4b0aa`로 고정한다. 해시를 검증하며 이미 존재하는 다른 이미지를 덮어쓰지 않는다. PNG 해시가 다른 환경에서는 `requirements-annotation.txt`의 Pillow 버전을 확인한다. 기존 평가와 다른 픽셀·파일을 동일한 입력으로 취급하지 않는다.

전체 다운로드 전에 자신의 담당 이미지만 복원할 수도 있다.

```bash
python -m src.evaluation.annotation_images --ids hf-0000 hf-0001 hf-0002
```

공개 데이터셋 접근이 가능하면 HF 로그인은 필요 없다. 접근 제한이 있는 계정에서는 로컬 HF 인증을 사용한다. 토큰을 Git에 올리지 않는다. 이미지 다운로드와 인코딩에는 저장 공간·시간이 필요하며 모델용 패키지는 필요 없다.

## 3. 담당 번호와 작업 브랜치

팀에서 담당 이미지 번호를 먼저 합의한다. 한 이미지 JSON은 한 사람이 편집하고 다른 사람이 검수하는 방식을 권장한다. 기존 verified 5장(hf-0003, hf-0014, hf-0035, hf-0081, hf-0094)은 중복 작업에서 제외한다. 개발용과 독립 최종 평가용 분할도 규칙을 조정하기 전에 팀에서 확정한다. 아직 담당자나 평가용 ID는 자동 배정하지 않았다.

```bash
git switch feat/hybridocr
git pull --ff-only origin feat/hybridocr
git switch -c annotation/your-name-batch-01
```

GitHub 웹에서도 `feat/hybridocr`에서 개인 브랜치를 만들고 담당 JSON만 수정할 수 있다.

## 4. 전사 파일 작성

[이미지별 라벨 폴더](../data/benchmark/qwen_ablation_pilot/labels)의 같은 번호 JSON을 연다.

```json
{
  "id": "hf-0000",
  "text": "이미지에서 읽은 한국어와 English\n다음 줄",
  "status": "draft",
  "annotator": "작성자 이름",
  "notes": "특이사항이 없으면 빈 문자열"
}
```

- 원본만 보고 전사한다. 모델 출력을 복사하지 않는다.
- id와 파일 이름을 바꾸지 않는다. `hf-0014`처럼 네 자리 번호를 사용한다.
- 미작성: `text: null`, `status: "unlabeled"`. 작성 중: `draft`. 사람 검수 완료: `verified`.
- 특이사항 없음: `notes: ""`. 따옴표는 `\"`, 줄바꿈은 `\n`, 탭은 `\t`로 표시한다.
- 한국어·영문·숫자·기호와 읽기 순서를 유지한다. 잘못 적힌 원문을 지식으로 교정하지 않는다.
- 포스트잇·괄호·여백 주석 포함 범위, 곱셈 기호, 반복부호 등 표기 기준을 팀에서 먼저 통일한다.
- 판독 불가·미완성 페이지는 draft로 두고 notes에 사유를 남긴다. 빈 verified 전사는 허용하지 않는다.
- 표는 합의된 행 순서·셀 구분으로 텍스트를 전사한다. 화살표 경로·셀 좌표·연결 대상은 별도 구조 라벨이 필요하며 텍스트 라벨만으로 구조 정확도를 평가하지 않는다.

`references.json`은 최초 5장 평가의 정답 스냅샷이다. 공동 작업에서는 **labels/*.json을 원본으로 수정**하고 통합 파일은 필요할 때 생성한다. 기존 실험 결과를 함께 손으로 수정하지 않는다.

## 5. 검사와 Pull Request

JSON 문법·ID·상태·빈 verified 전사를 검사한다. 이 명령은 표준 Python만 사용하며 이미지는 없어도 된다.

```bash
python -m src.evaluation.annotation_files check \
  --directory data/benchmark/qwen_ablation_pilot/labels \
  --manifest data/benchmark/qwen_ablation_pilot/additional_images.manifest.json
```

변경한 담당 파일만 커밋한다. 아래 경로는 예시이므로 실제 담당 번호로 바꾼다.

```bash
git add data/benchmark/qwen_ablation_pilot/labels/hf-0000.json
git commit -m "docs: transcribe hf-0000"
git push -u origin annotation/your-name-batch-01
```

GitHub에서 **base를 feat/hybridocr**, compare를 개인 작업 브랜치로 정해 PR을 연다. 담당 번호, 작성자·검수자, 제외 영역·판독 불가 사항을 기록한다. 브랜치에 GitHub Actions를 추가해 같은 라벨 검사를 실행하도록 했다. 실제 원문과 일치하는지는 사람이 확인한다.

## 6. 평가용 통합

평가 담당자가 필요한 manifest에 해당하는 라벨만 합친다.

```bash
# 기존 5장에 대응하는 현재 개별 라벨을 별도 통합
python -m src.evaluation.annotation_files merge \
  --directory data/benchmark/qwen_ablation_pilot/labels \
  --manifest data/benchmark/qwen_ablation_pilot/manifest.json \
  --output data/benchmark/qwen_ablation_pilot/references.from_labels.json

# 122장 전체 보관용 통합 (미라벨도 남음)
python -m src.evaluation.annotation_files merge \
  --directory data/benchmark/qwen_ablation_pilot/labels \
  --manifest data/benchmark/qwen_ablation_pilot/additional_images.manifest.json \
  --output data/benchmark/qwen_ablation_pilot/references.all.from_labels.json
```

추가 117장의 모델 추론은 아직 실행하지 않았다. 전사만 작성해도 기존 5장 결과에 자동 추가되지는 않는다. 새로운 평가 표본 manifest와 같은 이미지 ID의 모델 예측을 준비하고 verified 정답으로 비교해야 한다.

## 7. 기존 실험 확인

- [전체 작업 기록](../HYBRIDOCR_WORK_LOG.md)
- [정답 기반 성능 비교](QWEN_ABLATION_LABELLED_RESULTS.md)
- [표·화살표 구현 범위](STRUCTURE_RULES_IMPLEMENTED.md)

원본 이미지를 복원한 뒤 정적 비교 HTML을 실행할 수 있다.

```bash
python -m http.server 8013 --bind 127.0.0.1 \
  --directory data/benchmark/qwen_ablation_pilot
```

브라우저 주소: http://127.0.0.1:8013/ . GitHub 파일 보기에서는 HTML을 웹페이지로 실행하지 않으므로 위 로컬 서버를 사용한다.
