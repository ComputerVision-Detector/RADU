# 표·화살표 규칙 구현 및 이미지별 라벨 — 2026-10-21

## 구현 범위

`src/cv/structure_rules.py`에 GPU·추가 모델 없이 실행하는 OpenCV 규칙을 구현했다. 초기 지원 범위는 축에 정렬된 테두리 표와 직선 화살표다. 검출 결과는 자동 확정 정답이 아닌 시각 검수 후보다.

- 표: 이진화, 짧게 끊긴 선 결합, 수평·수직 선 추출, 닫힌 외곽 검사, 행·열 경계 검출.
- 단일 사각형과 가로 줄만 있는 종이는 표로 확정하지 않는다.
- 내부 경계가 거의 없는 구획은 직사각형 조건을 확인해 병합 셀을 구성한다. 불명확한 경계와 비직사각형 병합은 검수 표시를 남긴다.
- 각 셀에 row/col/rowspan/colspan/bbox/text_ids를 저장한다. OCR 영역이 셀에 80% 이상 포함되는 유일한 셀에 할당하며 경계를 걸친 텍스트는 강제로 배치하지 않는다.
- 화살표: OCR 글자 영역 및 검출 표의 선을 별도 마스크에서 제외하고 Hough 선분 끝에 모이는 양쪽 획으로 화살촉을 확인한다. 단방향·양방향을 구분한다.
- 연결: 시작/도착점에서 대상 경계까지 거리 및 진행 방향을 함께 검사한다. 대상 점수 차이가 작거나 멀면 source_id/target_id를 null로 남긴다.
- 구조 좌표는 원본 픽셀 기준이다. 큰 이미지는 최대 변 길이 1600으로 축소 검출 후 환산한다. 기본 임계값은 `RuleConfig`에 모았다. rule_version 및 사용 설정을 검출 결과에 저장한다.

곡선·꺾인 화살표, 테두리 없는 표, 자동 원근 보정과 완전한 손글씨 구조 복원은 미구현이다. 기울기·텍스트 오검출·OCR 누락에 영향을 받는다. OCR 영역 마스크는 글자에 겹친 실제 화살표도 놓칠 수 있다. 페이지 가장자리의 선은 제본·외곽선 오검출을 줄이기 위해 제외하므로 가장자리 화살표도 놓칠 수 있다. 채워진 화살촉은 별도 분류기를 구현한 것이 아니므로 검출을 보장하지 않는다. 구조 정답이 아직 없어 정확도는 미측정이다.

## 파이프라인 연결

`src/hybrid/document.py:layout_candidates`가 규칙 표·화살표를 생성하고 기존 일반 도형 후보와 결합한다. 표에 포함된 일반 박스 후보는 제거한다. 웹 분석과 새로운 ablation 실행에서 OCR 줄을 전달한다.

벡터 문서에서는 셀 메타데이터를 유지하고 Qwen으로 교정된 텍스트를 셀로 이동한다. 확실히 할당하지 못한 텍스트는 원래 요소로 유지한다. 화살표는 최종 벡터 요소에 대해 다시 연결해 삭제된 OCR ID를 참조하지 않도록 한다. 양방향 화살표는 양쪽 marker를 갖는다. 글자 영역 연결은 화살표의 source_id/target_id에 보관하고, 기존 웹 흐름 관계 스키마가 Box 연결만 허용하므로 flows_to의 글자 대상은 null로 남긴다.

새 실험의 소스 해시에 detector와 structure_rules를 포함한다. 기존 5장 예측·정답·CER/WER는 변경하지 않았으며 Qwen 또는 Paddle을 재실행하지 않았다. 새 규칙 적용 후의 OCR 성능 개선을 주장하지 않는다.

단독 실행:

```bash
.venv/bin/python -m src.cv.structure_rules \
  data/benchmark/qwen_ablation_pilot/images/hf-0003.png \
  --ocr data/benchmark/qwen_ablation_pilot/results/hf-0003/ocr.json \
  --output data/benchmark/qwen_ablation_pilot/structure_rules_manual/hf-0003.json
```

OCR 없이도 실행할 수 있으나 글자 획 오검출 억제와 셀 텍스트 할당을 위해 `--ocr`를 권장한다. 기존 5장 CPU 실행 결과·시각 오버레이는 `data/benchmark/qwen_ablation_pilot/structure_rules_v1/`에 별도 저장했다. 이 결과의 표·화살표 개수는 검출 후보 수이며 정확도나 정답 수가 아니다.

## 이미지별 정답 라벨

위치: `data/benchmark/qwen_ablation_pilot/labels/hf-0000.json` ~ `hf-0121.json`.

122개 파일 중 기존 5장은 원문·작성자·notes·추적 필드를 그대로 복사했다. 나머지 117개는 text=null, status=unlabeled, annotator="", notes=""다. 파일 이름 및 id는 이미지 번호와 일치한다. 각 파일의 text를 전사하고 사람이 검수한 뒤 verified로 바꾼다. 기존 통합 정답과 이미 작성된 개별 파일을 자동 덮어쓰지 않는다.

개별 라벨 병합 도구:

```bash
# 기존 5장 평가에 해당하는 라벨만 검증하고 병합
.venv/bin/python -m src.evaluation.annotation_files merge \
  --directory data/benchmark/qwen_ablation_pilot/labels \
  --manifest data/benchmark/qwen_ablation_pilot/manifest.json \
  --output data/benchmark/qwen_ablation_pilot/references.from_labels.json

# 전체 122장 라벨 보관용 병합 (미라벨도 유지, 전체 모델 예측이 있다는 뜻은 아님)
.venv/bin/python -m src.evaluation.annotation_files merge \
  --directory data/benchmark/qwen_ablation_pilot/labels \
  --manifest data/benchmark/qwen_ablation_pilot/additional_images.manifest.json \
  --output data/benchmark/qwen_ablation_pilot/references.all.from_labels.json
```

병합은 manifest ID 순서로 파일을 읽고 ID·상태·문자열 유형·빈 verified 전사를 검사한다. 추가 이미지 라벨을 작성해도 기존 5장 평가에 자동 포함되지 않는다. 해당 이미지들의 모델 예측과 새 평가 manifest가 필요하다. 표·화살표 구조 정답은 현재 텍스트 라벨과 별도로 작성해야 한다.

## 검증

전체 unittest 43개 통과. 추가 7개 검사에서 표·병합 셀·텍스트 배치, 빈 이미지·일반 선·사각형 오검출 억제, 화살표 방향·양방향, 연결 후보 동점 보류, 원본 좌표 환산·셀 교정문 보존, 개별 라벨 보존·병합 오류, 텍스트 연결과 웹 저장 스키마 호환 검사를 수행했다. 실제 손글씨 5장에도 CPU 구조 검출을 실행해 별도 후보 JSON과 오버레이를 생성했다. 원본 모델 출력 해시는 기존 기록과 일치한다.

### 실제 5장 실행에서 확인한 한계

최종 CPU 실행은 hf-0014에서 화살표 후보 7개, 나머지 4장에서 0개를 반환했고 테두리 표 후보는 0개였다. 이는 정답 수가 아니다. 시각 비교에서 제본·종이 선 오검출과 작은 손글씨 화살표 누락을 확인했다. 이 5장에는 구조 정답이 없으므로 precision/recall을 산출하지 않았고 실문서 구조 복원이 완료됐다고 판단하지 않는다. 표·병합 셀은 합성 입력 검사로 기능을 확인했으며 실제 표 이미지에 대한 별도 검증이 필요하다.
