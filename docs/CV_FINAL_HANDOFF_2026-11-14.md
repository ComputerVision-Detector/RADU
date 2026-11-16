# RADU CV 최종 인계 — 2026-11-14

CV 최종 구성은 **현재 bbox 라우터를 사용하는 자동 모드**와 **기존 고정 LoRA를 사용하는 한국어 전용 모드**다. 기존 모델·GT·검출·crop·읽기 순서·관계 분석은 유지했다. 최근 한국어 보호/precision/균형 학습 CNN 후보는 운영에 연결하지 않고 진단 산출물로 보존했다.

## 접속과 사용

- 워크스테이션: `user@100.116.246.125`
- 현재 웹: <http://100.116.246.125:8860>
- 실행 기준 경로: `/home/user/Desktop/RADU`
- 두 최종 모드는 GPU 0을 자동 선택한다. 기존 엔진으로 전환하면 이전 CPU/GPU 선택을 복원한다.

| 웹 선택 | API engine ID | 인식 경로 |
|---|---|---|
| 자동 모드 · 혼합 문서 — 웹 기본 | `paddleocrvl16routed` | 현재 HOG bbox 라우터가 기존 LoRA와 영문·도식 LoRA 중 하나를 선택 |
| 한국어 전용 모드 · 기존 LoRA | `paddleocrvl16lora` | 검출한 모든 텍스트 영역을 기존 LoRA로 처리 |

한국어 전용은 **모델 경로 고정**이라는 뜻이며 영문을 지우거나 한국어 인식을 완벽히 보장하는 기능이 아니다. 한국어 문서임을 알고 입력할 때 선택한다. 혼합 문서는 자동 모드를 기본으로 사용한다.

기존 `none`, `paddleocrv6`, `paddleocrvl16`, `clova` 엔진 ID와 API 계약은 유지했다. API에서 engine을 생략할 때의 기존 `none` 기본값도 유지했으므로, 직접 호출하는 클라이언트는 모드를 명시한다.

```text
POST /api/analyze?ocr_engine=paddleocrvl16routed&device=gpu:0
POST /api/analyze?ocr_engine=paddleocrvl16lora&device=gpu:0
```

이미지 바이트를 body로 보내고 `Content-Type`, `X-File-Name`을 지정한다. 기존 JSON의 `ocr.engine`, `ocr.lines`, `ocr.metadata`, `elements`, `relations` 계약을 그대로 사용한다. 한국어 전용의 `ocr.metadata.adapter_sha256`와 자동 모드의 parent/english adapter SHA로 실제 모델을 확인할 수 있다. 분류 점수는 OCR 정답 신뢰도가 아니다.

## 고정 모델·검증 기록

| 구성 | 경로 / SHA-256 |
|---|---|
| 기존 한국어 중심 LoRA | `data/experiments/ocr_v2_vl_lora/train/best-adapter` · `ab1bfcfba5c0711ebfc82e83432eca13b47fbb48ca0800d9e7b4d590e15421e5` |
| 자동 모드의 영문·도식 LoRA | `/home/data/ocr_radu_dual_route_20261113_01/epoch-2-adapter` · `9e786e53c6502c8a37f90b4614c1d568a90d8beda164968fc7f48cd4130b0f05` |
| Base | `models/vision_language/PaddleOCR-VL-1.6/model.safetensors` · `85a479d506a11e724e7285d395c551be69f41dbc16b6342d3cacfb189aed71db` |
| 한국어 고정 추론 binding | `configs/ocr_lora_backend.lock.json` — 기존 기록 유지 |
| 자동 모드의 원래 모델/정책 ledger | `data/experiments/ocr_radu_dual_route_20261113_01/deployment.json` — 기존 기록 유지 |
| 이번 웹 버전 binding | `configs/ocr_cv_final.lock.json` — `radu-cv-final-2026-11-14` |

웹 파일까지 고정하던 이전 배포 검증을 이번 UI 버전에 연결하기 위해 별도 release binding을 추가했다. 원래 deployment SHA, frozen 모델/정책 파일, 두 expert identity는 계속 검사한다. 승인된 설치 파일 집합과 hash도 검사하며 불일치하면 추론 전에 실패한다. 원래 모델 기록을 덮어쓰거나 검사를 생략하지 않았다. Decoder/seed/해상도/토큰 한도·분류 정책을 바꾸지 않았다. 이후 index.html/app.js 등 승인 설치 파일을 변경하면 이번 release hash도 재검증·갱신해야 한다. 원래 모델/GT/실험 ledger는 그대로 유지한다.

실행 경로와 기존 학습 binding은 위 워크스테이션 경로를 기준으로 한다. 다른 서버나 경로로 이식할 때는 환경·절대 경로 binding을 별도로 재검증해야 한다. Mac 작업본은 소스·결과 검토용이며 모델과 GPU 실행 환경은 워크스테이션에 있다.

## 이번 동작 검증

실제 RADU 화면에서 파일을 선택하고 분석 버튼을 눌러 HTTP→검출→고정 OCR→Text/관계 결과를 확인했다. 모델 응답은 mock하지 않았다.

| 입력 | 모드 | OCR 영역 | 실제 경로 | 저장·재조회 |
|---|---|---:|---|---|
| 한국어 필기 `hf-0137` | 자동 | 83 | parent 56 / diagram 27 | 통과 |
| 한국어 필기 `hf-0137` | 한국어 전용 | 83 | 기존 LoRA만 사용 | 통과 |
| 영문 도식 `ex07_writer0088` | 자동 | 20 | parent 1 / diagram 19 | 통과 |
| 영문 도식 `ex07_writer0088` | 한국어 전용 | 20 | 기존 LoRA만 사용 | 통과 |

총 4조건/206개 영역을 추론했고 같은 입력의 bbox와 순서가 일치했다. 자동/한국어 전용의 GPU0 선택, 기존 엔진 전환 시 CPU/GPU1 복원, 진단 후보 미노출, 화면 Text 수, 저장 후 OCR 데이터 유지, JavaScript 오류 없음을 확인했다. 초기 자동 모드의 추론은 정상이었으나 DevTools 응답 캐시 수집이 실패해 저장된 결과를 복구했다. 그 추론을 재사용하여 불필요한 재추론을 하지 않았다.

- 새 모드·release 회귀 6개, 기존 LoRA/API 13개, 기존 분석 6개 통과.
- 기존 editor-model 4개 통과(Node VM의 Deno.test adapter로 실행).
- 이미지 업로드 회귀 10개 중 **HEIF 디코더 미설치 1건 실패**. 수정 전 backend에서도 동일하게 재현됐으며 환경/패키지는 변경하지 않았다. 현재 검증 입력은 PNG/JPG이고, HEIF는 PNG/JPG로 변환해 사용한다.
- 원본·모델·GT·진단 기록 29,908개 파일의 변경 없음을 검증했다. 실제 결과와 서버 로그는 아래 검증 폴더에 보관한다.

## 성능 해석과 남은 한계

이번 작업은 웹 연결/모드 선택의 기능 검증이다. 앞선 test 성능 비교를 새 독립 test나 이번 웹 전체 페이지 정확도로 바꾸어 표현하지 않는다.

- 현재 자동 경로의 이전 고정 test: native FC-A CER 14.77%, native hdBPMN 13.25%, 고정 수식·코드 전사 22.45%. 검출 GT 유효 집합은 FC-A 14.69%, hdBPMN 10.12%였다.
- 한국어 보호 후보들은 일부 오분기를 줄였으나 영문·수식 손실과 validation 미채택이 남아 운영 교체하지 않았다. 이번 웹 자동 경로는 그 CNN들을 사용하지 않는다.
- 한국어 전용도 OCR 오류가 남는다. 원본 대조·편집을 전제로 사용한다.
- 수식은 일반 텍스트 전사 범위다. LaTeX 구조·분수/첨자 구조 정확도는 검증하지 않았다.
- Bbox/crop GT 범위가 불확실한 입력은 성능 평가에서 N/A다. 검출 한국어 GT 표본이 작고 이미 살펴본 test를 재평가했으므로 새 인물·문서 일반화를 주장하지 않는다.
- 업로드 입력은 이미지다. PDF 등 기존 미지원 입력의 동작은 변경하지 않았다.

진단 보관: `data/experiments/ocr_korean_router_guard_20261113_01`, `ocr_korean_router_precision_20261113_01`, `ocr_korean_router_balanced_20261114_01`. 모델·평가·hash 기록은 그대로이며 역사적 재현은 각 당시 고정 소스 snapshot과 분리된 환경을 기준으로 한다.

## 실행·소스·검증 위치

현재 RADU 전용 포트는 8860이다. 기존 8000의 다른 서비스는 변경하지 않았다. 재시작이 필요한 경우 먼저 현재 RADU PID/포트 사용을 확인한다.

```sh
cd /home/user/Desktop/RADU
.venv/bin/python -m uvicorn src.web.backend.app:app --host 100.116.246.125 --port 8860 --workers 1
```

이번 검증/전후 snapshot: `/home/data/radu_cv_final_20261114_01`.
`server-pid.json`, `web-e2e.json`, `http-results/`, screenshots, unit logs, release binding, `verification.json`을 참조한다. 원래 모델·GT·최근 실험을 재학습하거나 서비스 자동 시작을 설치하지 않았다. 다른 사람의 프로세스를 종료하지 않았다.

Mac 소스에는 이전 서버의 OCR 의존 코드와 이번 모드 UI를 동기화했다. 기존 dirty 변경을 되돌리지 않았고 Git push는 하지 않았다.

## 팀원 전달문

CV 파트는 이번 구성으로 최종 정리했습니다. RADU 웹에서 기본 ‘자동 모드’와 ‘한국어 전용 모드’를 선택할 수 있습니다. 자동 모드는 텍스트 영역별로 기존 LoRA와 영문·도식 LoRA를 선택하고, 한국어 전용 모드는 모든 영역에 기존 LoRA를 적용합니다. 두 모드는 GPU0에서 동작합니다.

기존 LoRA와 GT, 검출·crop·읽기 순서·RADU 구조화 경로는 고정했습니다. 최근 한국어 보호/균형 학습 후보는 일부 개선에도 validation 기준을 통과하지 못해 운영에 넣지 않고 진단용으로 보관했습니다. 실제 웹에서 한국어·영문 이미지 두 모드 추론과 저장·재조회를 검증했습니다.

접속은 http://100.116.246.125:8860 이며 상세 경로·checkpoint·검증 결과는 CV 최종 인계 문서에 정리했습니다. 입력은 PNG/JPG를 사용해 주세요. HEIF는 현재 디코더가 없어 변환이 필요합니다. OCR은 원본 대조가 필요하고 수식 구조/LaTeX 정확도를 보장하는 단계는 아닙니다.
