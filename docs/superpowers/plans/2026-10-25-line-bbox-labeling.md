# Line bbox V2 Labeling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` or `superpowers:subagent-driven-development` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 기존 HybridOCR 라벨링 앱에 V1 전체 전사와 독립된 V2 텍스트 줄 bbox 정답 수집·검수·저장을 추가한다.

**Architecture:** 현재 `labeling/data/live.json`을 V1로 유지하고, V2 상태는 `labeling/data/live-v2.json`과 별도 로컬 상태 파일에 저장한다. 텍스트 후보는 기존 Python OCR 엔진으로 오프라인 생성해 정적 sidecar로 제공하고, 브라우저에서 후보 박스와 정답 문구를 수정한다.

**Tech Stack:** Node.js 22, 기존 vanilla HTML/CSS/JavaScript 앱, Python, `src/cv/ocr_engine.py`의 PaddleOCR 어댑터.

**Spec:** `docs/superpowers/specs/2026-10-25-line-bbox-labeling-design.md`

## Global Constraints

- V1은 `RADU/labeling-data`의 `labeling/data/live.json`이다. V1의 163개 상태와 기존 전사 내용을 V2 저장이 수정하지 않는다.
- 코드 변경은 `RADU/labeling`에서 진행한다. 기존 V1 데이터는 별도 `RADU/labeling-data` 브랜치의 `labeling/data/live.json`에서 그대로 읽고 유지하며, V1 전사를 새 파일로 재마이그레이션하거나 다시 쓰지 않는다.
- V2는 `labeling/data/live-v2.json`을 사용하고 163개 이미지 모두 bbox 상태 `unlabeled`로 시작한다.
- 브라우저의 미저장 초안 키도 버전을 포함해 V1/V2 사이에 복원되지 않도록 한다.
- bbox는 원본 이미지 pixel 기준 `[x1, y1, x2, y2]`이며 이미지 원본 `width`, `height`를 함께 저장한다.
- OCR prediction과 사람의 ground truth `text`는 별도 값이다. prediction은 정답으로 자동 복사하지 않는다.
- bbox 검수 완료는 비어 있지 않은 정답 박스가 있거나, 사람이 `no_text: true`를 명시한 경우만 허용한다.
- OCR 추론은 오프라인 명령으로 실행한다. 웹 요청이 Python OCR 모델을 구동하지 않는다.
- 이 작업에서는 자동화 테스트를 추가하거나 실행하지 않는다. 빌드와 코드 변경 검토로 확인한다.

## Review Focus

- V2 상태 파일이 없는 최초 실행은 V1 데이터를 복사하지 않고 빈 V2 상태로 시작해야 한다. **Task 1**에서 초기 상태 생성 흐름을 확인한다.
- V1 저장, V2 저장, V1/V2 내보내기가 서로 다른 파일과 버전 표시를 유지해야 한다. **Task 2**에서 API 라우팅과 state path 구성을 검토한다.
- WebP 화면 좌표와 원본 픽셀 좌표가 서로 다를 때 bbox 위치가 원본 기준으로 저장되어야 한다. **Task 3**에서 좌표 변환을 검토한다.
- 후보 파일 재생성으로 기존 사용자 bbox가 덮어써지면 안 된다. **Task 3**에서 기본 출력 경로의 overwrite 방지를 검토한다.
- 빈 bbox 목록만으로 이미지가 검수 완료되면 안 된다. **Task 2**에서 `no_text` 조건을 검토한다.

---

### Task 1: V2 라벨 상태와 분리 저장소

**Files:**
- Create: `labeling/lib/bbox-state.mjs`
- Modify: `labeling/lib/github-store.mjs`
- Modify: `labeling/lib/file-store.mjs`
- Modify: `labeling/server.mjs`
- Modify: `labeling/compose.yaml`
- Modify: `labeling/Dockerfile`

**Interfaces:**
- Produces `createBBoxState(catalog=seed)`, `reconcileBBoxState(stored, catalog=seed)`, `listBBoxState(state, client, now)`, `getBBoxItem(state, id, client, now)`, `mutateBBox(state, action, body, now)`, and `validateBBoxState(state, catalog=seed)`.
- V2 item shape: `{ id, image, bbox_annotation, claim, history }`.
- `bbox_annotation` shape: `{ status, revision, annotator, notes, no_text, proposal_version, boxes }`.
- Box shape: `{ id, bbox, text, prediction, order, uncertain }`.
- V1 keeps its existing state shape and store path. V2 uses `labeling/data/live-v2.json`; local V2 state uses `.runtime/state-v2.json` with a separate backup directory.
- Make `GitHubStore` and `FileStore` accept injected path/initial-state/reconcile/validation options while retaining their current V1 defaults.

- [x] **Step 1: Define V2 state creation and reconciliation**

Create 163 V2 items from the existing image catalog without copying V1 `text`, notes, status, annotator, or history. Preserve existing V2 labels when the catalog adds images; reject unknown IDs and incompatible state schemas.

- [x] **Step 2: Define V2 label validation and mutations**

Validate unique box IDs, integer source-pixel bounds, line order, status, text length, author/client identity, and revision. Allow `verified` only when a non-empty GT box exists or `no_text` is explicitly true. Keep V2 claim and revision fields independent of V1.

- [x] **Step 3: Configure separate V1/V2 store instances**

Keep the existing `.runtime/state.json` and `labeling/data/live.json` defaults for V1. Configure V2 with `.runtime/state-v2.json` and `labeling/data/live-v2.json`. Give V2 its own local backup directory, and set the self-hosted Docker paths to `/data/state-v2.json` and `/data/backups-v2` so V2 state uses the persistent volume.

- [x] **Step 4: Review version isolation in the store configuration**

Confirm by code inspection that every V2 read/write resolves to the V2 path and its first state contains 163 unlabeled IDs, while V1 still resolves to the unchanged `live.json` path.

### Task 2: Version-aware API and independent exports

**Files:**
- Modify: `labeling/api/labeling.mjs`
- Modify: `labeling/tools/state.mjs`
- Modify: `labeling/README.md`
- Modify: `labeling/START_HERE.md`

**Interfaces:**
- `handlerFor(v1Store, v2Store=v1Store)` keeps legacy requests defaulting to V1.
- Requests with `version=v2` route list/item/claim/next/renew/release/save/export to the V2 store.
- V2 export contains `{ schema_version: 1, dataset_version: "v2", samples }`; V1 export is tagged `dataset_version: "v1"`.

- [x] **Step 1: Add explicit version routing**

Preserve existing V1 actions unchanged when `version` is omitted or `v1`. Dispatch `version=v2` to V2 state functions. Reject unknown versions with a clear 400 response.

- [x] **Step 2: Add V2 save and export actions**

Save only `bbox_annotation` fields to the V2 store. ZIP names and JSON payloads must identify V2. A V2 save must not call the V1 store.

- [x] **Step 3: Add V2 backup and operator documentation**

Document V1/V2 data paths and separate backup/restore commands for self-hosted deployments. Document that V1 is the existing page transcription state and V2 starts empty.

- [x] **Step 4: Review API version isolation**

Inspect each V2 handler branch to confirm it only reads and writes the V2 store. Inspect both export payloads to confirm their version labels and fields remain separate.

### Task 3: Offline OCR proposal generation

**Files:**
- Create: `src/evaluation/text_box_candidates.py`
- Modify: `labeling/build.mjs`
- Modify: `labeling/server.mjs`
- Modify: `labeling/compose.yaml`
- Modify: `labeling/Dockerfile`
- Modify: `labeling/README.md`

**Interfaces:**
- CLI: `python -m src.evaluation.text_box_candidates --image-dir labeling/assets --engine paddleocrv6 --output labeling/data/text-box-proposals.json`.
- CLI output is a proposal manifest with schema version, engine/model metadata, source image hash and size, and per-line `{ id, bbox, prediction, confidence, order }`.
- The default output must refuse to overwrite an existing file. A caller can choose a new `--output` path to create another proposal version.
- Build and local server use `LABEL_PROPOSALS_FILE` to select an approved versioned manifest; the browser URL remains a stable sidecar URL.

- [x] **Step 1: Implement candidate conversion from existing OCR engines**

Use `recognize(image, engine)` from `src/cv/ocr_engine.py`. Convert each returned polygon into an axis-aligned `xyxy` box and scale WebP coordinates into catalog source-pixel coordinates. Include prediction text and confidence when available.

- [x] **Step 2: Persist candidate metadata and protect prior generations**

Write each image's source hash, original dimensions, engine, model metadata, and candidate lines. Stop if the target file exists; do not mutate saved V2 annotations.

- [x] **Step 3: Serve the optional proposal sidecar**

Copy the selected sidecar (`LABEL_PROPOSALS_FILE`, default `labeling/data/text-box-proposals.json`) into `dist/data/` when present. Allow the local server and Vercel static output to serve it. The UI must still work with no proposal file and allow manually drawn boxes.

- [x] **Step 4: Document OCR prerequisites and output versioning**

Document local Python dependencies, model-weight paths, the command, and the separate `--output` path needed for a new proposal generation.

### Task 4: V2 bbox editor and version selector

**Files:**
- Modify: `labeling/index.html`
- Modify: `labeling/app.js`
- Modify: `labeling/app.css`

**Interfaces:**
- UI version selector has `v1` and `v2`; V1 keeps its current whole-image transcription workflow and V2 opens the bbox workflow.
- New V2 view uses the existing `api()` helper with `version=v2` for every request.
- Proposal manifest entries remain read-only suggestions. “후보 가져오기” copies candidate geometry into the editable draft with blank GT text and the original prediction preserved separately.

- [x] **Step 1: Add V1/V2 navigation and V2 progress state**

Keep current controls and list behavior under V1. Under V2, use bbox status for filters, counts, next-image selection, and progress. Make V2 the initial mode for new sessions while keeping V1 one click away.

- [x] **Step 2: Add the source-coordinate SVG overlay**

Render candidates and saved V2 boxes over `source-image`. Convert source-pixel boxes to rendered WebP dimensions and apply the existing image zoom/pan/rotation behavior. Keep candidate and saved-label visual styles distinct.

- [x] **Step 3: Add box editing and per-box ground truth**

Support selecting, adding by drag, deleting, moving, and resizing a box. Connect the selected box to its prediction, blank GT field, reading order, and uncertain flag. Provide explicit copy from prediction to GT. Keep V1 text available only as a separately labeled reference.

- [x] **Step 4: Add V2 save, review, and completion controls**

Save draft and verified states with V2 revision checks. Require either at least one non-empty verified box or an explicit `no_text` choice. Keep unsaved input in the browser on save failure.

- [x] **Step 5: Review mode-specific UI wiring**

Include the dataset version in each local draft key. Inspect all V1 and V2 actions to confirm the correct version query is sent, the correct progress state is shown, V1 drafts cannot appear in V2, and a V2 save leaves the V1 view's contents untouched.

### Task 5: Package and handoff updates

**Files:**
- Modify: `labeling/README.md`
- Modify: `labeling/START_HERE.md`
- Modify: `labeling/build.mjs`

- [x] **Step 1: Document the V1/V2 workflow and data schema**

Explain that the existing 146 verified page transcriptions are V1, the V1 draft/unlabeled states are preserved, and new line-level bbox labels are V2. Describe proposal files, annotation JSON, and separate exports.

- [x] **Step 2: Build the static output**

Run `npm run build` from `labeling/`. Expected: build completes and `dist/` contains the HTML, JS, CSS, assets directory, and the proposal manifest if one exists.

- [x] **Step 3: Inspect the final change list and handoff instructions**

Review the changed file list and provide the V2 annotation command, start command, and data-branch storage paths to the user. Final review findings on store isolation, save-time edits, image bounds, and proposal fingerprints were fixed and re-reviewed.
