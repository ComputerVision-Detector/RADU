# V2 OCR data pipeline implementation plan

Goal: immutable V2 snapshot, audited original-image crops, provisional grouped splits and three separate OCR evaluations.
Architecture: add src/ocr_data (snapshot, audit/build, split) and src/evaluation/ocr_v2.py; reuse annotation_images and ocr_metrics. Keep Qwen structure training and production OCR adapters unchanged.
Spec: user request dated 2026-11-09.

Constraints: no training, push, remote label writes, weight downloads or user-file replacement. text alone is GT. Originals must match dimensions and SHA256. Preserve multiline/formula strings in JSONL. All heuristic types remain suggestions. Hold pages needing review for full-page evaluation; excluded regions are also masked in predictions. Test is reserved; development smoke uses train only.

- [x] Write regression tests for integrity, exclusions, split leakage, edit counts, one-to-one matching, empty/failure cases; run RED.
- [x] Snapshot latest remote commit and exact bytes, catalog, asset provenance; restore originals at frozen revision into new local output.
- [x] Implement audit, review queue, original crops and overlays, grouped pilot splits and JSONL model lists. Recompute counts and inspect hf-0011/hf-0104 visually.
- [x] Implement saved-prediction detection, recognition-only and full-page evaluators; run GREEN and synthetic evaluator checks.
- [x] Run original-image conversion, representative visual checks, development-only inference if runtime permits; record evidence and commands.
- [x] Review implementation against edge cases, rerun relevant regression suite, document remaining review and limitations.

Review focus: malformed IDs/geometry; missing/corrupt originals; transitive group leakage and similar images; exclusion regions overlapping valid ones; prediction omissions and failed inference remain errors rather than disappearing from denominators.

Decision: work in current checkout with additive files only; existing untracked projects and checkpoint directories are untouched. No dependency installation needed unless import checks reveal a missing package.

Execution ledger:
- RED: initial dataset/metric tests failed because src.ocr_data did not exist; implementation yielded 10/10, then 14/14 integration/edge tests.
- Snapshot: remote 50aca983… confirmed; label SHA256 dc178e259b69180507176b80fd17696e6d99a2ebc91beef007af6a117e4f5299. Restored 163 original PNGs from frozen cached data; no model downloads.
- Independent review: requesting-code-review reviewer identified dataset/prediction binding, page inference original checks, snapshot binding and exact crop-pixel validation gaps. Five regression tests failed (4 failures/1 missing interface), then passed after fixes. Related regression suite: 35/35.
- Visual finding: hf-0011 fractions and hf-0104 visible handwriting confirmed. Representative hf-0000 overlay revealed unboxed title/sticky handwriting; add page hold and rebuild pilot-v2. Retain pilot-v1 as intermediate evidence.
- Ruling: broad LaTeX trace detector includes dollar syntax and yields 434 vs prior 418; persist exact rule and keep all semantic types suggestions. No automatic text correction.
- Local PP-OCRv6 CPU recognition-only 2-crop smoke succeeded on pilot-v1. Those crops belong to subsequently held hf-0000; retain evidence but rerun on eligible pilot-v2 train crops. PaddleOCR-VL full-page GPU smoke unavailable (CUDA device count 0).

Final: pilot-v2 generated 3,569 crops from 147 eligible pages; 16 pages/339 boxes held. Full integrity validation passed for 163 originals and all crops. CPU recognition-only smoke succeeded on two eligible train crops (hf-0001). PaddleOCR-VL token-screened 3,045 task-list rows with no unknown/overflow; max label length 70 tokens. Related suite 36/36 green. No training, push, merge, commit or user-file replacement. Final evidence: docs/ocr_v2/RESULTS.md.
