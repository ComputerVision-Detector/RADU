# V2 OCR actual fine-tuning execution

User supplied the full experiment specification and authorized implementation choices and actual execution. Continue in this checkout, preserving all pre-existing untracked work; isolate new experiment outputs and Python entry point. No push, deployment, label writes, or paid resources.

- [x] Inspect branches, resources, existing dataset and upstream Korean recognition support.
- [x] Test strict label screening, per-page comparisons and checkpoint compatibility.
- [x] Freeze official PaddleOCR code/dictionary/weights and reuse verified original-pixel V2 crops with fixed page-group splits.
- [x] Run train-only smoke: finite CTC loss, update, checkpoint reload, export equivalence.
- [x] Run bounded full-train CTC recognition fine-tuning (official backbone/CTC head; unused NRTR head preserved), select by validation micro CER.
- [x] Lock selection, evaluate before/after once on identical test crops and fixed detector regions, save errors/bootstrap/runtime and report limitations.

No dictionary extension or label replacement. Unsupported characters are held explicitly. Test cannot select settings. Missing writer metadata means page-group pilot only. Fixed detector and identical preprocessing isolate recognition effects. Whole-page evaluation must retain unmatched predictions and missing GT, with formula/symbol GT and predictions excluded consistently.

Completed evidence: docs/OCR_V2_FINETUNE_RESULTS.md, docs/ocr_v2_finetune/, data/experiments/ocr_v2_korean/.
Independent code review found a provenance gap (prepare width768 vs executed width832); fixed new-run receipts/hash guards and explicitly marked this run's reconstructed receipts as post-execution. Fresh-process resume verified step1177. 28 tests passed. Model was not retuned after test. Crop CER improved, whitespace-free and page CER worsened; do not replace operational OCR.
