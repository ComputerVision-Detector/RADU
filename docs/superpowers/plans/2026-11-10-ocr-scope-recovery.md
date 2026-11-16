# Fixed LoRA scope recovery implementation plan

Goal: execute priority1 from page audit: GT review and restore legacy-CTC-filtered detector regions, without training/tuning or operating-path mutation.
Spec: user's approved next priority, following fixed-model audit constraints. AI cannot certify independent human review.
Architecture: bind previous auditGTv2 and cached397 strings; rerun SAME local detector with SAME defaults, require exact recovery of every old box/order/crop and total500 before recognition. Recognize ONLY missing103 in a fixed batch4 schedule, append to unchanged historical cache, assemble original detector order. Compare397/500 with identical per-region strings, sameGT and ignore policy. Keep GTv3 review effects separate.
Files: src/evaluation/ocr_scope_recovery.py (stage prepare/detect/recognize/evaluate), configs/ocr_page_audit/scope_review_v3.json (image-only decisions), tests/test_ocr_scope_recovery.py; new data/experiments/ocr_page_audit_scope_recovery_v1 and docs/OCR_SCOPE_RECOVERY_RESULTS.md.

- [x] Tests first: exact detector/cache identity, missing ID uniqueness103, paired shared text invariance, scope-policy without GT inference filtering, GT changes/source linkage, hashes/drift/overwrite.
- [x] Freeze input/base/adapter/detector/config/environment/GTv2; generate enlarged originals for13held FP + truncated website and selected original held GT; AI visual re-review, unresolved remains held.
- [x] Actual detector reconstruction, no threshold or model change; inspect103restored regions and audit their GT scope. Abort main comparison on nonidentical retained geometry instead of forcing counts.
- [x] Actual selectedLoRA recognition of103only + first4cached representative repeat; no optimizer and no output-based choices. Preserve batch4 old397 group membership and new103group membership; cached bank must remain byte-identical for shared IDs.
- [x] Evaluate frozenGTv2 baseline397 vs recovered500 with shared text invariance; then separatelyGTv3 changes if evidence. Record detection/crop/order/CER scope, upstream misses vs errors, ignored/mixed/failure denominators. No sum-of-effects attribution.
- [x] Reproduction code, complete restored103 gallery, GT/changes/manifests/CSV, review-ready human queue, meaningful tests and final report.

Policy: all detector results saved before GT/ignore; evaluation-only ignore identical to prior audit >=50% predicted area inignore AND zero intersection validGT. Spatial original detector order fixed. No NMS/threshold sweep or new recognition model. No new learning, remote writes, push, paid resources. Old artifacts untouched. GTv2 primary remains frozen before reconstruction to measure scope-filter effect, GTv3 is separate provisional AI review, not independent benchmark.

Actual: same detector500 recovered exactly, original397 identity/text unchanged, extra103+8repeat inferred. GTv2 CER30.0615→28.9025 (CI includes0),GTv3 separate28.0843→25.0084; raw detector no GT filtering, operational path unchanged. Reviewed37original crops and all103restored crops by AI; human0 and unresolved13FP persist.60regression pass; full isolated suite164with7dependency errors disclosed. Reviewer confirmed core invariants; post-review review-freeze guard added RED→GREEN, actual run binding independently checked.
