# Fixed LoRA page scope audit implementation plan

Goal: reproduce legacy results and separate scope, crop and order on the existing17-page diagnostic set without learning or tuning.
Architecture: immutable input receipt, AI visual review decisions linked to original GT, independent versioned scoring policies, paired crop inference with fixed batch4, and full review gallery.
Spec: user request in this session; no training, alternate adapter, threshold sweep, remote edits or overwrite.

- [x] Read actual code/artifacts, hash base/adapter/processor/settings, reproduce199GT/397pred/201FP and235crop.
- [x] Add tests then implement scope/ignore priority, FP ID coverage, changes, M invariance and order-only invariance utilities in src/evaluation/ocr_page_audit.py.
- [x] Generate all original overlays and201FP contact sheets; visually inspect every item and17pages, freeze AI review decisions and versioned GT with unresolved text held explicitly.
- [x] Score legacy and new partial-audited scope; preserve mixed valid+ignore predictions, track unresolved spans and incomplete page coverage.
- [x] Fixed batch4 BF16 selected adapter inference only for required C0/C1/P1 crops; same paired batch membership, repeat representative batch and bind all inputs.
- [x] Compute detection/crop/order/page diagnostics, denominators and remaining uncertainty; create CSV/JSON/HTML, report, run tests and independent review.

Policies fixed before diagnostic scoring: IoU.5 maximum-cardinality matching; ignore only>=.5 prediction coverage with no intersection with valid GT; mixed overlaps retained and tagged. No text similarity in matching/order. M requires unique geometric correspondence, no uncertain/split/merge neighbors. O1 orders M solely by audited GT order; same predictions/strings. No forced additive attribution.

Work in current tree using new files/output only because existing untracked experiments and read-only.git must be preserved. Do not commit or move unrelated work. All AI review is provisional; no independent human certification. Partial scope and ambiguous regions must remain visible in report.

Actual outcome: legacy fully reproduced; all17 originals/all201 FP reviewed by AI, not humans; versioned scope306, raw103 upstream exclusion disclosed. Actual C0/C1 138 each plusP1 306 and8 repeat inputs; no optimizer. Final regression54pass; global LoRA suite7dependency import errors; main-env global suite interrupted in hybrid test (not claimed passing). Independent code review findings resolved through full change trace and conditional-scope disclosure. GT v1/v2 and initial/final FP labels preserved.
