# OCR CER improvement round 2

User requests ideas AND actual bounded experiments. Preserve original labels, all prior runs, base model and operational path. Use previous selected LoRA as common initialization, reset optimizer identically, fixed full train2513 crops. No test access during experiments/selection; prior test is exploratory only even at final.

## Evidence and hypotheses

Previous validation297: CER14.1733%, errors973: substitutions678/deletions132/insertions163, whitespace insertions65/deletions50. Higher-area62 crops account379 errors/2538 chars. Two validation pages have image-confirmed GT/crop problems: hf-0019 transcription typo/neighbor-line contamination, hf-0095 untranscribed printed footer/header fragments. Quarantine both pages before this round's selection, preserve all297 results as diagnostics. Recompute previous LoRA on the SAME primary validation subset; never compare unlike subsets. No transcription correction or OCR-to-GT substitution.

Bounded hypotheses: A) previous validation was still improving at epoch3, so three additional epochs with lr2e-5 may improve transcription. B) doubling min/max image pixels may preserve fine handwriting better; compare B with A at identical initialization, optimizer, batches, seed and steps. Do not claim B effect is isolated from training unless paired A comparison supports it.

- [x] Write tests for quarantine, inherited-adapter loading/config checks, train/validation no test access.
- [x] Freeze round config, source, parent-adapter SHA, dataset and quarantine list before execution.
- [x] Inference-only validation resolution probe at previous and doubled image pixels (same parent adapter).
- [x] A: parent LoRA +3epochs lr2e-5, original112896/200704 pixels.
- [x] B: same parent LoRA +3epochs lr2e-5, doubled225792/401408 pixels.
- [x] Select parent/A/B solely by minimum primary validation CER. Secondary no-space CER and exact match reported, no post-hoc tuning.
- [x] Lock selection, compare prior vs selected on unchanged exploratory test235 and same detector397 crops. Full validation297 is diagnostic, not selection.
- [x] Save real adapters, logs, hashes, metrics, paired bootstrap, failure cases and report including deterioration/no change.

Use dedicated existing LoRA venv (read-only shared train dependencies plus separate rapidfuzz). No new model download required. Native Transformers official implementation inspected locally; no framework/version changes. All new artifacts under data/experiments/ocr_v2_vl_lora_round2/. No existing file output overwritten. Data policy in round2 manifest is a derived local view, not changes to V2 source or pilot-v2.

Adaptive extension declared BEFORE any test: one .5 parent + .5 selected A exact adapter-delta interpolation, no coefficient sweep. PEFT cat rank16 unit-tested, fifth candidate. This did not beat parent either. Actual results primary244 parent12.7917%, A12.8095%, B13.3262%, interpolation12.9521%. Both full training runs3epochs945steps each completed; parent retained; final test235 CER15.2238% and page84.7647% unchanged. 36 tests pass. Independent review found only receipt/description guards, all addressed; no model/metric retuning after test. Evidence docs/OCR_LORA_ROUND2_RESULTS.md and docs/ocr_lora_round2/.
