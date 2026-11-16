# Actual PaddleOCR-VL 1.6 LoRA experiment

Follow-up to the user's authorized OCR experiment; implement and execute LoRA and inference, preserve all earlier runs and operational/Qwen models. Use a separate process/venv and output root. The previous test has informed this experiment choice, so follow-up test numbers are exploratory, not an untouched confirmatory evaluation.

- [x] Confirm GPU availability and strict native Transformers loading of cached official weights.
- [x] Test completion-only supervision and language-only LoRA selection; hash data, source and weights before starting.
- [x] Smoke finite loss, nonzero update, adapter save/reload.
- [x] Train all 2513 natural-text candidate crops, at most 3 epochs, select minimum validation CER (297 crops).
- [x] Infer unchanged base and selected LoRA on identical exploratory test crops (235); optionally fixed detector crops; save metrics, failures and examples.
- [x] Verify tests and write honest comparison and commands.

Settings fixed before test: rank8 alpha16 dropout0.05, language q/k/v/o only, AdamW lr5e-5, batch2 accumulation4, bf16 SDPA, seed42, train shuffle, no label replacement/truncation, prompt OCR:, max sequence768, max generation128, image min112896/max200704 pixels. Three full epochs, evaluation each epoch; no post-test retuning. Baseline uses identical native model/processor/decoding. Formula/symbol/held pages stay out.

Actual completion: 3epochs945steps, validation24.22→14.17%, exploratory test235 CER21.61→15.22%, no-space22.15→15.51%, fixed-detector test-page97.29→84.76%. 32 tests pass, adapter save/fresh-object reload and single-image inference verified. Independent review corrected reporting-only generation-limit padding flag and added final dataset/split hash guards. Exact executed training source archived. Selection-time adapter hash added for future runs; this run first binds it at final pre-test lock and says so. No resume implementation claimed. Evidence: docs/OCR_V2_VL_LORA_RESULTS.md, docs/ocr_vl_lora/, data/experiments/ocr_v2_vl_lora/.
