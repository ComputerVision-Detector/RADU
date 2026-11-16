# Clean train and visual adaptation experiment

User approved train quality audit followed by actual language-vs-vision LoRA experiments on the same cleaned data. Preserve V2 source, original model, all previous runs, existing GPU jobs and operational code. Existing validation244/quarantine53 from round2 is fixed; no new validation/test-driven training exclusions. Final test235 remains exploratory.

- [x] Audit every train2513 crop structurally and by hash; generate risk queues, per-page representatives and original-pixel review panels.
- [x] Visually inspect train-only risk/coverage samples; record confirmed/uncertain crop defects in a derived hold manifest, never replace transcription with OCR. Freeze cleaned list and audit scope/limits.
- [x] Test language/vision module targeting, simultaneous adapter activation, frozen language weights and zero-initialized vision identity, train-only cleaning and artifact binding.
- [x] Same parent LoRA initialization, clean train and schedule: language-only continued LoRA versus frozen parent language + new rank8 q/v LoRA on last4 visual blocks. Projector remains frozen to isolate visual adaptation.
- [x] Real smoke on train/validation: finite loss/gradient, changed intended parameters only, adapter save/reload inference equality, VRAM.
- [x] Complete both bounded full-train runs, choose among parent/language/vision solely by fixed validation244 CER. Preserve full297 diagnostics.
- [x] Lock selected bundle and hashes before unchanged exploratory test235 and fixed detector397/199GT page inference.
- [x] Run tests, save real checkpoints, before/after predictions, failure cases, paired intervals, exact commands and report honest outcome.

Bounded 3epochs each, seed42, AdamW lr2e-5, effective batch8, existing112896/200704 pixel budget, same OCR prompt/greedy/NFC. Start batch2 accumulation4; if vision smoke needs batch1/accum8 apply that to BOTH full runs before they start. No hyperparameter sweep or post-test retuning. Clean-data comparison changes data and possibly number of steps relative to prior rounds; do not claim isolated causal data-cleaning gain without a matched control.

Actual outcome: both full runs completed 879 steps each; no validation CER improvement, parent retained. All41 tests passed. Review added effective-config drift binding; original effective config verified unchanged. Single-crop vision inference reloaded both adapters successfully.
