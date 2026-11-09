"""Single-GPU Qwen3-VL language-only LoRA with exact assistant-prefix masking."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random

from .prepare import read_jsonl, sha256
from .schema import require


def language_lora_targets(names):
    """Explicit full names avoid accidentally adapting vision attention projections."""
    endings = {"q_proj", "k_proj", "v_proj", "o_proj"}
    selected = [name for name in names if ".language_model." in "." + name and name.rsplit(".", 1)[-1] in endings]
    require(bool(selected), "No Qwen3 language attention modules found; inspect model architecture")
    require(all("visual" not in name for name in selected), "vision LoRA target detected")
    return selected


def mask_labels(full_ids, prefix_ids, attention):
    """Pure Python core, tested without downloading a model. Keep EOS trainable."""
    require(len(full_ids) == len(attention), "attention length mismatch")
    require(full_ids[:len(prefix_ids)] == prefix_ids, "chat template prefix mismatch; refusing incorrect loss mask")
    require(len(prefix_ids) < sum(attention), "no completion tokens to supervise")
    return [token if index >= len(prefix_ids) and attention[index] else -100 for index, token in enumerate(full_ids)]


def training_schedule(train_rows, config, max_steps=None):
    """Optimizer steps and warmup; refuse a run that would never evaluate or checkpoint."""
    rows = config.get("samples_per_epoch") or train_rows
    per_epoch = math.ceil(rows / (config["batch_size"] * config["gradient_accumulation_steps"]))
    total = max_steps if max_steps is not None else per_epoch * config["epochs"]
    if max_steps is None:
        for key in ("eval_steps", "save_steps"):
            require(config[key] <= total, f"{key}={config[key]} exceeds {total} optimizer steps; it would never run")
    # Floor: warmup starts at lr 0, so a very short run with warmup would never update weights.
    return total, math.floor(0.03 * total)


def load_base_model(config):
    """Load processor and base model; 4-bit only when the config asks for it."""
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    revision = config.get("model_revision", "main")
    processor = AutoProcessor.from_pretrained(config["model_id"], revision=revision, max_pixels=config["max_pixels"])
    load_options = {}
    if config.get("load_in_4bit", False):
        load_options = {"quantization_config":BitsAndBytesConfig(load_in_4bit=True,
                        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        llm_int8_skip_modules=config.get("quantization_skip_modules")), "device_map":{"":0}}
    model = Qwen3VLForConditionalGeneration.from_pretrained(config["model_id"], revision=revision,
                torch_dtype=torch.bfloat16, attn_implementation=config["attn_implementation"], **load_options)
    return processor, model


def source_balanced_weights(rows, shares=None):
    """Per-row sampling weights: each dataset gets its share of training, equal when none is given."""
    counts = Counter(row["dataset"] for row in rows)
    if shares:
        missing = set(counts) - set(shares)
        require(not missing, f"sampling_shares lacks {sorted(missing)}; a source would silently never train")
    return [(shares[row["dataset"]] if shares else 1.0) / counts[row["dataset"]] for row in rows]


def quota_indices(rows, shares, count, seed=42):
    """Pick a source by its share, then its next unseen row; a source reshuffles only when used up."""
    by_source = {}
    for index, row in enumerate(rows):
        by_source.setdefault(row["dataset"], []).append(index)
    missing = set(by_source) - set(shares)
    require(not missing, f"sampling_shares lacks {sorted(missing)}; a source would silently never train")
    names = sorted(by_source)
    weights = [shares[name] for name in names]
    rng = random.Random(f"{seed}:quota")
    queues = {name: [] for name in names}
    for _ in range(count):
        name = rng.choices(names, weights)[0]
        if not queues[name]:
            queues[name] = by_source[name][:]
            rng.shuffle(queues[name])
        yield queues[name].pop()


class QuotaSampler:
    def __init__(self, rows, shares, count, seed):
        self.rows, self.shares, self.count, self.seed = rows, shares, count, seed

    def __iter__(self):
        return quota_indices(self.rows, self.shares, self.count, self.seed)

    def __len__(self):
        return self.count


def scan_report_path(config):
    """Bound to the settings that change token counts, so a stale scan cannot be reused."""
    return Path(config["train_file"]).parent / f"token_scan_{config['max_pixels']}_{config['max_sequence_tokens']}.json"


def load_scanned(path, excluded):
    """token_scan already hashed, decoded and measured these rows; only leave out what it rejected."""
    rows = [row for row in read_jsonl(path) if row["id"] not in excluded]
    require(bool(rows), f"empty prepared split: {path}")
    prompts = {}  # Prompts repeat by image size; sharing them saves gigabytes on a large pool.
    for row in rows:
        row["prompt"] = prompts.setdefault(row["prompt"], row["prompt"])
    return rows


def per_dataset_subsets(rows, count, seed=42):
    """A fixed sample per source, so each checkpoint shows a loss for every source."""
    by_dataset = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], []).append(row)
    return {name: random.Random(f"{seed}:{name}").sample(items, min(count, len(items))) for name, items in sorted(by_dataset.items())}


def evaluation_subset(rows, limit, seed=42):
    """Seeded and proportional per dataset, so loss checks stay cheap on large validation sets."""
    if not limit or limit >= len(rows):
        return rows
    by_dataset = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], []).append(row)
    picked = []
    for name, items in sorted(by_dataset.items()):
        count = max(1, round(limit * len(items) / len(rows)))
        picked.extend(random.Random(f"{seed}:{name}").sample(items, min(count, len(items))))
    return picked


class StructureCollator:
    def __init__(self, processor, max_tokens):
        self.processor = processor
        self.max_tokens = max_tokens

    def __call__(self, rows):
        import torch
        from PIL import Image

        images, full_texts, prefixes = [], [], []
        for row in rows:
            with Image.open(row["image"]) as image:
                images.append(image.convert("RGB"))
            messages = [{"role":"user", "content":[{"type":"image"}, {"type":"text", "text":row["prompt"]}]}]
            prefixes.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            full_texts.append(self.processor.apply_chat_template(messages + [{"role":"assistant", "content":[{"type":"text", "text":row["completion"]}]}], tokenize=False, add_generation_prompt=False))
        batch = self.processor(text=full_texts, images=images, padding=True, truncation=False, return_tensors="pt")
        labels = batch["input_ids"].clone()
        for i, (prefix, image) in enumerate(zip(prefixes, images)):
            # Processing the same image expands identical image placeholders in the prefix.
            prompt = self.processor(text=[prefix], images=[image], padding=False, truncation=False, return_tensors="pt")
            active = batch["attention_mask"][i].bool()
            ids = batch["input_ids"][i][active].tolist()
            require(len(ids) <= self.max_tokens, f"{rows[i]['id']}: {len(ids)} tokens exceeds {self.max_tokens}; resize/tile or raise budget, never truncate image/JSON tokens")
            prefix_ids = prompt["input_ids"][0].tolist()
            masked = mask_labels(ids, prefix_ids, [1] * len(ids))
            labels[i].fill_(-100)
            labels[i, active] = torch.tensor(masked, dtype=labels.dtype)
        batch["labels"] = labels
        return batch


def load_prepared(path):
    rows = list(read_jsonl(path))
    require(bool(rows), f"empty prepared split: {path}")
    for row in rows:
        require(Path(row["image"]).is_file(), f"missing prepared image {row['image']}")
        require(sha256(row["image"]) == row["image_sha256"], f"image changed since preparation: {row['id']}")
        require(isinstance(row.get("completion"), str) and isinstance(json.loads(row["completion"]), dict), "invalid completion")
    return rows


def run(config_path, preflight_only=False, max_steps=None, resume=None, output_dir=None):
    import platform
    import subprocess
    import sys
    import torch
    import transformers
    import peft
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import Trainer, TrainingArguments, set_seed

    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    require(torch.cuda.is_available(), "CUDA GPU required; prepare/import/tests can run on CPU")
    require(torch.cuda.is_bf16_supported(), "This BF16 profile requires a BF16-capable GPU")
    set_seed(config["seed"])
    train_path, val_path = Path(config["train_file"]), Path(config["validation_file"])
    require(train_path.parent == val_path.parent, "train/validation must come from one preparation run")
    manifest_path = train_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for split, path in (("train", train_path), ("validation", val_path)):
        require(manifest["split_sha256"][split] == sha256(path), f"prepared {split} changed; prepare again")
    scanned = config.get("require_token_scan", False)
    if scanned:
        report_path = scan_report_path(config)
        require(report_path.is_file(), f"{report_path} missing; run python -m src.training.token_scan --config {config_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        require(report["split_sha256"] == manifest["split_sha256"] and report["model_revision"] == config["model_revision"], "token scan is stale; scan again")
        excluded = {row_id for rows in report["excluded"].values() for row_id in rows}
        train, validation = load_scanned(train_path, excluded), load_scanned(val_path, excluded)
        print(json.dumps({"token_scan": {"max_tokens": report["max_tokens"], "excluded": {key: len(rows) for key, rows in report["excluded"].items()}}}))
    else:
        train = load_prepared(train_path)
        validation = load_prepared(val_path)
    total_steps, warmup_steps = training_schedule(len(train), config, max_steps)
    print(json.dumps({"plan": {"train_pool": len(train), "validation_pool": len(validation), "optimizer_steps": total_steps, "warmup_steps": warmup_steps, "learning_rate": config["learning_rate"]}}), flush=True)
    require(all(row["split"] == "train" for row in train) and all(row["split"] == "validation" for row in validation), "wrong split passed to training")
    require(not ({row["image_sha256"] for row in train} & {row["image_sha256"] for row in validation}), "image leakage")
    require(not ({g for row in train for g in row["groups"]} & {g for row in validation for g in row["groups"]}), "group leakage")
    output = Path(output_dir or config["output_dir"])
    if not preflight_only:
        require(resume is not None or not output.exists() or not any(output.iterdir()), "nonempty output; choose a new directory or --resume checkpoint")
    processor, model = load_base_model(config)
    processor.tokenizer.padding_side = "right"
    if config.get("load_in_4bit", False):
        if "model.visual" in config.get("quantization_skip_modules", []):
            quantized = [name for name, module in model.named_modules() if "visual" in name and type(module).__name__ == "Linear4bit"]
            require(not quantized, f"vision encoder was quantized despite skip list: {quantized[:3]}")
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                               gradient_checkpointing_kwargs={"use_reentrant":False})
        # peft upcasts frozen unquantized modules (vision encoder, lm_head) to fp32. Measured
        # on 8 GB: casting back saves about 2.5 GiB at peak with matching loss.
        for parameter in model.parameters():
            if parameter.dtype == torch.float32:
                parameter.data = parameter.data.to(torch.bfloat16)
    # The top-level flag does not reach the language model; without this, evaluation
    # keeps past_key_values for every token.
    model.config.use_cache = False
    model.config.get_text_config().use_cache = False
    targets = language_lora_targets(dict(model.named_modules()))
    model = get_peft_model(model, LoraConfig(r=config["lora_r"], lora_alpha=config["lora_alpha"],
                lora_dropout=config["lora_dropout"], target_modules=targets, bias="none", task_type="CAUSAL_LM"))
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    require(trainable and all("lora_" in name and "language_model" in name and "visual" not in name for name in trainable), "unexpected trainable parameters")
    model.print_trainable_parameters()
    collator = StructureCollator(processor, config["max_sequence_tokens"])
    # Scan every example before starting a costly training run; never truncate.
    if not scanned:
        for row in train + validation:
            collator([row])
    if preflight_only:
        print(json.dumps({"preflight":"passed", "train":len(train), "validation":len(validation), "lora_modules":len(targets), "partial":manifest["partial"]}))
        return
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"config":config, "model_commit":getattr(model.config, "_commit_hash", None),
                "python":platform.python_version(), "torch":torch.__version__, "transformers":transformers.__version__,
                "peft":peft.__version__, "train_sha256":sha256(train_path), "validation_sha256":sha256(val_path),
                "manifest_sha256":sha256(manifest_path), "partial":manifest["partial"], "lora_targets":targets,
                "trainable_parameters":trainable, "gpu":torch.cuda.get_device_name(0)}
    (output / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "environment.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    args = TrainingArguments(output_dir=str(output), num_train_epochs=config["epochs"], learning_rate=config["learning_rate"],
                per_device_train_batch_size=config["batch_size"], per_device_eval_batch_size=1,
                gradient_accumulation_steps=config["gradient_accumulation_steps"], bf16=True,
                gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant":False},
                remove_unused_columns=False, label_names=["labels"], report_to="none", seed=config["seed"],
                eval_strategy="steps", eval_steps=config["eval_steps"], save_steps=config["save_steps"],
                save_total_limit=config.get("save_total_limit", 2), logging_steps=config["logging_steps"], logging_first_step=True, warmup_steps=warmup_steps,
                # Workers decode and resize the next images while the GPU trains on the current one.
                dataloader_num_workers=config.get("dataloader_num_workers", 0),
                dataloader_persistent_workers=config.get("dataloader_num_workers", 0) > 0,
                max_steps=max_steps if max_steps is not None else -1)
    trainer_class = Trainer
    if config.get("balance_sources", False):
        class SourceBalancedTrainer(Trainer):
            def _get_train_sampler(self, train_dataset=None):
                if config.get("sample_without_repeats"):
                    return QuotaSampler(train, config["sampling_shares"], config["samples_per_epoch"], config["seed"])
                generator = torch.Generator().manual_seed(config["seed"])
                weights = source_balanced_weights(train, config.get("sampling_shares"))
                return torch.utils.data.WeightedRandomSampler(weights, config.get("samples_per_epoch") or len(train), generator=generator)
        trainer_class = SourceBalancedTrainer
    # Every sample was already scanned above; this only limits how much each loss check costs.
    if config.get("eval_samples_per_dataset"):
        eval_rows = per_dataset_subsets(validation, config["eval_samples_per_dataset"], config["seed"])
    else:
        eval_rows = evaluation_subset(validation, config.get("max_eval_samples"), config["seed"])
    trainer = trainer_class(model=model, args=args, train_dataset=train, eval_dataset=eval_rows,
                            data_collator=collator, processing_class=processor)
    trainer.train(resume_from_checkpoint=resume)
    # lora_B starts at exactly zero; all-zero after training means nothing was learned.
    require(any(float(p.abs().sum()) > 0 for name, p in model.named_parameters() if "lora_B" in name), "LoRA weights unchanged after training: no update reached the adapter")
    # Periodic evaluation can stop short of the last step; always score the saved weights.
    print(json.dumps({"final_eval":trainer.evaluate()}))
    print(json.dumps({"peak_allocated_mib":torch.cuda.max_memory_allocated() // 2**20, "peak_reserved_mib":torch.cuda.max_memory_reserved() // 2**20}))
    trainer.save_model(str(output / "final"))
    processor.save_pretrained(output / "final")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qwen3_vl_4b_lora.json")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--output-dir", help="Override the config output_dir, for example to keep a smoke test apart")
    args = parser.parse_args()
    if args.max_steps is not None:
        require(args.max_steps > 0, "max-steps must be positive")
    # Long and short answers alternate; without this, fragmentation caused OOM at 6,016 tokens with 3.4 GiB idle.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    run(args.config, args.preflight_only, args.max_steps, args.resume, args.output_dir)


if __name__ == "__main__":
    main()
