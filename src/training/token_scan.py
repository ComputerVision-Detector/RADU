"""Measure every prepared sample once, across CPU workers, so training need not repeat it.

Each row goes through the training collator, so the image hash, the decode, the prompt mask and
the token count are the ones training will use. Rows over the budget or failing are left out of
training by id and counted in the report; answers are never truncated.
"""
from __future__ import annotations

import argparse
from collections import Counter
import heapq
import json
from multiprocessing import get_context
import os
from pathlib import Path

from .prepare import read_jsonl, sha256
from .schema import require
from .train import StructureCollator, scan_report_path

state = {}


def start_worker(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    from transformers import AutoProcessor

    torch.set_num_threads(1)
    processor = AutoProcessor.from_pretrained(config["model_id"], revision=config["model_revision"], max_pixels=config["max_pixels"])
    processor.tokenizer.padding_side = "right"
    state["collator"] = StructureCollator(processor, 10**9)


def measure(row):
    try:
        require(sha256(row["image"]) == row["image_sha256"], "image changed since preparation")
        require(isinstance(json.loads(row["completion"]), dict), "invalid completion")
        batch = state["collator"]([row])
        return row["id"], row["dataset"], int(batch["attention_mask"].sum()), None
    except Exception as error:  # Reported by id; one bad file must not hide the others.
        return row["id"], row["dataset"], 0, f"{type(error).__name__}: {error}"[:300]


def scan(config, workers):
    train_path = Path(config["train_file"])
    manifest = json.loads((train_path.parent / "manifest.json").read_text(encoding="utf-8"))
    budget = config["max_sequence_tokens"]
    report = {"split_sha256": manifest["split_sha256"], "model_revision": config["model_revision"], "max_pixels": config["max_pixels"], "budget": budget,
              "excluded": {"over_budget": [], "failed": []}, "failures": [], "per_dataset": {}, "max_tokens": 0}
    longest, counts, sums, peaks = [], Counter(), Counter(), Counter()
    with get_context("spawn").Pool(workers, start_worker, (config,)) as pool:
        for split in ("train", "validation", "test"):
            rows = read_jsonl(train_path.parent / f"{split}.jsonl")
            for done, (row_id, dataset, tokens, error) in enumerate(pool.imap_unordered(measure, rows, chunksize=64), 1):
                if error:
                    report["excluded"]["failed"].append(row_id)
                    if len(report["failures"]) < 50:
                        report["failures"].append({"id": row_id, "error": error})
                elif tokens > budget:
                    report["excluded"]["over_budget"].append(row_id)
                else:
                    report["max_tokens"] = max(report["max_tokens"], tokens)
                    entry = (tokens, row_id, split)
                    heapq.heappush(longest, entry) if len(longest) < 64 else heapq.heappushpop(longest, entry)
                    counts[dataset] += 1
                    sums[dataset] += tokens
                    peaks[dataset] = max(peaks[dataset], tokens)
                if done % 20000 == 0:
                    print(json.dumps({"split": split, "done": done}), flush=True)
    report["longest"] = sorted(longest, reverse=True)
    report["per_dataset"] = {name: {"samples": counts[name], "mean_tokens": round(sums[name] / counts[name]), "max_tokens": peaks[name]} for name in sorted(counts)}
    scan_report_path(config).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    report = scan(config, args.workers)
    print(json.dumps({key: report[key] for key in ("max_tokens", "per_dataset", "failures")}, ensure_ascii=False))
    print(json.dumps({key: len(value) for key, value in report["excluded"].items()}))


if __name__ == "__main__":
    main()
