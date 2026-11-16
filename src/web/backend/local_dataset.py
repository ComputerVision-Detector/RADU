"""Browse the prepared RADU datasets in the project folder so their images can be analysed in the web.

Lists the validation and test splits of ``data/processed/radu_final`` (the 16 diagram, formula
and table sources of the Qwen LoRA run). The 1.87M-row train split is not listed.
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PREPARED = ROOT / "data/processed/radu_final"
SPLITS = ("validation", "test")
KINDS = {"nodes": "diagram", "arrows": "diagram", "relations": "diagram", "formulas": "formula", "tables": "table"}


@lru_cache(maxsize=1)
def _index() -> dict:
    samples, sources = {}, {}
    for split in SPLITS:
        with (PREPARED / f"{split}.jsonl").open(encoding="utf-8") as rows:
            for line in rows:
                if not line.strip():
                    continue
                row = json.loads(line)
                target = json.loads(row["completion"])
                kind = next(KINDS[key] for key in target if key in KINDS)
                samples[row["id"]] = {"id": row["id"], "source": row["dataset"], "split": split, "image": row["image"],
                                      "name": Path(row["image"]).name, "kind": kind,
                                      "labels": {key: len(value) for key, value in target.items() if isinstance(value, list)}}
                source = sources.setdefault(row["dataset"], {"id": row["dataset"], "kind": kind, "splits": {name: 0 for name in SPLITS}})
                source["splits"][split] += 1
    return {"samples": samples, "sources": sorted(sources.values(), key=lambda item: (item["kind"], item["id"]))}


def list_sources() -> dict:
    return {"root": str(PREPARED.relative_to(ROOT)), "splits": list(SPLITS), "sources": _index()["sources"]}


def list_samples(source: str, split: str) -> list[dict]:
    if split not in SPLITS or not any(item["id"] == source for item in _index()["sources"]):
        raise KeyError(source)
    return [{key: sample[key] for key in ("id", "name", "kind", "labels")}
            for sample in _index()["samples"].values() if sample["source"] == source and sample["split"] == split]


def image_path(sample_id: str) -> Path:
    """Only images named by the prepared splits, and only inside the project's data folder."""
    path = Path(_index()["samples"][sample_id]["image"]).resolve()
    if not path.is_relative_to(ROOT / "data") or not path.is_file():
        raise FileNotFoundError(sample_id)
    return path
