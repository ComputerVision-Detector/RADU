"""Validate six normalized sources and create reproducible image-only SFT splits."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .schema import PROMPT, TASK_INSTRUCTIONS, require, validate_record


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_prompt(record):
    # Literal JSON braces in the instruction must not be interpreted as str.format fields.
    # Diagram-only prompts stay byte-identical; other tasks add their format before the key list.
    extra = "".join(TASK_INSTRUCTIONS[task] for task in sorted(record["tasks"]) if task in TASK_INSTRUCTIONS)
    return (PROMPT.replace("Requested keys:", extra + "Requested keys:").replace("{tasks}", ", ".join(sorted(record["tasks"])))
            .replace("{width}", str(record["width"]))
            .replace("{height}", str(record["height"])))


def prepare(registry_path, output_dir, allow_partial=False):
    from PIL import Image, ImageOps

    registry_path = Path(registry_path).resolve()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    root = registry_path.parent.parent
    output_dir = Path(output_dir)
    require(not output_dir.exists(), "output directory exists; use a new run directory")
    splits = {key: [] for key in ("train", "validation", "test")}
    ids, hashes, groups = set(), {}, {}
    report = {"schema_version": 1, "registry_sha256": sha256(registry_path), "sources": {}, "group_policy": "All explicitly supplied global group keys and exact image hashes must stay in one split. Official template overlap may require a new stricter split."}
    for source in registry["datasets"]:
        path = root / source["records"]
        info = {"status": source["status"], "license": source["license"], "source": source["source"], "records": 0, "splits": Counter(), "tasks": Counter()}
        report["sources"][source["id"]] = info
        if not path.is_file():
            require(allow_partial, f"missing source {source['id']}: {path}; --allow-partial is for a pilot only")
            info["availability"] = "missing"
            continue
        info["availability"] = "normalized_file_present"
        info["normalized_sha256"] = sha256(path)
        for record in read_jsonl(path):
            validate_record(record)
            require(record["dataset"] == source["id"], "record dataset differs from registry")
            key = (record["dataset"], record["id"])
            require(key not in ids, f"duplicate sample {key}")
            ids.add(key)
            image_path = (path.parent / record["image"]).resolve()
            require(image_path.is_file(), f"missing image: {image_path}")
            with Image.open(image_path) as image:
                require(ImageOps.exif_transpose(image).size == (record["width"], record["height"]), f"image dimensions disagree: {image_path}")
                require(image.getexif().get(274, 1) == 1, "normalize EXIF orientation and transform labels before preparing")
            image_hash = sha256(image_path)
            split = record["split"]
            require(image_hash not in hashes or hashes[image_hash] == split, f"image leaked across splits: {key}")
            # Exact duplicates in one split also silently overweight a mirrored source.
            require(image_hash not in hashes, f"duplicate image content: {key}; deduplicate source mirrors")
            hashes[image_hash] = split
            for group in record["groups"]:
                require(group not in groups or groups[group] == split, f"group leaked across splits: {group}")
                groups[group] = split
            target = {name: record["target"][name] for name in sorted(record["tasks"])}
            example = {"id": f"{record['dataset']}:{record['id']}", "dataset": record["dataset"],
                       "image": str(image_path), "image_sha256": image_hash, "split": split,
                       "groups": record["groups"], "source_revision": record["source_revision"],
                       "annotation_origin": record["annotation_origin"],
                       "prompt": build_prompt(record), "completion": json.dumps(target, ensure_ascii=False, separators=(",", ":"))}
            splits[split].append(example)
            info["records"] += 1
            info["splits"][split] += 1
            info["tasks"].update(record["tasks"])
        require(info["records"] > 0, f"empty normalized source: {path}")
    require(splits["train"] and splits["validation"] and splits["test"], "nonempty train, validation, test required")
    report["counts"] = {name:len(rows) for name,rows in splits.items()}
    report["partial"] = any(info["records"] == 0 for info in report["sources"].values())
    output_dir.mkdir(parents=True)
    for name, rows in splits.items():
        rows.sort(key=lambda row: row["id"])
        (output_dir / f"{name}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    report["split_sha256"] = {name:sha256(output_dir / f"{name}.jsonl") for name in splits}
    (output_dir / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="configs/radu_datasets.json")
    parser.add_argument("--output-dir", default="data/processed/radu")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.registry, args.output_dir, args.allow_partial), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
