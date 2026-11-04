"""Audit on-disk public source assets against saved plans; reject LFS pointers."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from .acquire import checksums, matches, safe_path


def audit(root, plan_paths, output):
    from PIL import Image
    root = Path(root)
    items = {}
    for plan in plan_paths:
        for line in Path(plan).read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                items[item["path"]] = item
    sources = {}
    rows = []
    for index, item in enumerate(items.values(), 1):
        stats = sources.setdefault(item["dataset"], {"expected_files":0, "present_files":0, "verified_files":0, "verified_image_files":0, "bytes":0, "problems":[]})
        stats["expected_files"] += 1
        path = safe_path(root, item["path"])
        row = {"dataset":item["dataset"], "path":item["path"], "source_url":item["url"]}
        if not path.is_file():
            row["status"] = "missing"
        else:
            stats["present_files"] += 1
            with path.open("rb") as handle:
                pointer = handle.read(80).startswith(b"version https://git-lfs.github.com/spec/")
            if pointer:
                row["status"] = "git_lfs_pointer_not_data"
            else:
                actual = checksums(path)
                row.update(actual)
                assertions = [key for key in ("size","md5","sha256","git_blob_sha1") if item.get(key) is not None]
                row["upstream_assertions"] = assertions
                row["status"] = "verified" if matches(item, actual) else "upstream_integrity_mismatch"
                if row["status"] == "verified":
                    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                        try:
                            with Image.open(path) as image:
                                row["image_size"] = list(image.size)
                                image.verify()
                            stats["verified_image_files"] += 1
                        except Exception as exc:
                            row.update(status="image_decode_failure", error=str(exc))
                    if row["status"] == "verified":
                        stats["verified_files"] += 1
                        stats["bytes"] += actual["size"]
                        if not assertions:
                            row["verification_scope"] = "local_hash_only_no_upstream_digest"
        if row["status"] != "verified":
            stats["problems"].append({"path":item["path"], "status":row["status"]})
        rows.append(row)
        if index % 5000 == 0:
            print(f"AUDIT {index}/{len(items)}", flush=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "verified_assets.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    summary = {"created_at":datetime.now(timezone.utc).isoformat(), "sources":sources,
               "training_ready":False, "note":"Source asset audit only. Complete supervision and split checks are separate gates."}
    (output / "acquisition_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key:{k:v for k,v in value.items() if k != "problems"} for key,value in sources.items()}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plans", nargs="+")
    parser.add_argument("--root", default="data/raw/radu")
    parser.add_argument("--output", default="data/processed/radu_staging")
    args = parser.parse_args()
    audit(args.root, args.plans, args.output)


if __name__ == "__main__":
    main()
