"""Resumable public dataset acquisition with pinned source metadata and checksums.

Plan JSONL: {dataset, path, url, size?, git_blob_sha1?, sha256?, md5?}.
All output paths are relative to --root. Existing files are rehashed, not trusted.
"""
from __future__ import annotations

import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time
import urllib.request
import zipfile


def safe_path(root, name):
    if not isinstance(name, str) or "\\" in name or ":" in name:
        raise ValueError(f"unsafe archive/plan path: {name!r}")
    rel = PurePosixPath(name)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"unsafe archive/plan path: {name!r}")
    root = Path(root).resolve()
    path = root.joinpath(*rel.parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"path escapes destination: {name!r}")
    return path


def checksums(path):
    size = path.stat().st_size
    sha = hashlib.sha256()
    md5 = hashlib.md5()
    git = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
            md5.update(block)
            git.update(block)
    return {"size":size, "sha256":sha.hexdigest(), "md5":md5.hexdigest(), "git_blob_sha1":git.hexdigest()}


def matches(item, actual):
    return all(item[key] == actual[key] for key in ("size", "sha256", "md5", "git_blob_sha1") if item.get(key) is not None)


def fetch(item, root):
    destination = safe_path(root, item["path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not any(item.get(key) is not None for key in ("size", "sha256", "md5", "git_blob_sha1")):
            raise ValueError(f"existing file has no upstream/prior checksum: {destination}")
        actual = checksums(destination)
        if matches(item, actual):
            return {**item, **actual, "status":"verified_existing"}
        raise ValueError(f"existing file does not match source: {destination}; inspect before replacing")
    part = destination.with_name(destination.name + ".part")
    for attempt in range(4):
        try:
            req = urllib.request.Request(item["url"], headers={"User-Agent":"RADU-dataset-preparation/1.0"})
            with urllib.request.urlopen(req, timeout=60) as response, part.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=1024 * 1024)
            actual = checksums(part)
            if not matches(item, actual):
                raise ValueError(f"checksum/size mismatch: {item['path']}")
            part.replace(destination)
            return {**item, **actual, "status":"downloaded", "downloaded_at":datetime.now(timezone.utc).isoformat()}
        except Exception:
            if attempt == 3:
                raise
            time.sleep(min(2 ** attempt, 8))


def fetch_ranges(item, root, workers=8, chunk_size=4 * 1024 * 1024):
    """Resume public large objects by validated HTTP ranges, then verify whole object."""
    destination = safe_path(root, item["path"])
    if destination.exists():
        return fetch(item, root)
    size = item["size"]
    identity = hashlib.sha256((item["url"] + str(size)).encode()).hexdigest()[:16]
    chunks = destination.parent / (destination.name + ".chunks-" + identity)
    chunks.mkdir(parents=True, exist_ok=True)
    def chunk(start):
        end = min(size, start + chunk_size) - 1
        out = chunks / str(start)
        if out.exists() and out.stat().st_size == end - start + 1:
            return out
        for attempt in range(4):
            try:
                request = urllib.request.Request(item["url"], headers={"User-Agent":"RADU/1.0", "Range":f"bytes={start}-{end}"})
                with urllib.request.urlopen(request, timeout=40) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                        raise ValueError("server did not honor exact requested byte range")
                    data = response.read(end - start + 2)
                if len(data) != end - start + 1:
                    raise ValueError("incomplete range")
                with out.open("wb") as handle:
                    handle.write(data)
                return out
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
    starts = list(range(0, size, chunk_size))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for number, _ in enumerate(pool.map(chunk, starts), 1):
            if number % 40 == 0:
                print(f"RANGES {item['path']}: {number}/{len(starts)}", flush=True)
    part = destination.with_name(destination.name + ".part")
    with part.open("wb") as output:
        for start in starts:
            with (chunks / str(start)).open("rb") as source:
                shutil.copyfileobj(source, output)
    actual = checksums(part)
    if not matches(item, actual):
        raise ValueError(f"assembled object failed integrity: {item['path']}")
    part.replace(destination)
    # Keep range files until final verification, then remove only this owned cache.
    for start in starts:
        (chunks / str(start)).unlink()
    chunks.rmdir()
    return {**item, **actual, "status":"downloaded_ranges", "downloaded_at":datetime.now(timezone.utc).isoformat()}


def download(plan_path, root, workers=12, range_workers=0):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    plan = [json.loads(line) for line in Path(plan_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    receipt_path = root / "acquisition_receipts.jsonl"
    previous = {}
    if receipt_path.exists():
        for line in receipt_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("sha256") and row.get("status") != "failed":
                previous[row["path"]] = row
    for item in plan:
        if not any(item.get(key) is not None for key in ("size", "sha256", "md5", "git_blob_sha1")):
            old = previous.get(item["path"], {})
            if old.get("url") == item["url"]:
                item["sha256"] = old["sha256"]
    paths = [safe_path(root, item["path"]) for item in plan]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate destination paths in download plan")
    # Resolve/create directories before workers start. On Windows, concurrent
    # creation while resolve() queries a nonexistent path can yield transient
    # alternate path spellings and fail the containment check.
    for parent in sorted({path.parent for path in paths}, key=str):
        parent.mkdir(parents=True, exist_ok=True)
    failures = []
    total_bytes = 0
    with (root / "acquisition_receipts.jsonl").open("a", encoding="utf-8") as log:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(fetch_ranges, item, root, range_workers) if range_workers and item.get("size", 0) >= 20 * 1024**2
                       else pool.submit(fetch, item, root):item for item in plan}
            for number, future in enumerate(concurrent.futures.as_completed(pending), 1):
                item = pending[future]
                try:
                    result = future.result()
                    total_bytes += result["size"]
                except Exception as exc:
                    result = {**item, "status":"failed", "error":str(exc)}
                    failures.append(result)
                    print(f"FAILED {item['path']}: {exc}", flush=True)
                log.write(json.dumps(result, ensure_ascii=False) + "\n")
                log.flush()
                if number % 100 == 0 or number == len(plan):
                    print(f"{number}/{len(plan)} files; verified {total_bytes / 1024**2:.1f} MiB; failures={len(failures)}", flush=True)
    return failures


def extract(archive, destination):
    """No symlinks, traversal, overwrite, or extraction outside requested directory."""
    archive, destination = Path(archive), Path(destination)
    if destination.exists():
        raise ValueError("extraction destination exists; inspect it or choose a new directory")
    destination.mkdir(parents=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            entries = zf.infolist()
            if any(info.flag_bits & 1 for info in entries):
                raise ValueError("encrypted archive: obtain password through dataset provider; no extraction attempted")
            for info in entries:
                safe_path(destination, info.filename)
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("archive contains symlink")
            for info in entries:
                target = safe_path(destination, info.filename)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info) as source, target.open("xb") as out:
                        shutil.copyfileobj(source, out)
    else:
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                target = safe_path(destination, member.name)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tf.extractfile(member) as source, target.open("xb") as out:
                        shutil.copyfileobj(source, out)
                else:
                    raise ValueError(f"unsupported archive member: {member.name}")


def materialize(plan_path, checkout, root, prefix, lfs_objects=None):
    """Copy selected Git LFS materialized assets after verifying upstream hashes."""
    root = Path(root)
    plan = [json.loads(line) for line in Path(plan_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    failures = []
    with (root / "acquisition_receipts.jsonl").open("a", encoding="utf-8") as log:
        for number, item in enumerate(plan, 1):
            try:
                if not item["path"].startswith(prefix + "/"):
                    raise ValueError("materialization prefix mismatch")
                destination = safe_path(root, item["path"])
                if destination.exists():
                    result = fetch(item, root)
                else:
                    source = safe_path(checkout, item["path"][len(prefix) + 1:])
                    with source.open("rb") as handle:
                        if handle.read(80).startswith(b"version https://git-lfs.github.com/spec/"):
                            digest = item.get("sha256", "")
                            if not lfs_objects or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                                raise ValueError("Git LFS pointer is not an image; run git lfs pull")
                            # git lfs fetch can finish before checkout has replaced
                            # every pointer. Only use a SHA-pinned actual CAS object.
                            source = safe_path(lfs_objects, f"{digest[:2]}/{digest[2:4]}/{digest}")
                    actual = checksums(source)
                    if not matches(item, actual):
                        raise ValueError("Git asset failed upstream size/hash check")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    part = destination.with_name(destination.name + ".part")
                    shutil.copyfile(source, part)
                    if not matches({**item, "sha256":actual["sha256"]}, checksums(part)):
                        raise ValueError("copy failed integrity check")
                    part.replace(destination)
                    result = {**item, **actual, "status":"verified_git_lfs_asset", "downloaded_at":datetime.now(timezone.utc).isoformat()}
            except Exception as exc:
                result = {**item, "status":"failed", "error":str(exc)}
                failures.append(result)
            log.write(json.dumps(result, ensure_ascii=False) + "\n")
            if number % 1000 == 0 or number == len(plan):
                log.flush()
                print(f"MATERIALIZE {number}/{len(plan)}; failures={len(failures)}", flush=True)
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    dl = sub.add_parser("download")
    dl.add_argument("plan")
    dl.add_argument("--root", default="data/raw/radu")
    dl.add_argument("--workers", type=int, default=12)
    dl.add_argument("--range-workers", type=int, default=0)
    ex = sub.add_parser("extract")
    ex.add_argument("archive")
    ex.add_argument("destination")
    mat = sub.add_parser("materialize")
    mat.add_argument("plan")
    mat.add_argument("checkout")
    mat.add_argument("--root", default="data/raw/radu")
    mat.add_argument("--prefix", default="flowlearn/source")
    mat.add_argument("--lfs-objects")
    args = parser.parse_args()
    if args.command == "extract":
        extract(args.archive, args.destination)
    elif args.command == "materialize":
        raise SystemExit(bool(materialize(args.plan, args.checkout, args.root, args.prefix, args.lfs_objects)))
    else:
        raise SystemExit(bool(download(args.plan, args.root, args.workers, args.range_workers)))


if __name__ == "__main__":
    main()
