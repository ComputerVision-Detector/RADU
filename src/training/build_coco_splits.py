"""Build versioned leakage-aware split manifests for FC_A and hdBPMN.

FC_A: official test templates `Nb` repeat the graph topology of train template `N`
with renamed text, and 10 writers appear on both sides. v1 therefore assigns whole
writers to one split and holds some N+Nb families out of train/validation, so test
reports structure-unseen and structure-seen results separately.

hdBPMN: official partitions are already writer-disjoint and every exercise appears in
all of them. v1 keeps them and records the exercise for per-exercise evaluation.

Exact duplicate images are removed. Dropped images are listed with a reason.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re

FC_A_NAME = re.compile(r"writer(\d+)_(\d+)(b?)(_?)\.")
HDBPMN_NAME = re.compile(r"ex(\d+)_writer(\d+)\.")
CMP_NAME = re.compile(r"writer(\d+)_f[ca]_(\d+)(b?)\.png$")
PARTITION_SPLIT = {"train": "train", "val": "validation", "test": "test"}
# Templates 1 and 6 have the same labelled graph (Weisfeiler-Lehman hash over all images).
FC_A_FAMILY_ALIASES = {"1": "6"}
# Same drawing with a few changed pixels, filed under a template whose text it does not match.
FC_A_NEAR_DUPLICATES = {"writer8_1.png": "writer8_6.png"}


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def duplicate_exclusions(images):
    """Keep the first path per content hash. `images` maps file name to hash."""
    first, excluded = {}, {}
    for name in sorted(images):
        digest = images[name]
        if digest in first:
            excluded[name] = f"exact_duplicate_of:{first[digest]}"
        else:
            first[digest] = name
    return excluded


def fill_writers(counts, targets, seed):
    """Assign shuffled whole writers to held splits until each image target is reached."""
    order = sorted(counts, key=int)
    random.Random(seed).shuffle(order)
    assigned, totals = {}, Counter()
    for writer in order:
        for split, target in targets.items():
            if totals[split] < target:
                assigned[writer] = split
                totals[split] += counts[writer]
                break
        else:
            assigned[writer] = "train"
    return assigned


def build_fc_a(images, seed=42, test_share=0.20, validation_share=0.15, held_families=3):
    """`images` maps file name to content hash."""
    excluded = duplicate_exclusions(images)
    for name, original in FC_A_NEAR_DUPLICATES.items():
        if name in images and original in images:
            excluded[name] = f"near_duplicate_of:{original}"
    parsed = {}
    for name in images:
        match = FC_A_NAME.match(name)
        if not match:
            raise ValueError(f"unrecognized FC_A file name: {name}")
        parsed[name] = {"writer": match.group(1), "family": FC_A_FAMILY_ALIASES.get(match.group(2), match.group(2)),
                        "variant": match.group(2) + match.group(3)}
    kept = [name for name in parsed if name not in excluded]
    families = sorted({parsed[name]["family"] for name in kept}, key=int)
    held = set(random.Random(seed).sample(families, held_families))
    counts = Counter(parsed[name]["writer"] for name in kept)
    targets = {"test": test_share * len(kept), "validation": validation_share * len(kept)}
    writers = fill_writers(counts, targets, seed)
    samples = {}
    for name in sorted(kept):
        item = parsed[name]
        split = writers[item["writer"]]
        family_held = item["family"] in held
        if family_held and split != "test":
            # Keeping it would expose a held-out graph structure to the model or tuning.
            excluded[name] = "held_out_family_outside_test_writers"
            continue
        groups = [f"fc_a:writer:{item['writer']}"]
        if family_held:
            groups.append(f"fc_a:family:{item['family']}")
        samples[name] = {"split": split, "groups": groups, "family": item["family"], "variant": item["variant"],
                         "structure_seen_in_train": not family_held}
    return {"schema_version": 1, "dataset": "fc_a", "version": "fc_a_v1", "seed": seed,
            "policy": "Whole writers per split; templates with one graph structure (N, Nb, and aliases) share a family; held families are test-only.",
            "held_out_families": sorted(held, key=int), "samples": samples, "excluded": dict(sorted(excluded.items()))}


def build_hdbpmn(partitions, seed=42):
    """`partitions` maps file name to (official partition, content hash)."""
    excluded = duplicate_exclusions({name: value[1] for name, value in partitions.items()})
    samples = {}
    for name, (partition, _) in sorted(partitions.items()):
        match = HDBPMN_NAME.match(name)
        if not match:
            raise ValueError(f"unrecognized hdBPMN file name: {name}")
        if name in excluded:
            continue
        samples[name] = {"split": PARTITION_SPLIT[partition], "groups": [f"hdbpmn:writer:{match.group(2)}"],
                         "exercise": match.group(1), "official_partition": partition}
    return {"schema_version": 1, "dataset": "hdbpmn", "version": "hdbpmn_v1", "seed": seed,
            "policy": "Official writer-disjoint partitions kept; exercises overlap by design and are recorded for per-exercise evaluation.",
            "samples": samples, "excluded": dict(sorted(excluded.items()))}


def build_cmp(dataset, partitions, seed=42):
    """FC_B, FC_Bscan and FA: official writer-disjoint partitions, as with hdBPMN.

    FC_Bscan is FC_B's drawings in another form, so both use one writer group: a writer's
    drawings cannot appear in training through one source and in test through the other.
    FA's COCO validation file has no arrow connectivity, so those samples omit relations.
    """
    excluded = duplicate_exclusions({name: value[1] for name, value in partitions.items()})
    family = "fc_b" if dataset == "fc_b_scan" else dataset
    samples = {}
    for name, (partition, _) in sorted(partitions.items()):
        match = CMP_NAME.match(name)
        if not match:
            raise ValueError(f"unrecognized {dataset} file name: {name}")
        if name in excluded:
            continue
        samples[name] = {"split": PARTITION_SPLIT[partition], "groups": [f"{family}:writer:{match.group(1)}"],
                         "template": match.group(2), "variant": match.group(2) + match.group(3), "official_partition": partition}
        if dataset == "fa" and partition == "val":
            samples[name]["tasks"] = ["nodes", "arrows"]
    return {"schema_version": 1, "dataset": dataset, "version": f"{dataset}_v1", "seed": seed,
            "policy": "Official writer-disjoint partitions kept; templates repeat across writers by design and are recorded.",
            "samples": samples, "excluded": dict(sorted(excluded.items()))}


def verify_disjoint(manifest):
    """Every group, and every FC_A writer/family rule, must hold in the final manifest."""
    where = defaultdict(set)
    for name, sample in manifest["samples"].items():
        for group in sample["groups"]:
            where[group].add(sample["split"])
    leaked = {group: sorted(splits) for group, splits in where.items() if len(splits) > 1}
    if leaked:
        raise ValueError(f"group crosses splits: {leaked}")
    if manifest["dataset"] == "fc_a":
        for name, sample in manifest["samples"].items():
            if not sample["structure_seen_in_train"] and sample["split"] != "test":
                raise ValueError(f"held-out family outside test: {name}")
    return manifest


def summary(manifest):
    samples = manifest["samples"].values()
    result = {"splits": dict(Counter(s["split"] for s in samples)), "excluded": dict(Counter(v.split(":")[0] for v in manifest["excluded"].values()))}
    if manifest["dataset"] == "fc_a":
        result["test_structure"] = dict(Counter("seen" if s["structure_seen_in_train"] else "unseen" for s in samples if s["split"] == "test"))
    return result


def collect(root, relative, partitions):
    images = {}
    for partition in partitions:
        for path in sorted((root / relative / partition).iterdir()):
            if path.is_file():
                if path.name in images:
                    raise ValueError(f"file name repeated across partitions: {path.name}")
                images[path.name] = (partition, file_sha256(path))
    return images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw/radu")
    parser.add_argument("--output-dir", default="configs/splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", action="append", choices=["fc_a", "hdbpmn", "fc_b", "fc_b_scan", "fa"], help="Repeatable; omit to build fc_a and hdbpmn")
    args = parser.parse_args()
    root, out = Path(args.root), Path(args.output_dir)
    manifests = []
    for dataset in args.dataset or ["fc_a", "hdbpmn"]:
        if dataset == "fc_a":
            fc_a = collect(root, "fc_a/source", ("train", "test"))
            manifests.append(build_fc_a({name: value[1] for name, value in fc_a.items()}, seed=args.seed))
        elif dataset == "hdbpmn":
            manifests.append(build_hdbpmn(collect(root, "hdbpmn/coco_icdar2021", ("train", "val", "test")), seed=args.seed))
        else:
            manifests.append(build_cmp(dataset, collect(root, f"{dataset}/source", ("train", "val", "test")), seed=args.seed))
    out.mkdir(parents=True, exist_ok=True)
    for manifest in manifests:
        verify_disjoint(manifest)
        path = out / f"{manifest['version']}.json"
        if path.exists():
            raise ValueError(f"{path} exists; split manifests are versioned, use a new version")
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(path, json.dumps(summary(manifest), ensure_ascii=False))


if __name__ == "__main__":
    main()
