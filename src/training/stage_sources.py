"""Index real source assets and stage structure labels.

This is an acquisition/staging catalog, NOT normalized.jsonl accepted by prepare.
Source-specific split review and label checks remain explicit readiness gates. OCR is not
a training input; `ocr: null` records that staging ran none.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import zipfile


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    tmp.replace(path)
    return count


def stage_coco(root, out, dataset, source_dir):
    from PIL import Image
    stats = {"samples":0, "splits":Counter(), "nodes":0, "arrows":0, "relations":0, "issues":[], "training_ready":False,
             "next":"Review FC_A validation/group policy; validate normalized records before prepare."}
    def rows():
        for annotation in sorted(source_dir.glob("*.json")):
            split = {"val":"validation"}.get(annotation.stem, annotation.stem)
            document = json.loads(annotation.read_text(encoding="utf-8"))
            per_image = defaultdict(list)
            for item in document["annotations"]:
                per_image[item["image_id"]].append(item)
            for image in document["images"]:
                path = source_dir / annotation.stem / image["file_name"]
                image_issues = []
                if not path.is_file():
                    image_issues.append("missing_image")
                else:
                    try:
                        with Image.open(path) as im:
                            if im.size != (image["width"], image["height"]):
                                image_issues.append("dimension_mismatch")
                            im.verify()
                    except Exception as exc:
                        image_issues.append("image_decode:" + str(exc))
                nodes, arrows, relations, reference_text = [], [], [], []
                for item in per_image[image["id"]]:
                    aid = f"ann-{item['id']}"
                    x, y, w, h = item["bbox"]
                    box = [x, y, x + w, y + h]
                    if item["category"] == "text":
                        # Source text geometry is correspondence evidence, not model OCR.
                        reference_text.append({"source_id":aid, "bbox":box})
                    elif item["category"] in {"arrow", "sequenceFlow", "messageFlow", "dataAssociation"}:
                        keypoints = item.get("keypoints", [])
                        start = keypoints[:2] if len(keypoints) >= 6 and keypoints[2] > 0 else None
                        end = keypoints[3:5] if len(keypoints) >= 6 and keypoints[5] > 0 else None
                        arrows.append({"id":aid, "bbox":box, "start":start, "end":end})
                        if "arrow_prev" not in item or "arrow_next" not in item:
                            image_issues.append("missing_connectivity:" + aid)
                        else:
                            relations.append({"id":"edge-" + str(item["id"]), "source":None if item["arrow_prev"] is None else "ann-" + str(item["arrow_prev"]),
                                              "target":None if item["arrow_next"] is None else "ann-" + str(item["arrow_next"]), "directed":True, "arrow_id":aid})
                    else:
                        nodes.append({"id":aid, "bbox":box, "kind":item["category"]})
                writer = re.search(r"writer(\d+)", image["file_name"])
                groups = [f"{dataset}:writer:{writer.group(1)}"] if writer else []
                stats["samples"] += 1
                stats["splits"][split] += 1
                stats["nodes"] += len(nodes)
                stats["arrows"] += len(arrows)
                stats["relations"] += len(relations)
                if image_issues:
                    stats["issues"].append({"file":path.relative_to(root).as_posix(), "issues":image_issues})
                yield {"dataset":dataset, "id":annotation.stem + ":" + str(image["id"]), "source_split":split,
                       "image":path.relative_to(root).as_posix(), "width":image["width"], "height":image["height"],
                       "annotation_file":annotation.relative_to(root).as_posix(), "annotation_image_id":image["id"],
                       "groups":groups, "ocr":None, "ocr_status":"not_run", "training_ready":False,
                       "stage":"awaiting_split_review","issues":image_issues, "reference_text_regions":reference_text,
                       "target":{"nodes":nodes, "arrows":arrows, "relations":relations}}
    write_rows(out / dataset / "staged.jsonl", rows())
    return stats


def stage_didi(root, out):
    source = root / "didi/source"
    stats = {"samples":0, "splits":Counter(), "by_variant":{}, "training_ready":False,
             "next":"Render actual strokes; establish verified symbol/connection correspondence. Prompt DOT is not handwriting localization GT."}
    def rows():
        for path in sorted(source.glob("*.ndjson")):
            count = 0
            variant = "no_text" if "wo_text" in path.name else "text"
            with path.open("rb") as handle:
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    item = json.loads(line)
                    split = {"valid":"validation", "val":"validation"}.get(item["split"], item["split"])
                    stats["samples"] += 1
                    stats["splits"][split] += 1
                    count += 1
                    yield {"dataset":"didi", "id":variant + ":" + item["key"], "variant":variant,
                           "source_file":path.relative_to(root).as_posix(), "byte_offset":offset, "byte_length":len(line),
                           "source_split":split, "prompt_id":item["label_id"], "groups":["didi:prompt:" + item["label_id"]],
                           "stroke_count":len(item["drawing"]), "image":None, "ocr":None,
                           "training_ready":False, "stage":"raw_ink_indexed_no_verified_structure_gt"}
            stats["by_variant"][variant] = count
    write_rows(out / "didi" / "catalog.jsonl", rows())
    return stats


def stage_flowlearn(root, out):
    source = root / "flowlearn/source"
    stats = {"samples":0, "by_subset":Counter(), "splits":Counter(), "missing_images":[], "training_ready":False,
             "scientific_duplicate_references":[], "scientific_cross_split_images":[],
             "next":"Verify synthetic geometry against corresponding JPEG variant. Scientific subset has no equivalent complete graph GT."}
    scientific_references = defaultdict(list)
    def rows():
        for variant, image_dir in (("char", "mermaid_char_v2"), ("word", "mermaid_word")):
            for split in ("train", "test"):
                annotation = source / "SimFlowchart" / variant / (split + ".json")
                if not annotation.is_file():
                    continue
                items = json.loads(annotation.read_text(encoding="utf-8"))
                for index, item in enumerate(items):
                    path = source / "SimFlowchart/images" / image_dir / "jpeg" / item["file"]
                    stats["samples"] += 1
                    stats["by_subset"]["synthetic_" + variant] += 1
                    stats["splits"][split] += 1
                    if not path.is_file():
                        stats["missing_images"].append(path.relative_to(root).as_posix())
                    yield {"dataset":"flowlearn", "id":f"{variant}:{item['file']}", "source_split":split,
                           "image":path.relative_to(root).as_posix(), "annotation_file":annotation.relative_to(root).as_posix(), "annotation_index":index,
                           "groups":["flowlearn:synthetic:" + Path(item["file"]).stem], "ocr":None, "training_ready":False,
                           "stage":"synthetic_source_geometry_available",
                           "source_nodes":len(item.get("meta", {}).get("text", {})), "source_links":len(item.get("meta", {}).get("links", {}))}
        for split in ("train", "test"):
            annotation = source / "SciFlowchart" / (split + ".json")
            if not annotation.is_file():
                continue
            for index, item in enumerate(json.loads(annotation.read_text(encoding="utf-8"))):
                path = source / "SciFlowchart/images" / item["image_file"]
                stats["samples"] += 1
                stats["by_subset"]["scientific"] += 1
                stats["splits"][split] += 1
                if not path.is_file():
                    stats["missing_images"].append(path.relative_to(root).as_posix())
                scientific_references[item["image_file"]].append({"split":split, "index":index})
                yield {"dataset":"flowlearn", "id":f"sci:{split}:{index}:" + item["image_file"], "source_split":split,
                       "image":path.relative_to(root).as_posix(), "annotation_file":annotation.relative_to(root).as_posix(), "annotation_index":index,
                       "groups":["flowlearn:paper:" + item["file_name"]], "ocr":None, "training_ready":False,
                       "stage":"image_and_document_context_only_no_complete_structure_gt"}
    write_rows(out / "flowlearn" / "catalog.jsonl", rows())
    stats["scientific_unique_images"] = len(scientific_references)
    stats["scientific_duplicate_references"] = [{"image":name, "references":refs} for name,refs in scientific_references.items() if len(refs) > 1]
    stats["scientific_cross_split_images"] = [name for name,refs in scientific_references.items() if len({ref["split"] for ref in refs}) > 1]
    return stats


def archive_inventory(root):
    result = []
    for path in sorted(root.rglob("*.zip")):
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            result.append({"path":path.relative_to(root).as_posix(), "bytes":path.stat().st_size,
                           "files":sum(not i.is_dir() for i in entries),
                           "encrypted_files":sum(bool(i.flag_bits & 1) for i in entries),
                           "uncompressed_bytes":sum(i.file_size for i in entries)})
    return result


def load_rst_annotations(source, rst_dir):
    """Check native graph references; preserve null connectivity as missing data."""
    rst_dir = Path(rst_dir)
    categories = json.loads((rst_dir.parent / "categories_ai2d-rst.json").read_text(encoding="utf-8"))
    annotations = {}
    stats = {"rst_annotations":0, "rst_matched_images":0, "rst_connectivity_present":0,
             "rst_connectivity_null":0, "rst_connectivity_edges":0, "rst_edge_kinds":Counter()}
    for path in sorted(rst_dir.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        image_name = path.name.removesuffix(".json")
        if str(record["id"]) + ".png" != image_name or image_name in annotations:
            raise ValueError(f"RST duplicate/mismatched diagram ID: {path}")
        if not (source / "images" / image_name).is_file() or not (source / "annotations" / path.name).is_file():
            raise ValueError(f"RST image/segmentation missing: {image_name}")
        base = json.loads((source / "annotations" / path.name).read_text(encoding="utf-8"))
        original_ids = {key for field in ("arrows","blobs","text","arrowHeads","containers","imageConsts","meaningfulSpaces") for key in base.get(field, {})}
        grouping = record.get("grouping")
        if not isinstance(grouping, list):
            raise ValueError(f"invalid RST grouping: {path}")
        groups = {node["id"]:node for node in grouping}
        if len(groups) != len(grouping):
            raise ValueError(f"duplicate RST grouping node: {path}")
        for node in grouping:
            if node.get("data", {}).get("kind") != "group" and node["id"] not in original_ids:
                raise ValueError(f"RST element absent from original segmentation: {path}:{node['id']}")
            if any(edge["nodeTo"] not in groups for edge in node.get("adjacencies", [])):
                raise ValueError(f"dangling RST grouping adjacency: {path}")
        connectivity = record.get("connectivity")
        if connectivity is None:
            stats["rst_connectivity_null"] += 1
        else:
            if not isinstance(connectivity, dict) or not isinstance(connectivity.get("nodes"), list) or not isinstance(connectivity.get("edges"), list):
                raise ValueError(f"invalid RST connectivity: {path}")
            stats["rst_connectivity_present"] += 1
            connected_ids = {item[0] for item in connectivity["nodes"]}
            if len(connected_ids) != len(connectivity["nodes"]) or not connected_ids <= set(groups):
                raise ValueError(f"invalid RST connectivity node references: {path}")
            for start, end, attrs in connectivity["edges"]:
                if start not in connected_ids or end not in connected_ids:
                    raise ValueError(f"dangling RST connectivity endpoint: {path}")
                if attrs.get("kind") not in {"directional", "undirectional", "bidirectional"}:
                    raise ValueError(f"unknown RST connectivity kind: {path}")
                stats["rst_edge_kinds"][attrs["kind"]] += 1
                stats["rst_connectivity_edges"] += 1
        annotations[image_name] = {"path":path, "record":record, "category":categories.get(image_name)}
        stats["rst_annotations"] += 1
        stats["rst_matched_images"] += 1
    if not annotations or set(categories) != set(annotations):
        raise ValueError("RST category inventory differs from actual annotation IDs")
    return annotations, stats


def find_rst_directory(root):
    base = root / "ai2d_rst"
    found = []
    for location in (base / "rst_annotations", base / "ai2d-rst-v1-1"):
        if location.is_dir():
            for categories in location.rglob("categories_ai2d-rst.json"):
                candidate = categories.parent / "ai2d-rst"
                if candidate.is_dir():
                    found.append(candidate)
    if len(found) > 1:
        raise ValueError("multiple RST corpus copies found; specify --rst-dir explicitly")
    return found[0] if found else None


def stage_ai2d(root, out, rst_dir=None):
    from PIL import Image
    source = root / "ai2d_rst/source/ai2d"
    rst_dir = Path(rst_dir) if rst_dir else find_rst_directory(root)
    rst, rst_stats = load_rst_annotations(source, rst_dir) if rst_dir else ({}, {"rst_annotations":0})
    stats = {"samples":0, "arrows":0, "relationships":0, "issues":[], "training_ready":False, **rst_stats,
             "next":"Normalize native group/connection annotations and review splits. Null connectivity is not a complete empty graph." if rst else "Acquire RST corpus and join by diagram ID; retain original AI2D segmentation/relations separately."}
    subset = []
    def rows():
        for annotation in sorted((source / "annotations").glob("*.json")):
            image_name = annotation.name.removesuffix(".json")
            image = source / "images" / image_name
            document = json.loads(annotation.read_text(encoding="utf-8"))
            issue = None
            width = height = None
            try:
                with Image.open(image) as im:
                    width, height = im.size
                    im.verify()
            except Exception as exc:
                issue = str(exc)
                stats["issues"].append({"file":image.relative_to(root).as_posix(), "error":issue})
            stats["samples"] += 1
            stats["arrows"] += len(document.get("arrows", {}))
            stats["relationships"] += len(document.get("relationships", {}))
            extra = rst.get(image_name)
            connectivity = extra["record"]["connectivity"] if extra else None
            row = {"dataset":"ai2d_rst", "id":image_name, "image":image.relative_to(root).as_posix(), "width":width, "height":height,
                   "annotation_file":annotation.relative_to(root).as_posix(), "rst_annotation":extra["path"].relative_to(root).as_posix() if extra else None,
                   "source_arrows":len(document.get("arrows", {})), "source_relationships":len(document.get("relationships", {})),
                   "source_split":None, "groups":["ai2d:diagram:" + image_name], "ocr":None,
                   "training_ready":False, "stage":"rst_joined_awaiting_native_adapter" if extra else "original_ai2d_outside_rst_subset" if rst else "original_ai2d_available_rst_corpus_missing",
                   "rst_connectivity_status":"present" if connectivity is not None else "null_not_supervised" if extra else "not_in_rst_subset", "issue":issue}
            if extra:
                subset.append({**row, "rst_category":extra["category"], "source_structure":{"grouping":extra["record"]["grouping"], "connectivity":connectivity}})
            yield row
    write_rows(out / "ai2d_rst" / "catalog.jsonl", rows())
    write_rows(out / "ai2d_rst" / "rst_catalog.jsonl", subset)
    return stats


def stage_casia(root, out):
    archive_path = root / "casia_ohfc/archives/CASIA-OHFC-Release Edition.zip"
    stats = {"samples":0, "encrypted":True, "training_ready":False,
             "next":"Research group must complete provider application and obtain password; no agreement submitted by this tool."}
    def rows():
        with zipfile.ZipFile(archive_path) as archive:
            for item in archive.infolist():
                if not item.filename.endswith(".inkml"):
                    continue
                stats["samples"] += 1
                yield {"dataset":"casia_ohfc", "id":Path(item.filename).stem,
                       "source_archive":archive_path.relative_to(root).as_posix(), "archive_entry":item.filename,
                       "encrypted":bool(item.flag_bits & 1), "image":None, "ocr":None,
                       "source_split":None, "training_ready":False, "stage":"encrypted_source_requires_provider_password"}
    write_rows(out / "casia_ohfc" / "catalog.jsonl", rows())
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw/radu")
    parser.add_argument("--output", default="data/processed/radu_staging")
    parser.add_argument("--rst-dir", help="Directory containing the native RST diagram JSONs")
    args = parser.parse_args()
    root, out = Path(args.root), Path(args.output)
    result = {"created_at":datetime.now(timezone.utc).isoformat(), "stage":"raw_and_source_annotation_staging", "training_ready":False, "sources":{}}
    for dataset, sub in (("fc_a", "source"), ("hdbpmn", "coco_icdar2021")):
        if (root / dataset / sub).is_dir():
            result["sources"][dataset] = stage_coco(root, out, dataset, root / dataset / sub)
    if (root / "didi/source").is_dir():
        result["sources"]["didi"] = stage_didi(root, out)
    if (root / "flowlearn/source").is_dir():
        result["sources"]["flowlearn"] = stage_flowlearn(root, out)
    if (root / "ai2d_rst/source/ai2d").is_dir():
        result["sources"]["ai2d_rst"] = stage_ai2d(root, out, args.rst_dir)
    if (root / "casia_ohfc/archives/CASIA-OHFC-Release Edition.zip").is_file():
        result["sources"]["casia_ohfc"] = stage_casia(root, out)
    result["archives"] = archive_inventory(root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "inventory.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key:{k:v for k,v in value.items() if k not in ("issues", "missing_images")} for key,value in result["sources"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
