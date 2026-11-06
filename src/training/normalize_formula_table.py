"""Turn formula and table sources into normalized.jsonl, with a seeded cap per split.

Full sources are far larger than one GPU can train on (about 1.45 images per second), so each
split takes a reproducible sample. Raising the caps needs only a rerun into a moved-aside file.

A group or an exact image may appear in one split only: the stricter split keeps it, in the
order test, validation, train. Cell text is never a target; PaddleOCR handles it. Tables over
MAX_CELLS are left out, because their answers exceed the sequence budget and truncating an
answer would teach a wrong table.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re
import xml.etree.ElementTree as ET

from .prepare import sha256
from .schema import require, validate_record, validate_table

SPLIT_PRIORITY = ("test", "validation", "train")
MAX_CELLS = 120
MAX_LATEX_CHARS = 1500
REVISIONS = {"crohme_2023": "zenodo-8428035", "unimer": "hf-2343ddd963290469da36ca83e3a56c66e068add9", "mathwriting": "mathwriting-2024",
             "pubtabnet": "hf-06963b1af16203f4633c718e2c50109eb57ed658", "fintabnet_c": "hf-e5673a90b98d02c4832f9e836d72762f0e8933a0"}
TRUTH = re.compile(r'<annotation type="truth">(.*?)</annotation>', re.S)
WRITER = re.compile(r'<annotation type="writer">(.*?)</annotation>', re.S)
NORMALIZED_LABEL = re.compile(r'<annotation type="normalizedLabel">(.*?)</annotation>', re.S)
TRACE = re.compile(r"<trace\b[^>]*>(.*?)</trace>", re.S)
SPAN = re.compile(r'(rowspan|colspan)="(\d+)"')


def unescape(text):
    return text.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&apos;", "'").replace("&amp;", "&").strip()


def clip_box(box, width, height):
    """Enclosing integer box inside the image, or None when nothing remains."""
    x0, y0 = max(0, math.floor(box[0])), max(0, math.floor(box[1]))
    x1, y1 = min(width, math.ceil(box[2])), min(height, math.ceil(box[3]))
    return [x0, y0, x1, y1] if x0 < x1 and y0 < y1 else None


def html_table_cells(tokens, cells, width, height):
    """Lay PubTabNet structure tokens onto a grid. Returns (rows, cols, compact cells)."""
    occupied, placed = set(), []
    row, col, index = -1, 0, 0
    spans = {"rowspan": 1, "colspan": 1}
    for token in tokens:
        if token == "<tr>":
            row, col = row + 1, 0
        elif token in ("<td>", "<td"):
            spans = {"rowspan": 1, "colspan": 1}
        elif "span=" in token:
            for name, value in SPAN.findall(token):
                spans[name] = int(value)
        elif token == "</td>":
            while (row, col) in occupied:
                col += 1
            cell = [row, col, spans["rowspan"], spans["colspan"]]
            box = cells[index].get("bbox") if index < len(cells) else None
            clipped = clip_box(box, width, height) if box else None
            placed.append(cell + clipped if clipped else cell)
            occupied |= {(r, c) for r in range(row, row + spans["rowspan"]) for c in range(col, col + spans["colspan"])}
            col += spans["colspan"]
            index += 1
    require(index == len(cells), "structure and cell list disagree")
    return row + 1, max((c for _, c in occupied), default=-1) + 1, placed


def overlaps(low, high, span_low, span_high):
    """More than half of a row or column lies inside the span."""
    return min(high, span_high) - max(low, span_low) > (high - low) / 2


def voc_table(path):
    """FinTabNet.c: build a grid from rows, columns and spanning cells. Returns (width, height, table box, rows, cols, cells)."""
    root = ET.parse(path).getroot()
    size = root.find("size")
    width, height = int(size.findtext("width")), int(size.findtext("height"))
    objects = {}
    for item in root.iter("object"):
        box = item.find("bndbox")
        objects.setdefault(item.findtext("name"), []).append([float(box.findtext(key)) for key in ("xmin", "ymin", "xmax", "ymax")])
    tables = objects.get("table", [])
    rows = sorted(objects.get("table row", []), key=lambda b: b[1])
    columns = sorted(objects.get("table column", []), key=lambda b: b[0])
    if len(tables) != 1 or not rows or not columns:
        return None
    spanning = objects.get("table spanning cell", []) + objects.get("table projected row header", [])
    occupied, cells = set(), []
    for span in spanning:
        covered_rows = [i for i, b in enumerate(rows) if overlaps(b[1], b[3], span[1], span[3])]
        covered_cols = [i for i, b in enumerate(columns) if overlaps(b[0], b[2], span[0], span[2])]
        if not covered_rows or not covered_cols:
            return None
        spanned = {(r, c) for r in range(covered_rows[0], covered_rows[-1] + 1) for c in range(covered_cols[0], covered_cols[-1] + 1)}
        if spanned & occupied:
            return None
        occupied |= spanned
        box = clip_box(span, width, height)
        head = [covered_rows[0], covered_cols[0], covered_rows[-1] - covered_rows[0] + 1, covered_cols[-1] - covered_cols[0] + 1]
        cells.append(head + box if box else head)
    for r, row in enumerate(rows):
        for c, column in enumerate(columns):
            if (r, c) not in occupied:
                box = clip_box([column[0], row[1], column[2], row[3]], width, height)
                cells.append([r, c, 1, 1] + box if box else [r, c, 1, 1])
    cells.sort()
    return width, height, clip_box(tables[0], width, height), len(rows), len(columns), cells


def render_ink(inkml_text, output, line_width=3, target_height=224, pad=16, max_width=1600):
    """Draw MathWriting strokes on white; the ink has no image of its own."""
    from PIL import Image, ImageDraw

    strokes = []
    for trace in TRACE.findall(inkml_text):
        points = [tuple(float(v) for v in point.split()[:2]) for point in trace.split(",") if point.strip()]
        if points:
            strokes.append(points)
    require(bool(strokes), "ink without strokes")
    xs, ys = [p[0] for s in strokes for p in s], [p[1] for s in strokes for p in s]
    width, height = max(max(xs) - min(xs), 1.0), max(max(ys) - min(ys), 1.0)
    scale = min(target_height / height, max_width / width)
    canvas = Image.new("L", (round(width * scale) + 2 * pad, round(height * scale) + 2 * pad), 255)
    draw = ImageDraw.Draw(canvas)
    for stroke in strokes:
        points = [((x - min(xs)) * scale + pad, (y - min(ys)) * scale + pad) for x, y in stroke]
        draw.line(points if len(points) > 1 else points * 2, fill=0, width=line_width, joint="curve")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


# Each source returns candidates: {split, id, image (relative), groups, and how to build the target}.
def crohme_candidates(base):
    root = base / "source/TC11_CROHME23"
    folders = {"train": ["CROHME2019"], "val": ["CROHME2016_test", "CROHME2023_val"], "test": ["CROHME2019_test", "CROHME2023_test"]}
    names = {"train": "train", "val": "validation", "test": "test"}
    for partition, subsets in folders.items():
        for subset in subsets:
            for image in sorted((root / "IMG" / partition / subset).rglob("*.png")):
                ink = root / "INKML" / partition / subset / f"{image.stem}.inkml"
                if ink.exists():
                    yield {"split": names[partition], "id": f"{subset}:{image.stem}", "image": image.relative_to(base).as_posix(), "latex_from": str(ink)}


def crohme_latex(candidate):
    text = Path(candidate["latex_from"]).read_text(encoding="utf-8", errors="replace")
    truth, writer = TRUTH.search(text), WRITER.search(text)
    if not truth:
        return None, None
    latex = unescape(truth.group(1)).strip("$ \n")
    return latex, [f"crohme:writer:{writer.group(1).strip()}"] if writer and writer.group(1).strip() else [f"crohme:sample:{candidate['id']}"]


def indexed_lines(text_path, image_dir):
    """UniMER labels: line N belongs to image N, and some images are withheld for copyright."""
    lines = text_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for image in sorted(image_dir.iterdir()):
        index = int(image.stem)
        if index < len(lines) and lines[index].strip():
            yield image, lines[index].strip()


def unimer_candidates(base):
    train = base / "source_1m/UniMER-1M"
    for image, latex in indexed_lines(train / "train.txt", train / "images"):
        bucket = int(hashlib.sha256(image.stem.encode()).hexdigest(), 16) % 100
        yield {"split": "validation" if bucket == 0 else "train", "id": f"1m:{image.stem}", "image": image.relative_to(base).as_posix(), "latex": latex}
    test = base / "source/UniMER-Test"
    for subset in ("hwe", "spe", "cpe", "sce"):
        for image, latex in indexed_lines(test / f"{subset}.txt", test / subset):
            yield {"split": "test", "id": f"{subset}:{image.stem}", "image": image.relative_to(base).as_posix(), "latex": latex}


def mathwriting_candidates(base):
    root = base / "source/mathwriting-2024"
    for partition, split in (("train", "train"), ("valid", "validation"), ("test", "test")):
        for ink in sorted((root / partition).glob("*.inkml")):
            yield {"split": split, "id": ink.stem, "image": f"rendered/{partition}/{ink.stem}.png", "ink": str(ink)}


def pubtabnet_candidates(base):
    """Official labels cover train and val only; a stable half of val articles becomes test."""
    root = base / "source/pubtabnet"
    with (root / "PubTabNet_2.0.0.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            article = row["filename"].split("_")[0]
            split = "train"
            if row["split"] == "val":
                split = "test" if int(hashlib.sha256(article.encode()).hexdigest(), 16) % 2 else "validation"
            if len(row["html"]["cells"]) <= MAX_CELLS:
                yield {"split": split, "id": Path(row["filename"]).stem, "image": f"source/pubtabnet/{row['split']}/{row['filename']}",
                       "groups": [f"pubtabnet:article:{article}"], "html": row["html"]}


def fintabnet_candidates(base):
    root = base / "source/FinTabNet.c-Structure"
    for partition, split in (("train", "train"), ("val", "validation"), ("test", "test")):
        for path in sorted((root / partition).glob("*.xml")):
            document = path.stem.split("_page_")[0]
            yield {"split": split, "id": path.stem, "image": f"source/FinTabNet.c-Structure/images/{path.stem}.jpg", "voc": str(path),
                   "groups": [f"fintabnet:document:{document}"]}


SOURCES = {"crohme_2023": crohme_candidates, "unimer": unimer_candidates, "mathwriting": mathwriting_candidates,
           "pubtabnet": pubtabnet_candidates, "fintabnet_c": fintabnet_candidates}


def build_target(candidate, dataset, base, width, height):
    """Returns (tasks, target, groups), or None to skip the candidate."""
    groups = candidate.get("groups")
    if "html" in candidate:
        rows, cols, cells = html_table_cells(candidate["html"]["structure"]["tokens"], candidate["html"]["cells"], width, height)
        table = {"id": "t1", "bbox": [0, 0, width, height], "rows": rows, "cols": cols, "cells": cells}
        try:
            validate_table(table, width, height)
        except ValueError:  # A few source tables have spans past the grid; that answer is wrong.
            return None
        return ["tables"], {"tables": [table]}, groups
    if "voc" in candidate:
        parsed = voc_table(candidate["voc"])
        if parsed is None or parsed[2] is None or len(parsed[5]) > MAX_CELLS or (parsed[0], parsed[1]) != (width, height):
            return None
        return ["tables"], {"tables": [{"id": "t1", "bbox": parsed[2], "rows": parsed[3], "cols": parsed[4], "cells": parsed[5]}]}, groups
    if "latex_from" in candidate:
        latex, groups = crohme_latex(candidate)
    elif "ink" in candidate:
        match = NORMALIZED_LABEL.search(Path(candidate["ink"]).read_text(encoding="utf-8", errors="replace"))
        latex, groups = (unescape(match.group(1)) if match else None), [f"mathwriting:sample:{candidate['id']}"]
    else:
        latex, groups = candidate["latex"], [f"unimer:image:{candidate['id']}"]
    if not latex or len(latex) > MAX_LATEX_CHARS:
        return None
    return ["formulas"], {"formulas": [{"id": "f1", "bbox": [0, 0, width, height], "latex": latex}]}, groups


def build(dataset, base, caps, seed=42):
    """Sample each split, strictest first, skipping anything already used by a stricter split."""
    from PIL import Image

    pools = {split: [] for split in SPLIT_PRIORITY}
    for candidate in SOURCES[dataset](base):
        pools[candidate["split"]].append(candidate)
    rows, taken, group_split, image_hashes, skipped = [], Counter(), {}, set(), Counter()
    for split in SPLIT_PRIORITY:
        random.Random(f"{seed}:{dataset}:{split}").shuffle(pools[split])
        for candidate in pools[split]:
            if taken[split] >= caps[split]:
                break
            if "ink" in candidate and not (base / candidate["image"]).exists():
                render_ink(Path(candidate["ink"]).read_text(encoding="utf-8", errors="replace"), base / candidate["image"])
            image_path = base / candidate["image"]
            if not image_path.is_file():
                skipped["missing_image"] += 1
                continue
            with Image.open(image_path) as image:
                width, height = image.size
                if image.getexif().get(274, 1) != 1:
                    skipped["exif_rotated"] += 1
                    continue
            built = build_target(candidate, dataset, base, width, height)
            if built is None:
                skipped["no_usable_target"] += 1
                continue
            tasks, target, groups = built
            if any(group_split.get(group, split) != split for group in groups):
                skipped["group_in_stricter_split"] += 1
                continue
            digest = sha256(image_path)
            if digest in image_hashes:
                skipped["duplicate_image"] += 1
                continue
            image_hashes.add(digest)
            group_split.update({group: split for group in groups})
            taken[split] += 1
            rows.append(validate_record({"id": candidate["id"], "dataset": dataset, "image": candidate["image"], "width": width, "height": height,
                                         "split": split, "groups": groups, "source_revision": REVISIONS[dataset], "annotation_origin": "source_annotation",
                                         "tasks": tasks, "coverage": {task: "complete" for task in tasks}, "target": target}))
    return rows, dict(taken), dict(skipped), {split: len(pool) for split, pool in pools.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(SOURCES), required=True)
    parser.add_argument("--root", default="data/raw/radu")
    parser.add_argument("--train", type=int, default=5000)
    parser.add_argument("--validation", type=int, default=200)
    parser.add_argument("--test", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    base = Path(args.root) / args.dataset
    output = base / "normalized.jsonl"
    require(not output.exists(), f"{output} exists; move it aside deliberately before rebuilding")
    rows, taken, skipped, available = build(args.dataset, base, {"train": args.train, "validation": args.validation, "test": args.test}, args.seed)
    for split in ("train", "validation", "test"):
        require(taken.get(split), f"{args.dataset}: no {split} samples")
    output.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"dataset": args.dataset, "taken": taken, "skipped": skipped, "available": available}, ensure_ascii=False))


if __name__ == "__main__":
    main()
