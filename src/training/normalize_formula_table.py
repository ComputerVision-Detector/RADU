"""Turn formula, table and box-only diagram sources into normalized.jsonl.

Train keeps every usable sample unless --train caps it; the sampler decides how many are seen.
Validation and test keep a seeded sample, so evaluation stays short.

A group or an exact image may appear in one split only: the stricter split keeps it, in the
order test, validation, train. An image that also appears in another source is taken from one
source only. Cell text is never a target; PaddleOCR handles it. Tables over MAX_CELLS drop cell
boxes and keep the grid, and tables over MAX_GRID_CELLS are left out, because their answers
exceed the sequence budget and truncating an answer would teach a wrong table. token_scan makes
the final check against the real budget.
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
MAX_CELLS = 150
MAX_GRID_CELLS = 400
MAX_LATEX_CHARS = 1500
MAX_PAGE_ANSWER_CHARS = 9000
HDFLOWMIND_NODES = {"rectangle", "circle", "parallelogram", "triangle", "diamonds", "hexagon", "long_oval", "trapezoid"}
HDFLOWMIND_ARROWS = {"arrow", "double_arrow", "line"}
FOREIGN_IMAGES = {"hme100k": "unimer/source/UniMER-Test/hwe/*", "ctdar_2019": "read_abp/source/READ_ABP_TABLE/*/img/*"}
GROUPED_HELD_OUT = {"ibem"}  # Fewer papers in held-out splits keep more of them for training.
CJK = re.compile("[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]")
REVISIONS = {"crohme_2023": "zenodo-8428035", "unimer": "hf-2343ddd963290469da36ca83e3a56c66e068add9", "mathwriting": "mathwriting-2024",
             "pubtabnet": "hf-06963b1af16203f4633c718e2c50109eb57ed658", "fintabnet_c": "hf-e5673a90b98d02c4832f9e836d72762f0e8933a0",
             "hdflowmind": "hf-e40d72d9483afbf061843c820a9cb7360f1d6443", "hme100k": "user-supplied-archive-2026-11-07",
             "ibem": "zenodo-21528113", "tabrecset": "figshare-39439612", "ctdar_2019": "git-171934db38a9e230e23b87a5afe86f8c83edaa95",
             "read_abp": "zenodo-1243098"}
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
            if len(row["html"]["cells"]) <= MAX_GRID_CELLS:
                yield {"split": split, "id": Path(row["filename"]).stem, "image": f"source/pubtabnet/{row['split']}/{row['filename']}",
                       "groups": [f"pubtabnet:article:{article}"], "html": row["html"]}


def fintabnet_candidates(base):
    root = base / "source/FinTabNet.c-Structure"
    for partition, split in (("train", "train"), ("val", "validation"), ("test", "test")):
        for path in sorted((root / partition).glob("*.xml")):
            document = path.stem.split("_page_")[0]
            yield {"split": split, "id": path.stem, "image": f"source/FinTabNet.c-Structure/images/{path.stem}.jpg", "voc": str(path),
                   "groups": [f"fintabnet:document:{document}"]}


def stable_split(key, test, validation):
    """Percent of keys sent to test and validation; the same key always lands in the same split."""
    bucket = int(hashlib.sha256(key.encode()).hexdigest(), 16) % 100
    return "test" if bucket < test else "validation" if bucket < test + validation else "train"


def polygon_box(points):
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def coords_box(text):
    return polygon_box([[float(v) for v in pair.split(",")[:2]] for pair in text.split()])


def table_target(raw_tables, width, height, groups):
    """raw_tables: (box, [(row, col, rowspan, colspan, box)]), all tables of the page or nothing."""
    total = sum(len(cells) for _, cells in raw_tables)
    if not raw_tables or not all(cells for _, cells in raw_tables) or total > MAX_GRID_CELLS:
        return None
    tables = []
    for box, cells in sorted(raw_tables, key=lambda table: (table[0][1], table[0][0])):
        table_box = clip_box(box, width, height)
        if table_box is None:
            return None
        compact = []
        for *head, cell_box in sorted(cells, key=lambda cell: cell[:2]):
            clipped = clip_box(cell_box, width, height) if total <= MAX_CELLS else None
            compact.append(head + clipped if clipped else head)
        table = {"id": f"t{len(tables) + 1}", "bbox": table_box, "rows": max(c[0] + c[2] for c in compact), "cols": max(c[1] + c[3] for c in compact), "cells": compact}
        try:
            validate_table(table, width, height)
        except ValueError:  # Overlapping or impossible source cells: that answer would be wrong.
            return None
        # A source that omits empty cells does not fully annotate the grid.
        if sum(c[2] * c[3] for c in compact) != table["rows"] * table["cols"]:
            return None
        tables.append(table)
    return ["tables"], {"tables": tables}, groups


def hdflowmind_drawings(root):
    """One drawing was photographed in variants under nearby numbers, so link similar neighbours."""
    from PIL import Image, ImageOps

    def fingerprint(path):
        with Image.open(path) as image:
            pixels = list(ImageOps.exif_transpose(image).convert("L").resize((9, 8)).getdata())
        return sum(1 << i for i in range(72) if i % 9 != 8 and pixels[i] > pixels[i + 1])

    items = []
    for split in ("train", "validation", "test"):
        for path in (root / split / "images").glob("*.jpg"):
            prefix, number = path.stem.split("_")
            items.append((prefix, int(number), path.stem, fingerprint(path)))
    items.sort()
    parent = {item[2]: item[2] for item in items}

    def find(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    for index, item in enumerate(items):
        for other in items[index + 1:]:
            if other[0] != item[0] or other[1] - item[1] > 3:
                break
            if bin(item[3] ^ other[3]).count("1") <= 6:
                parent[find(other[2])] = find(item[2])
    return {name: find(name) for name in parent}


def hdflowmind_candidates(base):
    from PIL import Image, ImageOps

    root = base / "source/hdflowmind"
    drawings = hdflowmind_drawings(root)
    # "simple" holds single shapes without arrows, so it is not a diagram sample.
    for split in ("train", "validation", "test"):
        for path in sorted((root / split / "annotations").glob("*.xml")):
            image = f"source/hdflowmind/{split}/images/{path.stem}.jpg"
            # Labels follow the displayed orientation (checked on overlays), so rotated photos are saved upright.
            with Image.open(base / image) as handle:
                if handle.getexif().get(274, 1) != 1:
                    image = f"upright/{split}/{path.stem}.jpg"
                    if not (base / image).exists():
                        (base / image).parent.mkdir(parents=True, exist_ok=True)
                        ImageOps.exif_transpose(handle).convert("RGB").save(base / image, quality=95, subsampling=0)
            yield {"split": split, "id": path.stem, "image": image, "xml": str(path),
                   "groups": [f"hdflowmind:drawing:{drawings[path.stem]}"], "build": hdflowmind_target}


def hdflowmind_target(candidate, width, height):
    """Boxes only: most arrows have no head point and no sample has connectivity."""
    root = ET.parse(candidate["xml"]).getroot()
    if (int(root.findtext("size/width")), int(root.findtext("size/height"))) != (width, height):
        return None
    boxes = {"nodes": [], "arrows": []}
    for item in root.iter("object"):
        name, box = item.findtext("name"), item.find("bndbox")
        task = "nodes" if name in HDFLOWMIND_NODES else "arrows" if name in HDFLOWMIND_ARROWS else None
        if task and box is not None:
            clipped = clip_box([float(box.findtext(key)) for key in ("xmin", "ymin", "xmax", "ymax")], width, height)
            if clipped is None:
                return None
            boxes[task].append((clipped, name))
    if not boxes["nodes"] and not boxes["arrows"]:
        return None
    order = lambda entry: (entry[0][1], entry[0][0], entry[0][3], entry[0][2])
    nodes = [{"id": f"n{i + 1}", "bbox": box, "kind": name} for i, (box, name) in enumerate(sorted(boxes["nodes"], key=order))]
    arrows = [{"id": f"a{i + 1}", "bbox": box, "start": None, "end": None} for i, (box, _) in enumerate(sorted(boxes["arrows"], key=order))]
    return ["arrows", "nodes"], {"arrows": arrows, "nodes": nodes}, candidate["groups"]


def hme100k_candidates(base):
    """One label file and no official split in this archive; equal formulas share a split."""
    for line in (base / "source/train.txt").read_text(encoding="utf-8").splitlines():
        image, _, latex = line.partition("\t")
        compact = "".join(latex.split())
        # Numbered items and Chinese words in a label are natural language, not formula structure.
        if compact and not CJK.search(compact):
            yield {"split": stable_split(compact, 1, 1), "id": Path(image).stem, "image": f"source/{image}", "latex": latex.strip(),
                   "groups": [f"hme100k:latex:{hashlib.sha256(compact.encode()).hexdigest()}"], "build": single_formula_target}


def single_formula_target(candidate, width, height):
    if len(candidate["latex"]) > MAX_LATEX_CHARS:
        return None
    return ["formulas"], {"formulas": [{"id": "f1", "bbox": [0, 0, width, height], "latex": candidate["latex"]}]}, candidate["groups"]


def ibem_candidates(base):
    """Pages with labels only; a page missing from the label file is not a negative sample."""
    root = base / "source"
    labels = json.loads((root / "IBEM.json").read_text(encoding="utf-8"))
    for prefix, split in (("Ts", "test"), ("Va", "validation"), ("Tr", "train")):
        for listing in sorted((root / "partitions").glob(f"{prefix}*.lst")):
            for name in listing.read_text(encoding="utf-8").split():
                if name in labels:
                    paper, page = name.split("_page")
                    yield {"split": split, "id": name, "image": f"source/pages/{paper}_page{int(page):02d}.jpg", "page": labels[name],
                           "groups": [f"ibem:paper:{paper}"], "build": ibem_target}


def ibem_latex(text):
    """Same formula form as the other sources: no page-level math wrapper."""
    latex = text.strip().strip("$").strip()
    wrapped = re.fullmatch(r"\\begin\{equation\*\}(.*)\\end\{equation\*\}", latex, re.S)
    latex = wrapped.group(1).strip() if wrapped else latex
    return latex.replace("\\begin{align*}", "\\begin{aligned}").replace("\\end{align*}", "\\end{aligned}")


def ibem_target(candidate, width, height):
    """Label boxes are percent of the page; a page with one unusable formula is dropped whole.

    A formula broken across lines repeats its whole LaTeX on each piece, so no box matches its answer.
    """
    items = []
    for kind in ("embedded", "isolated"):
        for formula in candidate["page"].get(kind, {}).values():
            if formula.get("split", "single") != "single":
                return None
            box = clip_box([formula["x_min"] * width / 100, formula["y_min"] * height / 100, formula["x_max"] * width / 100, formula["y_max"] * height / 100], width, height)
            latex = ibem_latex(formula["latex_norm"])
            if box is None or not latex:
                return None
            items.append((box, latex))
    formulas = [{"id": f"f{i + 1}", "bbox": box, "latex": latex} for i, (box, latex) in enumerate(sorted(items, key=lambda item: (item[0][1], item[0][0])))]
    if not formulas or len(json.dumps(formulas, separators=(",", ":"))) > MAX_PAGE_ANSWER_CHARS:
        return None
    return ["formulas"], {"formulas": formulas}, candidate["groups"]


def tabrecset_group(stem):
    """Pages of one document and crops of one photo carry the same core name."""
    core = re.sub(r"_segR.*$", "", re.sub(r"^\d+_(?=[0-9a-f]{32})", "", stem))
    return re.sub(r"_(\d+|[a-z]{2,4})$", "", core)


def tabrecset_candidates(base):
    """All-line, three-line and no-line versions of a table share one annotation and one split."""
    root = base / "source/TabRecSet (CurveTabSet)"
    annotations = {path.stem: path for path in (root / "TSR_TCR_annotation").glob("*/*.json")}
    for folder in sorted(path for path in (root / "image").iterdir() if path.is_dir()):
        for image in sorted(folder.iterdir()):
            if image.stem in annotations:
                group = tabrecset_group(image.stem)
                yield {"split": stable_split(group, 3, 3), "id": f"{folder.name}:{image.stem}", "image": image.relative_to(base).as_posix(), "annotation": str(annotations[image.stem]),
                       "groups": [f"tabrecset:document:{group}"], "build": tabrecset_target}


def tabrecset_target(candidate, width, height):
    """Shape labels read row-col-rowspan-colspan-text, counted from 1; the text is discarded."""
    data = json.loads(Path(candidate["annotation"]).read_text(encoding="utf-8"))
    if (data.get("imageWidth"), data.get("imageHeight")) != (width, height):
        return None
    grouped = {}
    for shape in data["shapes"]:
        parts = shape["label"].split("-", 4)
        if len(parts) < 4 or not all(part.isdigit() for part in parts[:4]):
            return None
        row, col, rowspan, colspan = (int(part) for part in parts[:4])
        if row < 1 or col < 1:
            return None
        grouped.setdefault(shape.get("group_id"), []).append((row - 1, col - 1, rowspan, colspan, polygon_box(shape["points"])))
    tables = [(polygon_box([point for cell in cells for point in (cell[4][:2], cell[4][2:])]), cells) for cells in grouped.values()]
    return table_target(tables, width, height, candidate["groups"])


def ctdar_candidates(base):
    """Only Track B labels carry cells. Equal names across tracks are different images."""
    root = base / "source"
    sets = [("B1", root / "test_ground_truth/TRACKB1", root / "test/TRACKB1"), ("B2", root / "test_ground_truth/TRACKB2", root / "test/TRACKB2"),
            ("train", root / "training/TRACKB1/ground_truth", root / "training/TRACKB1/ground_truth")]
    for track, labels, images in sets:
        for path in sorted(labels.glob("*.xml")):
            found = sorted(p for p in images.glob(f"{path.stem}.*") if p.suffix.lower() != ".xml")
            if found:
                name = f"{track}:{path.stem}"
                split = "train" if track == "train" else "test" if int(hashlib.sha256(name.encode()).hexdigest(), 16) % 2 else "validation"
                yield {"split": split, "id": name, "image": found[0].relative_to(base).as_posix(), "xml": str(path),
                       "groups": [f"ctdar:image:{name}"], "build": ctdar_target}


def ctdar_target(candidate, width, height):
    tables = []
    for table in ET.parse(candidate["xml"]).getroot().iter("table"):
        cells = []
        for cell in table.findall("cell"):
            top, left = int(cell.get("start-row")), int(cell.get("start-col"))
            cells.append((top, left, int(cell.get("end-row")) - top + 1, int(cell.get("end-col")) - left + 1, coords_box(cell.find("Coords").get("points"))))
        tables.append((coords_box(table.find("Coords").get("points")), cells))
    return table_target(tables, width, height, candidate["groups"])


def read_abp_candidates(base):
    """No official split. Pages of one register book stay together."""
    root = base / "source/READ_ABP_TABLE"
    images = {path.stem: path for path in root.glob("*/img/*") if path.is_file()}
    for folder in ("dataset111/xml", "dataset150/GT_pxml"):
        for path in sorted((root / folder).iterdir()):
            if path.stem in images:
                book = re.sub(r"^\d+_", "", re.sub(r"_\d+$", "", path.stem))
                yield {"split": stable_split(book, 10, 10), "id": path.stem, "image": images[path.stem].relative_to(base).as_posix(), "xml": str(path),
                       "groups": [f"read_abp:book:{book}"], "build": read_abp_target}


def read_abp_target(candidate, width, height):
    tables = []
    for region in ET.parse(candidate["xml"]).getroot().findall(".//{*}TableRegion"):
        cells = [(int(cell.get("row")), int(cell.get("col")), int(cell.get("rowSpan", 1)), int(cell.get("colSpan", 1)), coords_box(cell.find("{*}Coords").get("points")))
                 for cell in region.findall("{*}TableCell")]
        tables.append((coords_box(region.find("{*}Coords").get("points")), cells))
    return table_target(tables, width, height, candidate["groups"])


SOURCES = {"crohme_2023": crohme_candidates, "unimer": unimer_candidates, "mathwriting": mathwriting_candidates,
           "pubtabnet": pubtabnet_candidates, "fintabnet_c": fintabnet_candidates, "hdflowmind": hdflowmind_candidates,
           "hme100k": hme100k_candidates, "ibem": ibem_candidates, "tabrecset": tabrecset_candidates,
           "ctdar_2019": ctdar_candidates, "read_abp": read_abp_candidates}


def build_target(candidate, dataset, base, width, height):
    """Returns (tasks, target, groups), or None to skip the candidate."""
    if "build" in candidate:
        return candidate["build"](candidate, width, height)
    groups = candidate.get("groups")
    if "html" in candidate:
        rows, cols, cells = html_table_cells(candidate["html"]["structure"]["tokens"], candidate["html"]["cells"], width, height)
        cells = cells if len(cells) <= MAX_CELLS else [cell[:4] for cell in cells]
        table = {"id": "t1", "bbox": [0, 0, width, height], "rows": rows, "cols": cols, "cells": cells}
        try:
            validate_table(table, width, height)
        except ValueError:  # A few source tables have spans past the grid; that answer is wrong.
            return None
        return ["tables"], {"tables": [table]}, groups
    if "voc" in candidate:
        parsed = voc_table(candidate["voc"])
        if parsed is None or parsed[2] is None or len(parsed[5]) > MAX_GRID_CELLS or (parsed[0], parsed[1]) != (width, height):
            return None
        cells = parsed[5] if len(parsed[5]) <= MAX_CELLS else [cell[:4] for cell in parsed[5]]
        return ["tables"], {"tables": [{"id": "t1", "bbox": parsed[2], "rows": parsed[3], "cols": parsed[4], "cells": cells}]}, groups
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
    foreign = {sha256(path) for path in base.parent.glob(FOREIGN_IMAGES[dataset]) if path.is_file()} if dataset in FOREIGN_IMAGES else set()
    for split in SPLIT_PRIORITY:
        random.Random(f"{seed}:{dataset}:{split}").shuffle(pools[split])
        if dataset in GROUPED_HELD_OUT:
            pools[split].sort(key=lambda candidate: hashlib.sha256(candidate["groups"][0].encode()).hexdigest())
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
            if digest in foreign:
                skipped["same_image_in_other_source"] += 1
                continue
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
    parser.add_argument("--train", type=int, default=math.inf, help="default keeps every usable train sample")
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
