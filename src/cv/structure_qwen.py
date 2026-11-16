"""Opt-in Qwen3-VL LoRA structure recognition (diagram, formula, table) for the RADU web.

The model sees only the image. Text stays with the OCR engine and is attached afterwards
by position in ``analyze_relations``; nothing here transcribes or removes OCR text.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
ENGINE = "qwen3vl-lora"
BINDING = "configs/structure_qwen_lora.lock.json"
MODES = {
    "opencv": [],
    "qwen-diagram": ["diagram"],
    "qwen-formula": ["formula"],
    "qwen-table": ["table"],
    "qwen-all": ["diagram", "formula", "table"],
}
# Node kinds of the training sources mapped to the editor's flowchart symbols.
SHAPES = {
    "process": "process", "task": "process", "rectangle": "process", "triangle": "process",
    "pool": "process", "lane": "process",
    "terminator": "terminator", "long_oval": "terminator",
    "decision": "decision", "diamonds": "decision", "exclusiveGateway": "decision",
    "parallelGateway": "decision", "eventBasedGateway": "decision",
    "data": "input_output", "parallelogram": "input_output", "dataObject": "input_output",
    "dataStore": "database", "subProcess": "predefined_process",
    "trapezoid": "manual_input", "hexagon": "preparation",
    "connection": "connector", "circle": "connector", "state": "connector", "final_state": "connector",
    "event": "connector", "messageEvent": "connector", "timerEvent": "connector",
}
CONTAINERS = {"pool", "lane"}  # They hold other nodes; arrows do not attach to them.
TIMEOUT_SECONDS = 800


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_binding(root: Path = ROOT) -> dict:
    """Refuse to run when the fixed config or adapters differ from the recorded ones."""
    path = root / BINDING
    if not path.is_file():
        raise RuntimeError("Qwen 구조 인식 binding 파일이 없습니다.")
    binding = json.loads(path.read_text(encoding="utf-8"))
    for name, expected in binding["files"].items():
        target = root / name
        if not target.is_file():
            raise RuntimeError(f"Qwen 구조 인식 구성 파일이 없습니다: {name}")
        if _sha256(target) != expected:
            raise RuntimeError(f"Qwen 구조 인식 모델 SHA-256 불일치: {name}. 다른 모델로 전환하지 않았습니다.")
    return binding


def cuda_device(device: str | None) -> str:
    if not device or not re.fullmatch(r"gpu(?::\d+)?", device):
        raise ValueError("Qwen 구조 인식은 GPU 장치(gpu:0 등)를 선택해 주세요.")
    return "cuda:" + (device.partition(":")[2] or "0")


def salvage(text: str, keys: list[str]) -> tuple[dict[str, list], bool]:
    """Parse the answer; if the JSON is cut off, keep every complete item of each list."""
    try:
        data = json.loads(text)
    except ValueError:
        data = None
    if isinstance(data, dict):
        return {key: data[key] if isinstance(data.get(key), list) else [] for key in keys}, True
    decoder = json.JSONDecoder()
    found = {}
    for key in keys:
        items = []
        start = text.find(f'"{key}"')
        position = text.find("[", start) if start >= 0 else -1
        if position >= 0:
            position += 1
            while True:
                while position < len(text) and text[position] in " \t\r\n,":
                    position += 1
                try:
                    item, position = decoder.raw_decode(text, position)
                except ValueError:
                    break
                items.append(item)
        found[key] = items
    return found, False


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _box(box, width: int, height: int):
    """Clamp to the image; None when nothing usable is left."""
    if not (isinstance(box, list) and len(box) == 4 and all(_number(v) for v in box)):
        return None
    x1, x2 = sorted((max(0.0, min(width, float(box[0]))), max(0.0, min(width, float(box[2])))))
    y1, y2 = sorted((max(0.0, min(height, float(box[1]))), max(0.0, min(height, float(box[3])))))
    if x2 - x1 < 1:  # A straight vertical or horizontal arrow has a flat box.
        x1 = max(0.0, min(x1, width - 1.0)); x2 = x1 + 1
    if y2 - y1 < 1:
        y1 = max(0.0, min(y1, height - 1.0)); y2 = y1 + 1
    return [x1, y1, x2, y2] if x2 <= width and y2 <= height else None


def _point(point, width: int, height: int):
    if not (isinstance(point, list) and len(point) == 2 and all(_number(v) for v in point)):
        return None
    return [max(0.0, min(width, float(point[0]))), max(0.0, min(height, float(point[1])))]


def build_document(raw: dict, width: int, height: int) -> tuple[dict, dict]:
    """Turn raw group answers into a RADU document and {arrow id: (source, target)} hints."""
    elements, warnings, tables, summary = [], [], [], {}
    hints: dict[str, tuple[str, str]] = {}
    skipped = 0
    for name, group in raw["groups"].items():
        parsed, valid = salvage(group["text"], group["tasks"])
        summary[name] = {key: group[key] for key in ("adapter", "generated_tokens", "truncated", "seconds")}
        summary[name]["valid_json"] = valid
        if not valid or group["truncated"]:
            warnings.append({"code": "structure_truncated", "message": f"Qwen 구조 인식({name}) 출력이 완전한 JSON이 아닙니다. 읽을 수 있는 항목만 배치했습니다."})
        nodes, arrows = {}, {}
        for item in parsed.get("nodes", []):
            box = _box(item.get("bbox"), width, height) if isinstance(item, dict) else None
            if box is None:
                skipped += 1
                continue
            kind = str(item.get("kind", ""))
            element = {"id": f"qwen-node-{len(nodes) + 1:04d}", "type": "Box", "bbox": box,
                       "shape": SHAPES.get(kind, "process"), "semantic_type": kind, "confidence": None, "source": ENGINE}
            if kind in CONTAINERS:
                element["flow_node"] = False
            if isinstance(item.get("id"), str):
                nodes.setdefault(item["id"], element)
            elements.append(element)
        for item in parsed.get("arrows", []):
            if not isinstance(item, dict):
                skipped += 1
                continue
            start, end = _point(item.get("start"), width, height), _point(item.get("end"), width, height)
            box = _box(item.get("bbox"), width, height)
            if box is None and start and end:
                box = _box([min(start[0], end[0]), min(start[1], end[1]), max(start[0], end[0]), max(start[1], end[1])], width, height)
            if box is None:
                skipped += 1
                continue
            element = {"id": f"qwen-arrow-{len(arrows) + 1:04d}", "type": "Arrow", "bbox": box, "confidence": None, "source": ENGINE}
            if start and end:
                element.update(start=start, end=end, arrowhead_detected=True)
            if isinstance(item.get("id"), str):
                arrows.setdefault(item["id"], element)
            elements.append(element)
        for item in parsed.get("relations", []):
            if not isinstance(item, dict):
                continue
            arrow, source, target = arrows.get(item.get("arrow_id")), nodes.get(item.get("source")), nodes.get(item.get("target"))
            if arrow and source and target and arrow["id"] not in hints and "flow_node" not in source and "flow_node" not in target:
                hints[arrow["id"]] = (source["id"], target["id"])
        count = sum(e.get("semantic_type") == "formula" for e in elements)
        for item in parsed.get("formulas", []):
            box = _box(item.get("bbox"), width, height) if isinstance(item, dict) else None
            if box is None or not isinstance(item.get("latex"), str) or not item["latex"].strip():
                skipped += 1
                continue
            count += 1
            elements.append({"id": f"qwen-formula-{count:04d}", "type": "Text", "bbox": box, "text": item["latex"],
                             "original_text": item["latex"], "latex": item["latex"], "semantic_type": "formula",
                             "confidence": None, "source": ENGINE})
        for item in parsed.get("tables", []):
            box = _box(item.get("bbox"), width, height) if isinstance(item, dict) else None
            if box is None:
                skipped += 1
                continue
            table_id = f"qwen-table-{len(tables) + 1:02d}"
            cells = [cell for cell in item.get("cells", []) if isinstance(cell, list) and len(cell) in (4, 8) and all(type(v) is int for v in cell)] if isinstance(item.get("cells"), list) else []
            placed = 0
            for cell in cells:
                cell_box = _box(cell[4:], width, height) if len(cell) == 8 else None
                if cell_box is None:
                    continue
                placed += 1
                elements.append({"id": f"{table_id}-cell-{placed:04d}", "type": "Box", "shape": "process", "bbox": cell_box,
                                 "semantic_type": "table-cell", "table_id": table_id, "row": cell[0], "col": cell[1],
                                 "rowspan": cell[2], "colspan": cell[3], "flow_node": False, "confidence": None, "source": ENGINE})
            if not placed:  # No cell boxes: keep the table region itself so it is visible and editable.
                elements.append({"id": table_id, "type": "Box", "shape": "process", "bbox": box, "semantic_type": "table",
                                 "flow_node": False, "confidence": None, "source": ENGINE})
            tables.append({"id": table_id, "bbox": box, "rows": item.get("rows"), "cols": item.get("cols"),
                           "cells": [cell[:4] for cell in cells], "placed_cells": placed})
    if skipped:
        warnings.append({"code": "structure_invalid_items", "message": f"Qwen 구조 인식 항목 {skipped}개는 좌표가 올바르지 않아 배치하지 않았습니다."})
    warnings.insert(0, {"code": "structure_model", "message": "Qwen3-VL LoRA 구조 인식 결과입니다. 도형·화살표·수식·표를 원본과 대조해 확인하세요."})
    document = {
        "version": "1.0", "image": {"width": width, "height": height}, "elements": elements, "relations": [],
        "warnings": warnings,
        "structure": {"engine": ENGINE, "tables": tables, "model_relations": len(hints),
                      "metadata": {"groups": summary, "model": raw.get("model"), "model_revision": raw.get("model_revision"),
                                   "device": raw.get("device"), "load_seconds": raw.get("load_seconds"),
                                   "peak_gpu_bytes": raw.get("peak_gpu_bytes")}},
    }
    return document, hints


def run_worker(image_path: Path, groups: list[str], cuda: str, workdir: Path) -> dict:
    python = Path(os.environ.get("RADU_STRUCTURE_PYTHON", str(ROOT / ".cache/venvs/train/bin/python")))
    if not python.is_file():
        raise RuntimeError("Qwen 구조 인식 실행 환경이 없습니다. RADU_STRUCTURE_PYTHON 설정을 확인해 주세요.")
    output = workdir / "structure.json"
    command = [str(python), "-m", "src.cv.structure_qwen_worker", str(image_path), str(output),
               "--groups", ",".join(groups), "--device", cuda]
    env = os.environ.copy()
    env.update(HF_HOME=os.environ.get("RADU_STRUCTURE_HF_HOME", str(ROOT / ".cache/huggingface")),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    log_path = workdir / "structure.log"
    with log_path.open("w") as log:
        try:
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as error:
            raise TimeoutError("Qwen 구조 인식 시간이 초과됐습니다.") from error
    if not output.is_file():
        raise RuntimeError("Qwen 구조 인식 worker가 결과를 반환하지 않았습니다: " + log_path.read_text(errors="replace")[-1600:])
    payload = json.loads(output.read_text(encoding="utf-8"))
    if completed.returncode != 0 or payload.get("error"):
        raise RuntimeError(f"Qwen 구조 인식 실패: {payload.get('error', 'worker failed')}")
    return payload


def analyze_structure(image, mode: str, device: str | None) -> tuple[dict, dict]:
    """Run the fixed adapters on a BGR or grayscale image; returns (document, relation hints)."""
    from PIL import Image

    groups = MODES[mode]
    if not groups:
        raise ValueError("OpenCV 모드에는 Qwen 구조 인식을 실행하지 않습니다.")
    cuda = cuda_device(device)
    binding = verify_binding()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="radu-structure-") as directory:
        workdir = Path(directory)
        image_path = workdir / "input.png"
        if image.ndim == 3 and image.shape[2] == 3:
            Image.fromarray(image[:, :, ::-1]).save(image_path)
        elif image.ndim == 2:
            Image.fromarray(image).convert("RGB").save(image_path)
        else:
            raise ValueError("Qwen 구조 인식에는 BGR 또는 회색조 이미지가 필요합니다.")
        raw = run_worker(image_path, groups, cuda, workdir)
    document, hints = build_document(raw, int(image.shape[1]), int(image.shape[0]))
    metadata = document["structure"]["metadata"]
    metadata["total_seconds"] = time.perf_counter() - started
    metadata["adapter_sha256"] = {
        binding["groups"][name]["adapter"]: binding["files"][binding["groups"][name]["adapter"] + "/adapter_model.safetensors"]
        for name in groups
    }
    document["structure"]["mode"] = mode
    return document, hints


_RELATION_WARNINGS = {"ambiguous_text_attachment", "ambiguous_arrow_endpoint", "isolated_element", "cycle_review", "manual_arrow_unlinked"}


def relate(document: dict, hints: dict[str, tuple[str, str]]) -> dict:
    """Relations for a model document, built on the unchanged ``analyze_relations``.

    Arrows connect only flow nodes (not table cells or swimlanes), and an arrow whose
    source and target the model reported uses them instead of the geometric guess.
    Text is attached to every kind of box. A later "관계 분석 실행" in the editor is
    geometric only.
    """
    from src.cv.relation import analyze_relations

    elements = document.get("elements", [])
    by_id = {element["id"]: element for element in elements}
    flow_ids = {e["id"] for e in elements if e.get("type") == "Box" and e.get("flow_node", True)}
    hints = {arrow: pair for arrow, pair in hints.items() if arrow in by_id and set(pair) <= flow_ids}
    flow_elements = []
    for element in elements:
        if element["id"] in hints:
            # Put the endpoints inside the reported boxes so geometry agrees with the model.
            (sx1, sy1, sx2, sy2), (tx1, ty1, tx2, ty2) = (by_id[key]["bbox"] for key in hints[element["id"]])
            element = dict(element, start=[(sx1 + sx2) / 2, (sy1 + sy2) / 2], end=[(tx1 + tx2) / 2, (ty1 + ty2) / 2], arrowhead_detected=True)
        if element["id"] in flow_ids or element.get("type") == "Arrow":
            flow_elements.append(element)
    flow = analyze_relations(dict(document, elements=flow_elements))
    text = analyze_relations(dict(document, elements=[e for e in elements if e.get("type") in {"Box", "Text"}]))
    relations = []
    for relation in flow["relations"]:
        if relation.get("arrow_id") in hints:
            source, target = hints[relation["arrow_id"]]
            relation.update(source=source, target=target, endpoint_method="model_relation", status="inferred")
        relations.append(relation)
    relations += [relation for relation in text["relations"] if relation["type"] != "flows_to"]
    warnings = [w for w in document.get("warnings", []) if not isinstance(w, dict) or w.get("code") not in _RELATION_WARNINGS]
    warnings += [w for w in flow["warnings"] if isinstance(w, dict) and w.get("code") in _RELATION_WARNINGS
                 and not (w["code"] == "ambiguous_arrow_endpoint" and w.get("element_id") in hints)]
    warnings += [w for w in text["warnings"] if isinstance(w, dict) and w.get("code") == "ambiguous_text_attachment"]
    return dict(document, relations=relations, warnings=warnings)
