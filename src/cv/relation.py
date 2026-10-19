"""Geometry-first flowchart relation inference with explicit review warnings."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


def _center(box: list[float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _distance_to_box(point: tuple[float, float], box: list[float]) -> float:
    x, y = point
    dx = max(box[0] - x, 0, x - box[2])
    dy = max(box[1] - y, 0, y - box[3])
    return math.hypot(dx, dy)


def _contains(outer: list[float], inner: list[float]) -> bool:
    cx, cy = _center(inner)
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def _arrow_geometry(arrow: dict[str, Any]) -> tuple[tuple[float, float], tuple[float, float], str]:
    if arrow.get("start") is not None and arrow.get("end") is not None:
        method = "explicit" if arrow.get("arrowhead_detected", True) else "line_endpoints_no_head"
        def point(value: Any) -> tuple[float, float]:
            if isinstance(value, dict):
                return value["x"], value["y"]
            return value[0], value[1]

        return point(arrow["start"]), point(arrow["end"]), method
    x1, y1, x2, y2 = arrow["bbox"]
    # A detector should provide start/end when it can identify the arrow head.
    return (x1, y1), (x2, y2), "bbox_approximation"


def analyze_relations(document: dict[str, Any]) -> dict[str, Any]:
    elements = document.get("elements", [])
    boxes = [e for e in elements if e.get("type") == "Box"]
    arrows = [e for e in elements if e.get("type") == "Arrow"]
    texts = [e for e in elements if e.get("type") == "Text"]
    relations = []
    generated_warning_codes = {
        "ambiguous_text_attachment",
        "ambiguous_arrow_endpoint",
        "isolated_element",
        "cycle_review",
        "manual_arrow_unlinked",
    }
    warnings = [
        warning for warning in document.get("warnings", [])
        if not isinstance(warning, dict)
        or warning.get("code") not in generated_warning_codes
    ]
    existing_relations = document.get("relations", [])
    degree: dict[str, int] = defaultdict(int)
    graph: dict[str, list[str]] = defaultdict(list)
    for text in texts:
        candidates = [(0 if _contains(box["bbox"], text["bbox"]) else 1,
                       _distance(_center(box["bbox"]), _center(text["bbox"])), box) for box in boxes]
        if candidates:
            inside = [c for c in candidates if c[0] == 0]
            if inside:
                target = min(inside, key=lambda x: x[1])[2]
                relations.append({"id": f"contains-{target['id']}-{text['id']}", "type": "contains", "source": target["id"], "target": text["id"], "status": "inferred"})
            else:
                target = min(candidates, key=lambda x: x[1])[2]
                diagonal = math.hypot(target["bbox"][2] - target["bbox"][0], target["bbox"][3] - target["bbox"][1])
                if _distance(_center(target["bbox"]), _center(text["bbox"])) < diagonal * .8:
                    relations.append({"id": f"near-{target['id']}-{text['id']}", "type": "near_text", "source": target["id"], "target": text["id"], "status": "review"})
                    warnings.append({"code": "ambiguous_text_attachment", "element_id": text["id"], "message": f"텍스트 {text['id']}의 도형 포함 관계를 확인하세요."})
    for arrow in arrows:
        start, end, method = _arrow_geometry(arrow)
        starts = sorted(((_distance_to_box(start, b["bbox"]), b) for b in boxes), key=lambda p: p[0])
        ends = sorted(((_distance_to_box(end, b["bbox"]), b) for b in boxes), key=lambda p: p[0])
        src = starts[0][1] if starts else None
        dst = ends[0][1] if ends else None
        status = "inferred"
        if method != "explicit":
            status = "review"
        if src and dst and src["id"] == dst["id"]:
            status = "review"
        if not src or not dst or len(starts) > 1 and abs(starts[0][0] - starts[1][0]) < max(8, starts[0][0] * .12) or len(ends) > 1 and abs(ends[0][0] - ends[1][0]) < max(8, ends[0][0] * .12):
            status = "review"
        relation = {"id": f"edge-{arrow['id']}", "type": "flows_to", "source": src["id"] if src else None,
                    "target": dst["id"] if dst else None, "arrow_id": arrow["id"], "direction": "unknown",
                    "endpoint_method": method, "status": status}
        if src and dst:
            sx, sy = _center(src["bbox"]); tx, ty = _center(dst["bbox"])
            relation["direction"] = "right" if abs(tx - sx) >= abs(ty - sy) and tx >= sx else "left" if abs(tx - sx) >= abs(ty - sy) else "down" if ty >= sy else "up"
            degree[src["id"]] += 1; degree[dst["id"]] += 1
            graph[src["id"]].append(dst["id"])
        previous = next(
            (
                item for item in existing_relations
                if item.get("type") == "flows_to" and item.get("arrow_id") == arrow["id"]
            ),
            None,
        )
        if previous is not None:
            for key, value in previous.items():
                if key not in {
                    "id", "type", "source", "target", "arrow_id", "direction",
                    "endpoint_method", "status",
                }:
                    relation[key] = value
            relation["id"] = previous["id"]
        relations.append(relation)
        if status == "review":
            warnings.append({"code": "ambiguous_arrow_endpoint", "element_id": arrow["id"], "message": f"화살표 {arrow['id']}의 연결 대상을 확인하세요."})
    for box in boxes:
        if degree[box["id"]] == 0:
            warnings.append({"code": "isolated_element", "element_id": box["id"], "message": f"도형 {box['id']}에 연결된 화살표가 없습니다. 원본의 고립 요소인지 확인하세요."})
    # Cycles are flagged for human review; they may be intentional in the source.
    # Use iterative DFS so long user documents cannot exceed Python's recursion
    # limit. Colors are 1 while a node is on the active path and 2 when done.
    colors: dict[str, int] = {}
    cycle_nodes: set[str] = set()
    for root in list(graph):
        if colors.get(root) == 2:
            continue
        path = [root]
        path_index = {root: 0}
        colors[root] = 1
        stack = [(root, iter(graph.get(root, ())))]
        while stack:
            node, neighbors = stack[-1]
            try:
                nxt = next(neighbors)
            except StopIteration:
                stack.pop()
                colors[node] = 2
                path_index.pop(node, None)
                path.pop()
                continue
            color = colors.get(nxt, 0)
            if color == 1:
                cycle_nodes.update(path[path_index[nxt]:])
            elif color == 0:
                colors[nxt] = 1
                path_index[nxt] = len(path)
                path.append(nxt)
                stack.append((nxt, iter(graph.get(nxt, ()))))
    for node in sorted(cycle_nodes):
        warnings.append({"code": "cycle_review", "element_id": node, "message": f"순환 연결에 포함된 {node}를 원본과 대조하세요."})
    for relation in relations:
        if relation["type"] == "flows_to":
            continue
        previous = next(
            (
                item for item in existing_relations
                if item.get("type") == relation["type"]
                and item.get("source") == relation["source"]
                and item.get("target") == relation["target"]
            ),
            None,
        )
        if previous is not None:
            relation["id"] = previous["id"]
            for key, value in previous.items():
                if key not in {"id", "type", "source", "target", "status"}:
                    relation[key] = value
    result = dict(document)
    result["relations"] = relations
    result["warnings"] = warnings
    return result
