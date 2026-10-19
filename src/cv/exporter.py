"""JSON and vector/raster export for the editable RADU document schema."""
from __future__ import annotations

import io
import json
import re
import math
from math import ceil, floor
import xml.sax.saxutils as sax
from pathlib import Path
from typing import Any


def _number(value: Any) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Coordinates must be finite numbers")
    return f"{number:g}"


def _point(value: Any) -> tuple[float, float]:
    if isinstance(value, dict):
        x, y = value["x"], value["y"]
    elif isinstance(value, (list, tuple)) and len(value) >= 2:
        x, y = value[0], value[1]
    else:
        raise ValueError("Arrow endpoints must be coordinate pairs")
    return float(x), float(y)


def _box_edge(element: dict, toward: tuple[float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = map(float, element["bbox"])
    width, height = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = toward[0] - cx, toward[1] - cy
    if not dx and not dy:
        dy = 1
    shape = element.get("shape")
    if shape == "decision":
        scale = .5 / (abs(dx) / width + abs(dy) / height)
    elif shape == "connector":
        scale = 1 / math.sqrt((dx / (width / 2)) ** 2 + (dy / (height / 2)) ** 2)
    else:
        scale = .5 / max(abs(dx) / width, abs(dy) / height)
    return cx + dx * scale, cy + dy * scale


def _box_geometry(shape: str, x: float, y: float, width: float, height: float) -> str:
    n = _number
    if shape == "decision":
        return f'<polygon points="{n(x+width/2)},{n(y)} {n(x+width)},{n(y+height/2)} {n(x+width/2)},{n(y+height)} {n(x)},{n(y+height/2)}"/>'
    if shape == "terminator":
        return f'<rect x="{n(x)}" y="{n(y)}" width="{n(width)}" height="{n(height)}" rx="{n(height/2)}"/>'
    if shape == "input_output":
        skew = min(width * .16, height * .35)
        return f'<polygon points="{n(x+skew)},{n(y)} {n(x+width)},{n(y)} {n(x+width-skew)},{n(y+height)} {n(x)},{n(y+height)}"/>'
    if shape == "document":
        return f'<path d="M{x:g} {y:g}H{x+width:g}V{y+height*.78:g} Q{x+width*.75:g} {y+height*1.08:g} {x+width/2:g} {y+height*.86:g} Q{x+width*.25:g} {y+height*.66:g} {x:g} {y+height*.9:g}Z"/>'
    if shape == "database":
        return f'<path d="M{x:g} {y+height*.14:g} C{x:g} {y-height*.08:g} {x+width:g} {y-height*.08:g} {x+width:g} {y+height*.14:g} V{y+height*.84:g} C{x+width:g} {y+height*1.08:g} {x:g} {y+height*1.08:g} {x:g} {y+height*.84:g}Z M{x:g} {y+height*.14:g} C{x:g} {y+height*.38:g} {x+width:g} {y+height*.38:g} {x+width:g} {y+height*.14:g} M{x:g} {y+height*.5:g} C{x:g} {y+height*.74:g} {x+width:g} {y+height*.74:g} {x+width:g} {y+height*.5:g}"/>'
    if shape == "predefined_process":
        return f'<rect x="{n(x)}" y="{n(y)}" width="{n(width)}" height="{n(height)}" rx="3"/><path d="M{n(x+width*.16)} {n(y)}V{n(y+height)} M{n(x+width*.84)} {n(y)}V{n(y+height)}"/>'
    if shape == "manual_input":
        return f'<polygon points="{n(x)},{n(y+height*.2)} {n(x+width)},{n(y)} {n(x+width)},{n(y+height)} {n(x)},{n(y+height)}"/>'
    if shape == "preparation":
        return f'<polygon points="{n(x+width*.2)},{n(y)} {n(x+width*.8)},{n(y)} {n(x+width)},{n(y+height/2)} {n(x+width*.8)},{n(y+height)} {n(x+width*.2)},{n(y+height)} {n(x)},{n(y+height/2)}"/>'
    if shape == "offpage_connector":
        return f'<polygon points="{n(x)},{n(y)} {n(x+width)},{n(y)} {n(x+width)},{n(y+height*.6)} {n(x+width/2)},{n(y+height)} {n(x)},{n(y+height*.6)}"/>'
    if shape == "connector":
        return f'<ellipse cx="{n(x+width/2)}" cy="{n(y+height/2)}" rx="{n(width/2)}" ry="{n(height/2)}"/>'
    if shape == "delay":
        return f'<path d="M{n(x)} {n(y)}H{n(x+width/2)} A{n(height/2)} {n(height/2)} 0 0 1 {n(x+width/2)} {n(y+height)} H{n(x)}Z"/>'
    return f'<rect x="{n(x)}" y="{n(y)}" width="{n(width)}" height="{n(height)}" rx="{n(min(10,height*.12))}"/>'


def _jump_geometry(sx: float, sy: float, ex: float, ey: float,
                   previous: list[dict], mode: str) -> str:
    dx, dy = ex - sx, ey - sy
    length = math.hypot(dx, dy)
    if mode not in {"arc", "gap", "sharp"} or not length:
        return f"M {_number(sx)} {_number(sy)} L {_number(ex)} {_number(ey)}"
    crossings = []
    for other in previous:
        if other.get("route", "straight") != "straight" or not other.get("start") or not other.get("end"):
            continue
        ox, oy = _point(other["start"]); tx, ty = _point(other["end"])
        vx, vy = tx - ox, ty - oy
        det = dx * vy - dy * vx
        if abs(det) < 1e-6:
            continue
        rx, ry = ox - sx, oy - sy
        t = (rx * vy - ry * vx) / det
        u = (rx * dy - ry * dx) / det
        if .05 < t < .95 and .05 < u < .95:
            crossings.append(t)
    path = f"M {_number(sx)} {_number(sy)}"
    last = 0.
    def at(value):
        return sx + dx * value, sy + dy * value
    for crossing in sorted(crossings):
        pad = min(8 / length, (crossing - last) / 3, (1 - crossing) / 3)
        before, after = crossing - pad, crossing + pad
        if before <= last:
            continue
        lx, ly = at(before); rx, ry = at(after); mx, my = at(crossing)
        nx, ny = -dy / length, dx / length
        path += f" L {_number(lx)} {_number(ly)}"
        if mode == "gap":
            path += f" M {_number(rx)} {_number(ry)}"
        elif mode == "sharp":
            path += f" L {_number(mx + nx*10)} {_number(my + ny*10)} L {_number(rx)} {_number(ry)}"
        else:
            path += f" Q {_number(mx + nx*13)} {_number(my + ny*13)} {_number(rx)} {_number(ry)}"
        last = after
    return path + f" L {_number(ex)} {_number(ey)}"


def _arrow_markers(style: dict, index: int, color: str) -> tuple[str, str]:
    def marker(kind: str, position: str) -> str:
        if kind == "none":
            return ""
        shape = "M1 5L5 1L9 5L5 9Z" if kind == "diamond" else "M1 1L9 5L1 9Z"
        start = position == "start"
        ref = 2 if start else 8
        orient = "auto-start-reverse" if start else "auto"
        return f'<marker id="radu-{index}-{position}" markerWidth="10" markerHeight="10" refX="{ref}" refY="5" orient="{orient}" markerUnits="userSpaceOnUse"><path d="{shape}" fill="{sax.escape(color)}"/></marker>'
    first = style.get("marker_start", "none")
    last = style.get("marker_end", "arrow")
    attrs = (f' marker-start="url(#radu-{index}-start)"' if first != "none" else "") + (f' marker-end="url(#radu-{index}-end)"' if last != "none" else "")
    return marker(first, "start") + marker(last, "end"), attrs


def _arrow_geometry(start: Any, end: Any, route: str = "straight", previous: list[dict] | None = None, jump: str = "none") -> tuple[float, float, float, float, str]:
    sx, sy = _point(start); ex, ey = _point(end)
    if route == "orthogonal":
        d = f"M {_number(sx)} {_number(sy)} H {_number((sx+ex)/2)} V {_number(ey)} H {_number(ex)}"
    elif route == "curved":
        d = f"M {_number(sx)} {_number(sy)} C {_number((sx+ex)/2)} {_number(sy)} {_number((sx+ex)/2)} {_number(ey)} {_number(ex)} {_number(ey)}"
    else:
        d = _jump_geometry(sx, sy, ex, ey, previous or [], jump)
    return sx, sy, ex, ey, d


def _lines(text: str, width: float, size: float) -> list[str]:
    limit = max(1, int((width - 12) / (size * .9)))
    result: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph:
            result.append("")
            continue
        words = re.findall(r"\s+|\S+", paragraph)
        current = ""
        for word in words:
            if len(current) + len(word) <= limit:
                current += word
            else:
                if current.strip():
                    result.append(current.rstrip())
                current = ""
                while len(word) > limit:
                    result.append(word[:limit])
                    word = word[limit:]
                current = word.strip() if not word.isspace() else ""
        result.append(current.rstrip())
    return result or [""]


def _text_markup(label: str, style: dict, bbox: tuple[float,float,float,float], centered: bool) -> str:
    x1,y1,x2,y2=bbox
    width,height=x2-x1,y2-y1
    size=float((style.get("font_size") or 18))
    while size>8 and len(_lines(label,width,size))*size*1.12 > max(size,height-8):
        size-=1
    lines=_lines(label,width,size)
    align=style.get("text_align","center" if centered else "left")
    anchor={"left":"start","right":"end"}.get(align,"middle")
    x=x1+(6 if anchor=="start" else width-6 if anchor=="end" else width/2)
    y=y1+height/2-(len(lines)-1)*size*.56
    color=sax.escape(str((style.get("text_color") or "#334057")))
    family=sax.escape(str((style.get("font_family") or "system-ui,sans-serif")), {'"':'&quot;'})
    weight="700" if style.get("bold") else "400"
    italic="italic" if style.get("italic") else "normal"
    tspans="".join(f'<tspan x="{_number(x)}" y="{_number(y+i*size*1.12)}">{sax.escape(line)}</tspan>' for i,line in enumerate(lines))
    return f'<text x="{_number(x)}" y="{_number(y)}" text-anchor="{anchor}" font-family="{family}" font-size="{_number(size)}" font-weight="{weight}" font-style="{italic}" fill="{color}">{tspans}</text>'


def _arrow_label_markup(label: str, sx: float, sy: float, ex: float, ey: float) -> str:
    lines = [line for part in label.split("\n") for line in _lines(part, 400, 16)]
    x, y = (sx + ex) / 2, (sy + ey) / 2 - 7 - (len(lines) - 1) * 9
    spans = "".join(f'<tspan x="{_number(x)}" dy="{0 if i == 0 else 18}">{sax.escape(line)}</tspan>' for i, line in enumerate(lines))
    return f'<text x="{_number(x)}" y="{_number(y)}" text-anchor="middle" font-size="16" fill="#172033">{spans}</text>'


def to_svg(document: dict[str, Any], *, transparent: bool = False) -> str:
    image = document.get("image", {})
    width = int(image.get("width", 1000))
    height = int(image.get("height", 800))
    layers={layer.get("id"):layer for layer in document.get("layers", [])}
    visible=[element for element in document.get("elements", []) if not element.get("hidden") and layers.get(element.get("layer_id"),{}).get("visible",True)]
    elements = {element["id"]: element for element in visible}
    bounds = []
    for element in visible:
        bounds.append(tuple(map(float, element["bbox"])))
        if element.get("type") == "Arrow":
            for endpoint in (element.get("start"), element.get("end")):
                if endpoint is not None:
                    px, py = _point(endpoint)
                    bounds.append((px, py, px, py))
    if bounds:
        margin = 24
        left = max(0, floor(min(box[0] for box in bounds) - margin))
        top = max(0, floor(min(box[1] for box in bounds) - margin))
        right = min(width, ceil(max(box[2] for box in bounds) + margin))
        bottom = min(height, ceil(max(box[3] for box in bounds) + margin))
        if right <= left or bottom <= top:
            left, top, right, bottom = 0, 0, width, height
    else:
        left, top, right, bottom = 0, 0, width, height
    output_width, output_height = right - left, bottom - top
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{output_width}" height="{output_height}" viewBox="{left} {top} {output_width} {output_height}">',
        *( [] if transparent else [f'<rect x="{left}" y="{top}" width="{output_width}" height="{output_height}" fill="{sax.escape(str(document.get("canvas",{}).get("background","#ffffff")))}"/>'] ),
        '<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 Z" fill="#5268e8"/></marker></defs>',
    ]
    rendered_arrows: set[str] = set()
    arrow_index = {element["id"]: index for index, element in enumerate(visible) if element.get("type") == "Arrow"}
    jump_mode = document.get("canvas", {}).get("line_jump", "none")
    for relation in document.get("relations", []):
        if relation.get("type") != "flows_to":
            continue
        arrow = elements.get(relation.get("arrow_id"))
        if arrow and arrow.get("type") == "Arrow":
            if arrow["id"] not in rendered_arrows:
                rendered_arrows.add(arrow["id"])
                start, end = arrow.get("start"), arrow.get("end")
                if start and end:
                    sx, sy, ex, ey, d = _arrow_geometry(start, end, arrow.get("route", "straight"), visible[:arrow_index[arrow["id"]]], jump_mode)
                    arrow_style = arrow.get("style") or {}
                    dash = ' stroke-dasharray="10 7"' if arrow_style.get("line_style") == "dashed" else ' stroke-dasharray="2 7"' if arrow_style.get("line_style") == "dotted" else ""
                    marker_defs, marker_attrs = _arrow_markers(arrow_style, arrow_index[arrow["id"]], str((arrow_style.get("stroke") or "#5268e8")))
                    chunks.append("<defs>" + marker_defs + "</defs>")
                    box = arrow["bbox"]
                    rotation = float(arrow.get("rotation", 0) or 0)
                    arrow_transform = (f' transform="rotate({_number(rotation)} {_number((box[0]+box[2])/2)} {_number((box[1]+box[3])/2)})"' if rotation else "")
                    chunks.append(f'<g{arrow_transform} opacity="{_number(arrow_style.get("opacity", 1))}">')
                    chunks.append(
                        f'<path d="{d}" fill="none" stroke="{sax.escape(str((arrow_style.get("stroke") or "#5268e8")))}" stroke-width="{_number(max(1,min(30,float((arrow_style.get("stroke_width") or 3)))))}"{dash}{marker_attrs}/>'
                    )
                    if arrow.get("text"):
                        chunks.append(_arrow_label_markup(str(arrow["text"]), sx, sy, ex, ey))
                    chunks.append('</g>')
            continue
        source, target = elements.get(relation.get("source")), elements.get(relation.get("target"))
        if not source or not target:
            continue
        a, b = source["bbox"], target["bbox"]
        source_center = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
        target_center = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        ax, ay = _box_edge(source, target_center)
        bx, by = _box_edge(target, source_center)
        chunks.append(
            f'<path d="M {_number(ax)} {_number(ay)} L {_number(bx)} {_number(by)}" fill="none" stroke="#5268e8" stroke-width="3" marker-end="url(#arrow)"/>'
        )
    for element in visible:
        x1, y1, x2, y2 = map(_number, element["bbox"])
        x1f, y1f, x2f, y2f = map(float, element["bbox"])
        box_width, box_height = x2f - x1f, y2f - y1f
        kind = element.get("type")
        style = element.get("style") or {}
        stroke_color = str((style.get("stroke") or "#5268e8"))
        fill_color = str((style.get("fill") or "white"))
        stroke_width = max(1, min(30, float((style.get("stroke_width") or 3))))
        rotation = float(element.get("rotation", 0) or 0)
        transform = (f' transform="rotate({_number(rotation)} {_number(x1f+box_width/2)} {_number(y1f+box_height/2)})"'
                     if rotation else "")
        opacity=max(.1,min(1,float(style.get("opacity",1))))
        if kind == "Image":
            href=sax.escape(element.get("image_data",""), {'"':'&quot;'})
            chunks.append(f'<g{transform} opacity="{_number(opacity)}"><image x="{x1}" y="{y1}" width="{_number(box_width)}" height="{_number(box_height)}" href="{href}" preserveAspectRatio="none"/></g>')
            continue
        if kind == "Stroke":
            points=" ".join(f'{_number(p["x"])},{_number(p["y"])}' for p in element.get("points",[]))
            chunks.append(f'<g{transform} opacity="{_number(opacity)}"><polyline points="{points}" fill="none" stroke="{sax.escape(stroke_color)}" stroke-width="{_number(stroke_width)}" stroke-linecap="round" stroke-linejoin="round"/></g>')
            continue
        if kind == "Box":
            geometry = _box_geometry(element.get("shape", "process"), x1f, y1f, box_width, box_height)
            chunks.append(f'<g{transform} fill="{sax.escape(fill_color)}" stroke="{sax.escape(stroke_color)}" stroke-width="{_number(stroke_width)}" opacity="{_number(opacity)}" stroke-linejoin="round">{geometry}</g>')
            label = str(element.get("text", ""))
            if label:
                chunks.append(f'<g{transform} opacity="{_number(opacity)}">{_text_markup(label,style,(x1f,y1f,x2f,y2f),True)}</g>')
        elif kind == "Text":
            label = str(element.get("text", ""))
            if label:
                chunks.append(f'<g{transform} opacity="{_number(opacity)}">{_text_markup(label,style,(x1f,y1f,x2f,y2f),False)}</g>')
        elif kind == "Arrow" and element["id"] not in rendered_arrows:
            start, end = element.get("start"), element.get("end")
            if start and end:
                sx, sy, ex, ey, d = _arrow_geometry(start, end, element.get("route", "straight"), visible[:arrow_index[element["id"]]], jump_mode)
                arrow_style = element.get("style") or {}
                dash = ' stroke-dasharray="10 7"' if arrow_style.get("line_style") == "dashed" else ' stroke-dasharray="2 7"' if arrow_style.get("line_style") == "dotted" else ""
                marker_defs, marker_attrs = _arrow_markers(arrow_style, arrow_index[element["id"]], str((arrow_style.get("stroke") or "#188c70")))
                chunks.append("<defs>" + marker_defs + "</defs>")
                chunks.append(f'<g{transform} opacity="{_number(opacity)}">')
                chunks.append(f'<path d="{d}" fill="none" stroke="{sax.escape(str((arrow_style.get("stroke") or "#188c70")))}" stroke-width="{_number(max(1,min(30,float((arrow_style.get("stroke_width") or 3)))))}"{dash}{marker_attrs}/>')
                if element.get("text"):
                    chunks.append(_arrow_label_markup(str(element["text"]), sx, sy, ex, ey))
                chunks.append('</g>')
            else:
                chunks.append(f'<rect x="{x1}" y="{y1}" width="{_number(box_width)}" height="{_number(box_height)}" fill="none" stroke="#188c70" stroke-dasharray="5 4"/>')
    chunks.append("</svg>")
    return "\n".join(chunks)


def export_document(document: dict[str, Any], output_path: str | Path, fmt: str | None = None, *, scale: float = 1, transparent: bool = False, all_pages: bool = False) -> Path:
    if not 1 <= scale <= 4:
        raise ValueError("PNG 해상도 배율은 1~4입니다")
    path = Path(output_path)
    kind = (fmt or path.suffix.lstrip(".")).lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "json":
        path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    elif kind == "svg":
        path.write_text(to_svg(document, transparent=transparent), encoding="utf-8")
    elif kind in {"png", "pdf"}:
        svg = to_svg(document, transparent=transparent)
        try:
            import cairosvg
        except ImportError as exc:
            raise RuntimeError("PNG/PDF 내보내기에는 cairosvg가 필요합니다.") from exc
        if kind == "png":
            cairosvg.svg2png(bytestring=svg.encode(), write_to=str(path), scale=scale)
        elif all_pages and document.get("pages"):
            from PIL import Image
            page_documents = [document, *document["pages"]]
            raster_pages = []
            for page in page_documents:
                page_svg = to_svg(page, transparent=False)
                bitmap = cairosvg.svg2png(bytestring=page_svg.encode(), scale=2)
                image = Image.open(io.BytesIO(bitmap)).convert("RGB")
                raster_pages.append(image)
            raster_pages[0].save(path, "PDF", save_all=True,
                                 append_images=raster_pages[1:], resolution=144.0)
            for image in raster_pages:
                image.close()
        else:
            cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(path))
    else:
        raise ValueError(f"Unsupported export format: {kind}")
    return path
