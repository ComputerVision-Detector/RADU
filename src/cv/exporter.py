"""JSON and vector/raster export for the editable RADU document schema."""
from __future__ import annotations

import json
import xml.sax.saxutils as sax
from pathlib import Path
from typing import Any


def to_svg(document: dict[str, Any]) -> str:
    width = int(document.get("image", {}).get("width", 1000)); height = int(document.get("image", {}).get("height", 800))
    elements = {e["id"]: e for e in document.get("elements", [])}
    chunks = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
              '<rect width="100%" height="100%" fill="white"/>', '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#334155"/></marker></defs>']
    for rel in document.get("relations", []):
        if rel.get("type") != "flows_to" or rel.get("source") not in elements or rel.get("target") not in elements: continue
        a = elements[rel["source"]]["bbox"]; b = elements[rel["target"]]["bbox"]
        ax, ay = (a[0]+a[2])/2, (a[1]+a[3])/2; bx, by = (b[0]+b[2])/2, (b[1]+b[3])/2
        chunks.append(f'<path d="M {ax:g} {ay:g} L {bx:g} {by:g}" fill="none" stroke="#334155" stroke-width="2" marker-end="url(#arrow)"/>')
    for e in document.get("elements", []):
        x1,y1,x2,y2=e["bbox"]; w=x2-x1; h=y2-y1; kind=e.get("type")
        if kind == "Box":
            chunks.append(f'<rect x="{x1}" y="{y1}" width="{w}" height="{h}" rx="8" fill="white" stroke="#2563eb" stroke-width="2"/>')
        elif kind == "Text":
            text=sax.escape(e.get("text", "")); chunks.append(f'<text x="{x1}" y="{y1+h*.75:g}" font-family="sans-serif" font-size="16" fill="#111827">{text}</text>')
        elif kind == "Arrow":
            chunks.append(f'<rect x="{x1}" y="{y1}" width="{w}" height="{h}" fill="transparent" stroke="#16a34a" stroke-dasharray="4 4"/>')
    chunks.append('</svg>')
    return '\n'.join(chunks)


def export_document(document: dict[str, Any], output_path: str | Path, fmt: str | None = None) -> Path:
    path=Path(output_path); kind=(fmt or path.suffix.lstrip('.')).lower(); path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "json": path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    elif kind == "svg": path.write_text(to_svg(document), encoding="utf-8")
    elif kind in {"png", "pdf"}:
        svg=to_svg(document)
        try: import cairosvg
        except ImportError as exc: raise RuntimeError("Install cairosvg to export PNG/PDF") from exc
        if kind == "png": cairosvg.svg2png(bytestring=svg.encode(), write_to=str(path))
        else: cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(path))
    else: raise ValueError(f"Unsupported export format: {kind}")
    return path
