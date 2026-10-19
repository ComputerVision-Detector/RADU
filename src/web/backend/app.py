"""RADU web API for flowchart analysis, review, editing, and export."""
from __future__ import annotations

import asyncio
import base64
import binascii
import email.policy
import json
import io
import math
import mimetypes
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import xml.etree.ElementTree as ElementTree
import urllib.parse
from datetime import datetime, timezone
from email.parser import BytesParser
from pathlib import Path
from uuid import uuid4
from defusedxml.common import DefusedXmlException

import cv2
import numpy as np
from defusedxml import ElementTree as SafeElementTree
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask
from PIL import Image, ImageOps, UnidentifiedImageError

try:
    import pillow_avif  # noqa: F401
except ImportError:
    pass

try:
    import pillow_heif
except ImportError:
    pass
else:
    pillow_heif.register_heif_opener()

from src.cv.detector import analyze_image
from src.cv.exporter import export_document
from src.cv.ocr_engine import recognize
from src.cv.preprocess import preprocess_image
from src.cv.relation import analyze_relations

ROOT = Path(__file__).resolve().parents[3]
FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
OUT = ROOT / "data" / "processed" / "sessions"
OUT.mkdir(parents=True, exist_ok=True)
VERSIONS = OUT / "versions"
VERSIONS.mkdir(parents=True, exist_ok=True)
EXPORTS = ROOT / "data" / "processed" / "exports"
EXPORTS.mkdir(parents=True, exist_ok=True)
MAX_EMBEDDED_IMAGE_PIXELS = 40_000_000
ANALYSIS_TIMEOUT_SECONDS = 900
SESSION_ID_RE = re.compile(r"^[a-f0-9]{32}$")
SUPPORTED_ENGINES = {"none", "paddleocrv6", "paddleocrvl16", "clova"}
INFERENCE_LOCK = threading.Lock()
IMAGE_DECODE_LOCK = threading.Lock()
Image.MAX_IMAGE_PIXELS = None

app = FastAPI(
    title="RADU",
    version="0.2.0",
    description="Relation-aware handwritten flowchart restoration and review",
)
app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")


@app.get("/", response_class=HTMLResponse)
async def home() -> str:
    return (FRONTEND / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok", "version": app.version}


@app.post("/api/analyze-relations")
async def reanalyze_relations(document: dict) -> dict:
    valid = _validate_document(document)
    return analyze_relations(valid)


def _session_path(session_id: str) -> Path:
    if not SESSION_ID_RE.fullmatch(session_id):
        raise HTTPException(400, "Invalid session ID")
    return OUT / f"{session_id}.json"


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".radu-", delete=False) as stream:
            temp_name = stream.name
            stream.write(payload)
            stream.flush()
        Path(temp_name).replace(path)
    finally:
        if temp_name and Path(temp_name).exists():
            Path(temp_name).unlink(missing_ok=True)


def _normalize_upload_image(payload: bytes) -> tuple[bytes, str]:
    if payload[:16].lstrip().startswith(b"%PDF-"):
        raise HTTPException(415, "PDF 입력은 아직 지원하지 않습니다. PDF 페이지를 SVG 또는 래스터 이미지로 내보내 주세요.")
    stripped = payload[:4096].lstrip()
    if stripped.startswith(b"<"):
        try:
            root = SafeElementTree.fromstring(payload)
        except (DefusedXmlException, ElementTree.ParseError) as exc:
            raise HTTPException(400, "SVG 파일이 올바른 XML 문서가 아닙니다.") from exc
        root_name = root.tag.rsplit("}", 1)[-1] if isinstance(root.tag, str) else ""
        if root_name.lower() != "svg":
            raise HTTPException(415, "지원 형식은 SVG 또는 래스터 이미지입니다. PDF는 아직 지원하지 않습니다.")
        try:
            import cairosvg
            return cairosvg.svg2png(bytestring=payload), "png"
        except (ImportError, OSError) as exc:
            raise HTTPException(
                503,
                "SVG 변환을 사용할 수 없습니다. CairoSVG와 시스템 Cairo 라이브러리를 설치해 주세요.",
            ) from exc
        except (ValueError, RuntimeError, TypeError) as exc:
            raise HTTPException(400, f"SVG를 PNG 이미지로 변환하지 못했습니다: {exc}") from exc

    try:
        with IMAGE_DECODE_LOCK:
            with Image.open(io.BytesIO(payload)) as uploaded:
                original_format = uploaded.format
                if not original_format or original_format in {"PDF", "EPS", "PS", "MPEG"}:
                    raise HTTPException(
                        415,
                        f"지원하지 않는 이미지 형식입니다: {original_format or 'unknown'}. SVG 또는 래스터 이미지를 사용하세요. PDF는 아직 지원하지 않습니다.",
                    )
                oriented = ImageOps.exif_transpose(uploaded)
                if "A" in oriented.getbands():
                    rgba = oriented.convert("RGBA")
                    image = Image.new("RGB", rgba.size, "white")
                    image.paste(rgba, mask=rgba.getchannel("A"))
                else:
                    image = oriented.convert("RGB")
                if original_format in {"JPEG", "PNG", "WEBP", "BMP"}:
                    extension = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp", "BMP": "bmp"}[original_format]
                    return payload, extension
                normalized = io.BytesIO()
                image.save(normalized, format="PNG", optimize=True)
                return normalized.getvalue(), "png"
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(
            400, "서버에서 읽을 수 없는 이미지입니다. SVG 또는 지원되는 래스터 이미지로 다시 저장해 주세요.",
        ) from exc


def _validate_document(document: object, *, page: bool = False) -> dict:
    if not isinstance(document, dict):
        raise HTTPException(422, "Document body must be a JSON object")
    image = document.get("image")
    elements = document.get("elements")
    relations = document.get("relations")
    if not isinstance(image, dict) or not isinstance(elements, list) or not isinstance(relations, list):
        raise HTTPException(422, "Document needs image, elements, and relations fields")
    width, height = image.get("width"), image.get("height")
    if (not isinstance(width, (int, float)) or not isinstance(height, (int, float))
            or not math.isfinite(width) or not math.isfinite(height) or width <= 0 or height <= 0):
        raise HTTPException(422, "Document image dimensions must be positive finite numbers")
    ids = set()
    element_by_id = {}
    for element in elements:
        if not isinstance(element, dict):
            raise HTTPException(422, "Every element must be an object")
        element_id = element.get("id")
        bbox = element.get("bbox")
        if not isinstance(element_id, str) or not element_id or element_id in ids:
            raise HTTPException(422, "Element IDs must be non-empty and unique")
        ids.add(element_id)
        element_by_id[element_id] = element
        if element.get("type") not in {"Text", "Box", "Arrow", "Image", "Stroke"}:
            raise HTTPException(422, f"Unsupported element type for {element_id}")
        confidence = element.get("confidence")
        if confidence is not None and (
            not isinstance(confidence, (int, float)) or not math.isfinite(confidence)
            or confidence < 0 or confidence > 1
        ):
            raise HTTPException(422, f"Invalid confidence for {element_id}")
        if element.get("text") is not None and not isinstance(element.get("text"), str):
            raise HTTPException(422, f"Text content for {element_id} must be a string")
        if element.get("type") == "Box" and element.get("shape", "process") not in {
            "process", "decision", "terminator", "input_output", "document",
            "database", "predefined_process", "manual_input", "preparation",
            "connector", "offpage_connector", "delay",
        }:
            raise HTTPException(422, f"Unsupported shape for {element_id}")
        rotation = element.get("rotation", 0)
        if not isinstance(rotation, (int, float)) or not math.isfinite(rotation) or not -180 <= rotation <= 180:
            raise HTTPException(422, f"Invalid rotation for {element_id}")
        style = element.get("style", {})
        if not isinstance(style, dict):
            raise HTTPException(422, f"Invalid style for {element_id}")
        for color_key in ("fill", "stroke", "text_color"):
            color = style.get(color_key)
            if color is not None and (not isinstance(color, str) or not re.fullmatch(r"#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})", color)):
                raise HTTPException(422, f"Invalid {color_key} for {element_id}")
        for numeric_key, lower, upper in (("stroke_width", 1, 30), ("font_size", 8, 96)):
            value = style.get(numeric_key)
            if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value) or not lower <= value <= upper):
                raise HTTPException(422, f"Invalid {numeric_key} for {element_id}")
        opacity = style.get("opacity", 1)
        if not isinstance(opacity, (int, float)) or not math.isfinite(opacity) or not 0.1 <= opacity <= 1:
            raise HTTPException(422, f"Invalid opacity for {element_id}")
        if style.get("text_align", "center") not in {"left", "center", "right"}:
            raise HTTPException(422, f"Invalid text alignment for {element_id}")
        if style.get("marker_start", "none") not in {"none", "arrow", "diamond"} or style.get("marker_end", "arrow") not in {"none", "arrow", "diamond"}:
            raise HTTPException(422, f"Invalid arrow marker for {element_id}")
        if style.get("line_style", "solid") not in {"solid", "dashed", "dotted"}:
            raise HTTPException(422, f"Invalid line style for {element_id}")
        if element.get("type") == "Arrow" and element.get("route", "straight") not in {
            "straight", "orthogonal", "curved",
        }:
            raise HTTPException(422, f"Invalid arrow route for {element_id}")
        if (not isinstance(bbox, list) or len(bbox) != 4
                or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in bbox)
                or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]
                or bbox[0] < 0 or bbox[1] < 0 or bbox[2] > width or bbox[3] > height):
            raise HTTPException(422, f"Invalid bounding box for {element_id}")
        if element.get("type") == "Image":
            image_data = element.get("image_data", "")
            if not isinstance(image_data, str) or len(image_data) > 8_000_000 or not re.fullmatch(r"data:image/(?:png|jpeg|webp);base64,[A-Za-z0-9+/=]+", image_data):
                raise HTTPException(422, f"Invalid embedded image for {element_id}")
            try:
                raw_image = base64.b64decode(image_data.partition(",")[2], validate=True)
                with Image.open(io.BytesIO(raw_image)) as embedded:
                    if embedded.width * embedded.height > MAX_EMBEDDED_IMAGE_PIXELS:
                        raise HTTPException(422, f"Embedded image is too large for {element_id}")
                    expected = image_data[11:image_data.index(";base64,")].upper()
                    if embedded.format != expected:
                        raise HTTPException(422, f"Embedded image format mismatch for {element_id}")
                    embedded.verify()
            except (binascii.Error, UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
                raise HTTPException(422, f"Invalid embedded image for {element_id}") from exc
        if element.get("type") == "Stroke":
            points = element.get("points")
            if not isinstance(points, list) or not 2 <= len(points) <= 20_000:
                raise HTTPException(422, f"Invalid pen points for {element_id}")
            for point in points:
                if not isinstance(point, dict) or any(not isinstance(point.get(axis), (int, float)) or not math.isfinite(point[axis]) or not 0 <= point[axis] <= limit for axis, limit in (("x", width), ("y", height))):
                    raise HTTPException(422, f"Invalid pen point for {element_id}")
        if element.get("type") == "Arrow" and ("start" in element or "end" in element):
            for endpoint_name in ("start", "end"):
                endpoint = element.get(endpoint_name)
                if isinstance(endpoint, dict):
                    coordinates = [endpoint.get("x"), endpoint.get("y")]
                elif isinstance(endpoint, (list, tuple)) and len(endpoint) >= 2:
                    coordinates = list(endpoint[:2])
                else:
                    coordinates = []
                if (len(coordinates) != 2
                        or any(not isinstance(value, (int, float)) or not math.isfinite(value)
                               for value in coordinates)
                        or coordinates[0] < 0 or coordinates[0] > width
                        or coordinates[1] < 0 or coordinates[1] > height):
                    raise HTTPException(422, f"Invalid {endpoint_name} point for {element_id}")
    relation_ids = set()
    attached_arrow_ids = set()
    for relation in relations:
        if not isinstance(relation, dict):
            raise HTTPException(422, "Every relation must be an object")
        relation_id = relation.get("id")
        if not isinstance(relation_id, str) or not relation_id or relation_id in relation_ids:
            raise HTTPException(422, "Relation IDs must be non-empty and unique")
        relation_ids.add(relation_id)
        if relation.get("type") not in {"flows_to", "contains", "near_text"}:
            raise HTTPException(422, "Unsupported relation type")
        if relation.get("source") is not None and (not isinstance(relation["source"], str) or relation["source"] not in ids):
            raise HTTPException(422, "Relation source references an unknown element")
        if relation.get("target") is not None and (not isinstance(relation["target"], str) or relation["target"] not in ids):
            raise HTTPException(422, "Relation target references an unknown element")
        if relation.get("arrow_id") is not None:
            arrow = element_by_id.get(relation["arrow_id"])
            if not arrow or arrow.get("type") != "Arrow":
                raise HTTPException(422, "Relation arrow_id must reference an Arrow element")
            if relation.get("type") == "flows_to":
                if relation["arrow_id"] in attached_arrow_ids:
                    raise HTTPException(422, "One arrow cannot represent multiple flow relations")
                attached_arrow_ids.add(relation["arrow_id"])
        if relation.get("type") == "flows_to":
            for endpoint in (relation.get("source"), relation.get("target")):
                if endpoint is not None and element_by_id[endpoint].get("type") != "Box":
                    raise HTTPException(422, "Flow relations must connect Box elements")
    layers = document.get("layers", [])
    if layers:
        if not isinstance(layers, list) or len(layers) > 100 or any(not isinstance(layer, dict) or not isinstance(layer.get("id"), str) or not layer["id"] for layer in layers):
            raise HTTPException(422, "Invalid layer list")
        layer_ids = {layer["id"] for layer in layers}
        if len(layer_ids) != len(layers):
            raise HTTPException(422, "Duplicate layer ID")
        if document.get("active_layer") not in layer_ids:
            raise HTTPException(422, "Active layer is missing")
        if any(element.get("layer_id") not in layer_ids for element in elements):
            raise HTTPException(422, "Element references an unknown layer")
    canvas = document.get("canvas", {})
    if not isinstance(canvas, dict):
        raise HTTPException(422, "Invalid canvas settings")
    grid_size = canvas.get("grid_size", 10)
    if not isinstance(grid_size, (int, float)) or not math.isfinite(grid_size) or not 2 <= grid_size <= 100:
        raise HTTPException(422, "Grid size must be 2 to 100")
    if canvas.get("line_jump", "none") not in {"none", "arc", "gap", "sharp"}:
        raise HTTPException(422, "Invalid line jump style")
    if canvas.get("background") is not None and (not isinstance(canvas["background"], str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", canvas["background"])):
        raise HTTPException(422, "Invalid canvas background")
    if canvas.get("grid_color") is not None and (not isinstance(canvas["grid_color"], str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", canvas["grid_color"])):
        raise HTTPException(422, "Invalid grid color")
    if any(key in canvas and not isinstance(canvas[key], bool) for key in ("grid", "ruler")):
        raise HTTPException(422, "Invalid grid or ruler setting")
    if not page:
        pages = document.get("pages", [])
        if not isinstance(pages, list) or len(pages) > 50:
            raise HTTPException(422, "Invalid pages list")
        page_ids = [document.get("active_page", "page-1")]
        for other in pages:
            if not isinstance(other, dict) or not isinstance(other.get("id"), str) or not other["id"]:
                raise HTTPException(422, "Invalid page ID")
            page_ids.append(other["id"])
            _validate_document(other, page=True)
        if len(page_ids) != len(set(page_ids)):
            raise HTTPException(422, "Duplicate page ID")
        revision = document.get("revision", 0)
        if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
            raise HTTPException(422, "Invalid document revision")
    return document


def _load_document(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(500, "Saved session is unreadable") from exc


async def _stream_file(path: Path):
    with path.open("rb") as stream:
        while chunk := stream.read(256 * 1024):
            yield chunk


async def _remove_export(path: Path) -> None:
    path.unlink(missing_ok=True)


async def _run_analysis_worker(image_path: Path, result_path: Path, engine: str,
                               device: str, session_id: str, original_name: str) -> dict:
    command = [
        sys.executable, "-m", "src.web.backend.analysis_worker",
        str(image_path), str(result_path), engine, device, session_id, original_name, str(OUT),
    ]
    log_path = result_path.with_suffix(".log")
    deadline = asyncio.get_running_loop().time() + ANALYSIS_TIMEOUT_SECONDS
    with log_path.open("wb") as log_stream:
        try:
            process = subprocess.Popen(
                command, cwd=str(ROOT), stdout=log_stream, stderr=subprocess.STDOUT,
            )
        except OSError as exc:
            raise RuntimeError(f"분석 작업을 시작하지 못했습니다: {exc}") from exc
        try:
            while process.poll() is None:
                if asyncio.get_running_loop().time() >= deadline:
                    process.kill()
                    process.wait()
                    log_stream.flush()
                    output = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
                    if output:
                        print(f"[RADU worker:{session_id}] {output.rstrip()}", flush=True)
                    raise TimeoutError(
                        f"문서 분석이 {ANALYSIS_TIMEOUT_SECONDS}초 안에 끝나지 않아 작업을 종료했습니다. {output}"
                    )
                await asyncio.sleep(.1)
        except asyncio.CancelledError:
            if process.poll() is None:
                process.kill()
                process.wait()
            raise
        return_code = process.returncode
    output = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    if output:
        print(f"[RADU worker:{session_id}] {output.rstrip()}", flush=True)
    if return_code != 0:
        raise RuntimeError(f"분석 작업 프로세스가 종료 코드 {return_code}로 실패했습니다. {output}")
    try:
        return json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"분석 작업이 결과 문서를 만들지 못했습니다. {output[-2000:]}") from exc


def _analyze(image: np.ndarray, engine: str, device: str, session_id: str,
             source_path: Path, original_name: str) -> dict:
    processed = preprocess_image(image)
    image_height, image_width = processed.shape[:2]
    vl_only = engine == "paddleocrvl16"
    print(
        f"[RADU] analysis_start session={session_id} engine={engine} "
        f"requested_device={device} input={image_width}x{image_height}",
        flush=True,
    )
    ok, encoded_processed = cv2.imencode(".jpg", processed, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError("전처리 이미지를 저장하지 못했습니다.")
    processed_path = OUT / f"{session_id}.processed.jpg"
    _atomic_write(processed_path, encoded_processed.tobytes())
    if vl_only:
        document = {
            "version": "1.0",
            "image": {"width": image_width, "height": image_height},
            "elements": [],
            "relations": [],
            "warnings": [{
                "code": "radu_not_run",
                "message": "PaddleOCR-VL 비교 모드입니다. RADU 검출·관계 복원은 실행하지 않았습니다.",
            }],
            "radu": {"status": "not_run"},
        }
    else:
        document = analyze_image(processed, preprocess=False, include_text=engine == "none")
    warnings = list(document.get("warnings", []))
    if engine != "none":
        try:
            with INFERENCE_LOCK:
                text, details = recognize(processed, engine, device=device)
            lines = details.get("lines", [])
            document["ocr"] = {
                "engine": engine,
                "text": text,
                "lines": lines,
                "metadata": details.get("metadata", {}),
            }
            for index, line in enumerate(lines):
                raw = line.get("bbox") or []
                if raw and isinstance(raw[0], (list, tuple)):
                    xs = [point[0] for point in raw]
                    ys = [point[1] for point in raw]
                    bbox = [min(xs), min(ys), max(xs), max(ys)]
                elif len(raw) == 4:
                    bbox = raw
                else:
                    continue
                if len(bbox) != 4 or any(not isinstance(value, (int, float)) or not math.isfinite(value)
                                         for value in bbox):
                    continue
                bbox = [max(0, min(image_width, float(bbox[0]))),
                        max(0, min(image_height, float(bbox[1]))),
                        max(0, min(image_width, float(bbox[2]))),
                        max(0, min(image_height, float(bbox[3])))]
                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    continue
                document["elements"].append({
                    "id": f"ocr-text-{index + 1:04d}", "type": "Text", "bbox": bbox,
                    "text": line.get("text", ""), "original_text": line.get("text", ""), "confidence": line.get("confidence"),
                    "source": engine,
                })
            metadata = details.get("metadata", {})
            valid_boxes = sum(
                element.get("source") == engine and element.get("type") == "Text"
                for element in document["elements"]
            )
            sample_lines = [
                {"text": str(element.get("text", ""))[:40],
                 "bbox": [round(value, 1) for value in element["bbox"]]}
                for element in document["elements"]
                if element.get("source") == engine and element.get("type") == "Text"
            ][:3]
            print(
                f"[RADU] ocr_success session={session_id} engine={engine} "
                f"model={metadata.get('model', engine)} "
                f"device={metadata.get('device', device)} "
                f"lines={len(lines)} valid_boxes={valid_boxes} "
                f"sample_lines={json.dumps(sample_lines, ensure_ascii=True)}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[RADU] ocr_failure session={session_id} engine={engine} "
                f"requested_device={device} error={type(exc).__name__}: {exc}",
                flush=True,
            )
            warnings.append({"code": "ocr_unavailable", "message": f"OCR 분석에 실패했습니다: {exc}"})
    document["warnings"] = warnings
    if not vl_only:
        document = analyze_relations(document)
    document["session_id"] = session_id
    document["created_at"] = datetime.now(timezone.utc).isoformat()
    document["image"]["source_url"] = f"/api/sessions/{session_id}/image"
    document["image"]["processed_source_url"] = f"/api/sessions/{session_id}/processed-image"
    document["image"]["source_file"] = Path(original_name).name or source_path.name
    _atomic_write(OUT / f"{session_id}.json", json.dumps(document, ensure_ascii=False, indent=2).encode("utf-8"))
    return document


@app.post("/api/analyze")
async def analyze(request: Request, ocr_engine: str = Query("none"),
                  device: str = Query("cpu")) -> dict:
    engine = ocr_engine.strip().lower()
    if engine not in SUPPORTED_ENGINES:
        raise HTTPException(400, f"Unsupported OCR engine: {engine}")
    if device != "cpu" and not re.fullmatch(r"gpu(?::\d+)?", device):
        raise HTTPException(400, "Paddle device must be cpu, gpu, or gpu:<index>")
    # Raw image uploads are preferred; multipart is retained for older clients.
    content_type = request.headers.get("content-type", "application/octet-stream")
    payload = await request.body()
    original_name = urllib.parse.unquote(request.headers.get("x-file-name", ""))
    if content_type.startswith("multipart/form-data"):
        try:
            envelope = (f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n").encode() + payload
            message = BytesParser(policy=email.policy.default).parsebytes(envelope)
            part = next((item for item in message.walk() if item.get_filename()), None)
            if part is None:
                raise ValueError("multipart image field is missing")
            original_name = part.get_filename()
            payload = part.get_payload(decode=True) or b""
        except (ValueError, TypeError, email.errors.MessageError) as exc:
            raise HTTPException(400, "이미지 업로드 형식을 읽을 수 없습니다.") from exc
    original_name = Path(original_name or "upload-image").name
    if not payload:
        raise HTTPException(400, "업로드한 파일이 비어 있습니다.")
    payload, source_ext = _normalize_upload_image(payload)

    session_id = uuid4().hex
    source_path = OUT / f"{session_id}.source.{source_ext}"
    try:
        _atomic_write(source_path, payload)
        with tempfile.TemporaryDirectory(prefix="radu-analysis-") as temp_dir:
            worker_result = Path(temp_dir) / "result.json"
            return await _run_analysis_worker(source_path, worker_result, engine, device,
                                              session_id, original_name or source_path.name)
    except HTTPException:
        source_path.unlink(missing_ok=True)
        (OUT / f"{session_id}.processed.jpg").unlink(missing_ok=True)
        (OUT / f"{session_id}.json").unlink(missing_ok=True)
        raise
    except TimeoutError as exc:
        source_path.unlink(missing_ok=True)
        (OUT / f"{session_id}.processed.jpg").unlink(missing_ok=True)
        (OUT / f"{session_id}.json").unlink(missing_ok=True)
        raise HTTPException(504, str(exc)) from exc
    except Exception as exc:
        source_path.unlink(missing_ok=True)
        (OUT / f"{session_id}.processed.jpg").unlink(missing_ok=True)
        (OUT / f"{session_id}.json").unlink(missing_ok=True)
        if isinstance(exc, OSError):
            raise HTTPException(507, "이미지 저장 공간이 부족하거나 파일을 기록할 수 없습니다.") from exc
        raise HTTPException(500, f"문서 분석 중 오류가 발생했습니다: {exc}") from exc


@app.get("/api/sessions")
async def list_sessions() -> dict:
    sessions = []
    for path in OUT.glob("*.json"):
        try:
            if not SESSION_ID_RE.fullmatch(path.stem):
                continue
            document = json.loads(path.read_text(encoding="utf-8"))
            stat = path.stat()
            sessions.append({
                "session_id": path.stem,
                "name": document.get("name") or document.get("source_file") or "이름 없는 순서도",
                "updated_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "element_count": len(document.get("elements", [])),
            })
        except (OSError, json.JSONDecodeError, TypeError):
            continue
    sessions.sort(key=lambda item: item["updated_at"], reverse=True)
    return {"sessions": sessions}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str) -> dict:
    path = _session_path(session_id)
    if not path.is_file():
        raise HTTPException(404, "저장된 문서를 찾을 수 없습니다.")
    return _load_document(path)


@app.get("/api/sessions/{session_id}/image")
async def get_session_image(session_id: str) -> StreamingResponse:
    _session_path(session_id)
    paths = list(OUT.glob(f"{session_id}.source.*"))
    if not paths:
        raise HTTPException(404, "원본 이미지를 찾을 수 없습니다.")
    path = paths[0]
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return StreamingResponse(_stream_file(path), media_type=media_type,
                             headers={"Content-Disposition": "inline", "Content-Length": str(path.stat().st_size)})


@app.get("/api/sessions/{session_id}/processed-image")
async def get_processed_session_image(session_id: str) -> StreamingResponse:
    _session_path(session_id)
    path = OUT / f"{session_id}.processed.jpg"
    if not path.is_file():
        raise HTTPException(404, "전처리 이미지를 찾을 수 없습니다.")
    return StreamingResponse(_stream_file(path), media_type="image/jpeg",
                             headers={"Content-Disposition": "inline", "Content-Length": str(path.stat().st_size)})


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str) -> dict:
    path = _session_path(session_id)
    if not path.exists():
        raise HTTPException(404, "저장된 문서를 찾을 수 없습니다.")
    path.unlink()
    for source_path in OUT.glob(f"{session_id}.source.*"):
        source_path.unlink(missing_ok=True)
    (OUT / f"{session_id}.processed.jpg").unlink(missing_ok=True)
    for version_path in VERSIONS.glob(f"{session_id}.*.json"):
        version_path.unlink(missing_ok=True)
    return {"session_id": session_id, "deleted": True}


@app.post("/api/export/{fmt}")
async def export(fmt: str, document: dict, scale: float = Query(1, ge=1, le=4), transparent: bool = False, all_pages: bool = False) -> StreamingResponse:
    if fmt not in {"json", "svg", "png", "pdf"}:
        raise HTTPException(400, "지원하는 형식은 JSON, SVG, PNG, PDF입니다.")
    valid = _validate_document(document)
    path = EXPORTS / f"{uuid4().hex}.{fmt}"
    try:
        export_document(valid, path, fmt, scale=scale, transparent=transparent, all_pages=all_pages)
    except (ValueError, RuntimeError, OSError) as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(422, f"내보내기 실패: {exc}") from exc
    media = {"json": "application/json", "svg": "image/svg+xml", "png": "image/png", "pdf": "application/pdf"}[fmt]
    return StreamingResponse(_stream_file(path), media_type=media,
                             headers={"Content-Disposition": f'attachment; filename="radu-flowchart.{fmt}"',
                                      "Content-Length": str(path.stat().st_size)},
                             background=BackgroundTask(_remove_export, path))


@app.get("/api/sessions/{session_id}/versions")
async def list_versions(session_id: str) -> dict:
    _session_path(session_id)
    versions = []
    paths = sorted(
        VERSIONS.glob(f"{session_id}.v*.json"),
        key=lambda path: int(path.stem.rsplit(".v", 1)[-1]) if path.stem.rsplit(".v", 1)[-1].isdigit() else -1,
        reverse=True,
    )
    for path in paths:
        try:
            doc = _load_document(path)
            versions.append({"id": path.stem.split(".")[-1],
                             "saved_at": doc.get("updated_at"),
                             "element_count": len(doc.get("elements", []))})
        except (HTTPException, OSError):
            continue
    return {"versions": versions}


@app.get("/api/sessions/{session_id}/versions/{version_id}")
async def get_version(session_id: str, version_id: str) -> dict:
    _session_path(session_id)
    if not re.fullmatch(r"v[0-9]{1,10}", version_id):
        raise HTTPException(400, "Invalid version ID")
    path = VERSIONS / f"{session_id}.{version_id}.json"
    if not path.is_file():
        raise HTTPException(404, "이전 버전을 찾을 수 없습니다.")
    return _load_document(path)


@app.post("/api/sessions/{session_id}/duplicate")
async def duplicate_session(session_id: str) -> dict:
    original = _session_path(session_id)
    if not original.is_file():
        raise HTTPException(404, "저장된 문서를 찾을 수 없습니다.")
    doc = _load_document(original)
    new_id = uuid4().hex
    doc["session_id"] = new_id
    doc["revision"] = 1
    doc["name"] = (doc.get("name") or "순서도") + " 복사본"
    doc["updated_at"] = datetime.now(timezone.utc).isoformat()
    copied_source = False
    for old in OUT.glob(f"{session_id}.source.*"):
        suffix = old.name[len(session_id):]
        shutil.copyfile(old, OUT / f"{new_id}{suffix}")
        copied_source = True
    old_processed = OUT / f"{session_id}.processed.jpg"
    copied_processed = old_processed.is_file()
    if copied_processed:
        shutil.copyfile(old_processed, OUT / f"{new_id}.processed.jpg")
    for page_doc in [doc, *doc.get("pages", [])]:
        page_image = page_doc.get("image", {})
        if page_image.get("source_url") == f"/api/sessions/{session_id}/image":
            if copied_source:
                page_image["source_url"] = f"/api/sessions/{new_id}/image"
            else:
                page_image.pop("source_url", None)
        if page_image.get("processed_source_url") == f"/api/sessions/{session_id}/processed-image":
            if copied_processed:
                page_image["processed_source_url"] = f"/api/sessions/{new_id}/processed-image"
            else:
                page_image.pop("processed_source_url", None)
    _atomic_write(_session_path(new_id), json.dumps(doc, ensure_ascii=False, indent=2).encode())
    return {"session_id": new_id, "saved": True}


@app.post("/api/save")
async def save(document: dict) -> dict:
    valid = _validate_document(document)
    session_id = valid.get("session_id") or uuid4().hex
    path = _session_path(session_id)
    current = _load_document(path) if path.is_file() else None
    current_revision = int(current.get("revision", 0)) if current else 0
    if valid.get("revision", 0) != current_revision:
        raise HTTPException(409, "다른 창에서 문서가 변경되었습니다. 현재 편집본을 JSON으로 내보내고 최신 문서를 다시 여세요.")
    if current:
        _atomic_write(VERSIONS / f"{session_id}.v{current_revision}.json", json.dumps(current, ensure_ascii=False, indent=2).encode())
    valid["session_id"] = session_id
    valid["revision"] = current_revision + 1
    valid["updated_at"] = datetime.now(timezone.utc).isoformat()
    try:
        serialized = json.dumps(valid, ensure_ascii=False, indent=2).encode("utf-8")
        _atomic_write(path, serialized)
    except (TypeError, ValueError, OSError) as exc:
        raise HTTPException(500, f"문서를 저장하지 못했습니다: {exc}") from exc
    return {"session_id": session_id, "saved": True, "updated_at": valid["updated_at"], "revision": valid["revision"]}
