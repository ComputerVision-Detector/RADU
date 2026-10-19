"""OCR adapters and reproducible PaddleOCR/CLOVA benchmark runner."""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests


def _prepare_paddlex_cache() -> None:
    """Keep PaddleX runtime caches writable and model weights project-local."""
    project_cache = Path(__file__).resolve().parents[2] / ".cache" / "paddlex"
    cache_root = Path(os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(project_cache)))
    cache_root.mkdir(parents=True, exist_ok=True)


def _model_path(relative: str) -> str:
    path = Path(__file__).resolve().parents[2] / "models" / relative
    if not path.is_dir():
        raise RuntimeError(f"Required local RADU model is missing: {path}")
    return str(path)


def _allow_cpu_vl_on_cuda_build(device: str) -> None:
    """Disable only an optional GPU fast-path probe when CPU-only hardware is present.

    Some CUDA-enabled Paddle builds call get_device_capability while creating the
    VL attention layers even when PaddleX was explicitly configured for CPU.
    Returning None selects the regular attention implementation.
    """
    if device.lower() != "cpu":
        return
    try:
        import paddle
        if not paddle.is_compiled_with_cuda() or paddle.device.cuda.device_count() > 0:
            return
        from paddlex.inference.models.doc_vlm.modeling.paddleocr_vl import _siglip
        _siglip.get_gpu_compute_capability = lambda: None
    except (ImportError, OSError, RuntimeError):
        # Keep the upstream behavior if the installed PaddleX internals differ.
        return


def _result_payload(result: Any) -> dict[str, Any]:
    """Normalize PaddleOCR 3.x result wrappers and plain result dictionaries."""
    if isinstance(result, dict):
        return result.get("res", result)
    value = getattr(result, "json", None)
    if callable(value): value = value()
    if isinstance(value, dict): return value.get("res", value)
    value = getattr(result, "res", None)
    return value if isinstance(value, dict) else {}


def _paddleocrv6(image: np.ndarray, requested_device: str | None = None) -> tuple[str, dict[str, Any]]:
    """Local PP-OCRv6 text recognition pipeline (PaddleOCR 3.x)."""
    _prepare_paddlex_cache()
    from paddleocr import PaddleOCR
    global _PADDLE_V6
    device = requested_device or os.getenv("RADU_PADDLE_DEVICE", "cpu")
    if "_PADDLE_V6" in globals() and globals().get("_PADDLE_V6_DEVICE") != device:
        release_paddleocrv6()
    if "_PADDLE_V6" not in globals():
        _PADDLE_V6 = PaddleOCR(ocr_version="PP-OCRv6",
                               text_detection_model_dir=_model_path("detection/PP-OCRv6_medium_det"),
                               text_recognition_model_dir=_model_path("recognition/PP-OCRv6_medium_rec"),
                               use_doc_orientation_classify=False,
                               use_doc_unwarping=False, use_textline_orientation=False,
                               device=device)
        globals()["_PADDLE_V6_DEVICE"] = device
    results = _PADDLE_V6.predict(image)
    lines = []
    for result in results:
        payload = _result_payload(result)
        texts = payload.get("rec_texts", [])
        scores = payload.get("rec_scores", [])
        boxes = payload.get("rec_polys", payload.get("dt_polys", []))
        for i, text in enumerate(texts):
            box = np.asarray(boxes[i]).tolist() if i < len(boxes) else []
            lines.append({"text": str(text), "confidence": float(scores[i]) if i < len(scores) else None,
                          "bbox": box})
    return "\n".join(x["text"] for x in lines), {"lines": lines, "resource": "local",
            "model": "PP-OCRv6", "device": device}


def release_paddleocrv6() -> None:
    """Release PP-OCRv6 GPU allocations before starting the isolated VL worker."""
    model = globals().pop("_PADDLE_V6", None)
    globals().pop("_PADDLE_V6_DEVICE", None)
    if model is not None:
        close = getattr(model, "close", None)
        if callable(close):
            close()
        del model
    try:
        import paddle
        paddle.device.cuda.empty_cache()
    except Exception:
        pass


def _paddleocrvl16(image: np.ndarray, device: str | None = None) -> tuple[str, dict[str, Any]]:
    """Run the known working direct spotting path in a killable GPU worker."""
    from PIL import Image

    device = device or os.getenv("RADU_PADDLE_DEVICE", "cpu")
    with tempfile.TemporaryDirectory(prefix="radu-vl-") as workdir:
        image_path = Path(workdir) / "input.png"
        result_path = Path(workdir) / "result.json"
        if image.ndim == 3 and image.shape[2] == 3:
            Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(image_path)
        else:
            Image.fromarray(image).save(image_path)
        command = [
            sys.executable, "-m", "src.evaluation.paddleocr_vl_worker",
            str(image_path), str(result_path), "--device", device,
        ]
        worker_env = os.environ.copy()
        # Paddle reads this during import; setting it in the worker module
        # also covers direct worker invocation, while this guarantees the
        # benchmark subprocess inherits the same allocator configuration.
        worker_env.setdefault("FLAGS_allocator_strategy", "naive_best_fit")
        # Long or visually noisy pages can make spotting decode much longer
        # than ordinary pages. Preserve the reference 2048-token pass first;
        # if it exceeds the per-image budget, retry with a bounded decode so
        # one pathological page does not leave the benchmark without OCR.
        try:
            completed = subprocess.run(command, timeout=60, check=False, env=worker_env)
        except subprocess.TimeoutExpired as error:
            retry_env = os.environ.copy()
            retry_env.setdefault("FLAGS_allocator_strategy", "naive_best_fit")
            retry_env["RADU_VL_MAX_NEW_TOKENS"] = "1024"
            print(
                "[VL] 60초 내 spotting 미완료; max_new_tokens=1024로 재시도합니다.",
                flush=True,
            )
            try:
                completed = subprocess.run(
                    command, timeout=60, check=False, env=retry_env,
                )
            except subprocess.TimeoutExpired as retry_error:
                raise TimeoutError(
                    "PaddleOCR-VL이 기본 생성과 1024-token 재시도 모두 60초 안에 끝나지 않았습니다."
                ) from retry_error
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if "error" in result:
                raise RuntimeError(result["error"])
            raw = result["raw"]
            spotting = raw.get("spotting_res", {})
            texts = spotting.get("rec_texts", []) if isinstance(spotting, dict) else []
            polygons = spotting.get("rec_polys", []) if isinstance(spotting, dict) else []
            raw["lines"] = [
                {"text": text, "bbox": polygons[index] if index < len(polygons) else []}
                for index, text in enumerate(texts)
            ]
            raw["metadata"] = result["metadata"]
            return result["prediction"], raw
        raise RuntimeError(
            f"PaddleOCR-VL worker exited without a result (exit code {completed.returncode})."
        )


def _clova(image: np.ndarray) -> tuple[str, dict[str, Any]]:
    endpoint = os.getenv("CLOVA_OCR_INVOKE_URL")
    secret = os.getenv("CLOVA_OCR_SECRET")
    if not endpoint or not secret:
        raise RuntimeError("Set CLOVA_OCR_INVOKE_URL and CLOVA_OCR_SECRET to enable CLOVA OCR")
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Unable to encode image for CLOVA OCR")
    payload = {"images": [{"format": "png", "name": "radu-" + uuid.uuid4().hex[:8],
                           "data": base64.b64encode(encoded).decode("ascii")}],
               "requestId": str(uuid.uuid4()), "version": "V2", "timestamp": int(time.time() * 1000)}
    response = requests.post(endpoint, headers={"X-OCR-SECRET": secret}, json=payload, timeout=60)
    response.raise_for_status()
    body = response.json()
    lines = []
    for field in body.get("images", [{}])[0].get("fields", []):
        vertices = field.get("boundingPoly", {}).get("vertices", [])
        bbox = [[p.get("x", 0), p.get("y", 0)] for p in vertices]
        lines.append({"text": field.get("inferText", ""), "confidence": field.get("inferConfidence"), "bbox": bbox})
    return "\n".join(x["text"] for x in lines), {"lines": lines, "resource": "remote"}


def recognize(image: np.ndarray, engine: str, device: str | None = None) -> tuple[str, dict[str, Any]]:
    key = engine.strip().lower().replace("-", "").replace("_", "")
    if key in {"paddleocrv6", "ppocrv6"}:
        return _paddleocrv6(image, requested_device=device)
    if key in {"paddleocrvl16", "paddleocrvl1.6", "ppocrvl16"}:
        return _paddleocrvl16(image, device=device)
    if key == "clova":
        return _clova(image)
    raise ValueError(f"Unsupported OCR engine: {engine}")


def _cer(reference: str, hypothesis: str) -> float:
    try:
        from jiwer import cer
        return float(cer(reference, hypothesis))
    except ImportError:
        # Levenshtein distance, including spaces and punctuation, divided by reference length.
        previous = list(range(len(hypothesis) + 1))
        for i, a in enumerate(reference, 1):
            current = [i]
            for j, b in enumerate(hypothesis, 1):
                current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
            previous = current
        return previous[-1] / max(1, len(reference))


def benchmark(image_dir: str | Path, ground_truth_json: str | Path, output_json: str | Path,
              engines: tuple[str, ...] = ("paddleocrv6", "paddleocrvl16")) -> dict[str, Any]:
    """Run OCR on image_dir; GT JSON maps image filenames to reference strings."""
    root = Path(image_dir)
    gt = json.loads(Path(ground_truth_json).read_text(encoding="utf-8"))
    rows = []
    for path in sorted(p for p in root.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}):
        image = cv2.imread(str(path))
        if image is None:
            rows.append({"image": path.name, "error": "unreadable image"})
            continue
        for engine in engines:
            started = time.perf_counter()
            try:
                text, details = recognize(image, engine)
                latency = time.perf_counter() - started
                reference = gt.get(path.name)
                rows.append({"image": path.name, "engine": engine, "latency_seconds": round(latency, 4),
                             "cer": _cer(str(reference), text) if reference is not None else None,
                             "reference_available": reference is not None, "text": text, "details": details})
            except Exception as exc:
                rows.append({"image": path.name, "engine": engine, "error": str(exc),
                             "latency_seconds": round(time.perf_counter() - started, 4)})
    summaries = {}
    for engine in engines:
        valid = [r for r in rows if r.get("engine") == engine and "error" not in r]
        cer_values = [r["cer"] for r in valid if r["cer"] is not None]
        resource_notes = {
            "clova": "remote CLOVA OCR API; network, credentials, quota, and provider cost required",
            "paddleocrv6": "local PP-OCRv6; PaddlePaddle runtime, model storage, and CPU/GPU memory required",
            "paddleocrvl16": "local full PaddleOCR-VL-1.6 layout+VLM pipeline; substantially higher memory/compute; optional VLM server backend",
        }
        summaries[engine] = {"samples": len(valid), "mean_latency_seconds": round(float(np.mean([r["latency_seconds"] for r in valid])), 4) if valid else None,
                             "first_inference_seconds": round(valid[0]["latency_seconds"], 4) if valid else None,
                             "mean_warm_inference_seconds": round(float(np.mean([r["latency_seconds"] for r in valid[1:]])), 4) if len(valid) > 1 else None,
                             "mean_cer": round(float(np.mean(cer_values)), 4) if cer_values else None,
                             "resource": resource_notes.get(engine, "configure resource notes for this engine")}
    eligible = [(v["mean_cer"], v["mean_latency_seconds"], k) for k, v in summaries.items() if v["mean_cer"] is not None]
    recommendation = min(eligible)[2] if eligible else None
    report = {"schema_version": "1.0", "image_dir": str(root), "sample_count": len({r["image"] for r in rows}),
              "candidate_engines": list(engines), "summaries": summaries,
              "recommended_baseline": recommendation,
              "baseline_selection": {"status": "pending_team_review", "recommended_model": recommendation},
              "results": rows}
    output = Path(output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
