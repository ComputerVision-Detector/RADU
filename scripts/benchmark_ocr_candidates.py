"""Compare OCR/VLM candidates on the same manually transcribed word crops.

This is a recognition-only pilot. It does not measure flowchart element or edge
reconstruction; those require a separately annotated flowchart test set.
"""
import argparse
import base64
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
LABEL_FILE = ROOT / "data/ocr_real_eval/rec_gt.txt"
OUTPUT_DIR = ROOT / "data/ocr_candidate_benchmark"
ENGINES = ("ppocrv6", "paddleocr_vl_1_6", "clova", "qwen")
DEFAULT_ENGINES = ("ppocrv6", "paddleocr_vl_1_6", "qwen")


def jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "json"):
        data = value.json
        return jsonable(data() if callable(data) else data)
    return str(value)


def _unwrap(value):
    value = jsonable(value)
    if isinstance(value, dict) and isinstance(value.get("res"), dict):
        return value["res"]
    return value


def require_device(device):
    if not device.startswith("gpu"):
        return
    import paddle

    if not paddle.is_compiled_with_cuda() or paddle.device.cuda.device_count() < 1:
        raise RuntimeError(
            f"요청한 Paddle 장치 {device}를 사용할 수 없습니다. CPU로 자동 대체하지 않도록 중단합니다. "
            "GPU가 필요하면 CUDA/Paddle 설치와 GPU 접근 권한을 확인하거나 --device cpu를 명시하세요."
        )


def _first_prediction_text(value):
    value = _unwrap(value)
    if isinstance(value, list):
        return " ".join(filter(None, (_first_prediction_text(item) for item in value)))
    if not isinstance(value, dict):
        return ""
    for key in ("rec_text", "text", "markdown_text", "markdown"):
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    for key in ("rec_texts", "texts"):
        candidate = value.get(key)
        if isinstance(candidate, list):
            return " ".join(str(item).strip() for item in candidate if str(item).strip())
    # PaddleOCR-VL returns nested layout/recognition result objects. Collect
    # text-like values in output order while skipping paths and model metadata.
    parts = []
    for key, child in value.items():
        if key in {"input_path", "model_settings", "page_index"}:
            continue
        if isinstance(child, (dict, list)):
            text = _first_prediction_text(child)
            if text:
                parts.append(text)
    return "\n".join(parts)


def load_labels():
    rows = []
    for line in LABEL_FILE.read_text(encoding="utf-8").splitlines():
        image, target = line.split("\t", 1)
        rows.append({"image": Path(image), "target": target})
    if not rows:
        raise ValueError(f"정답 라벨이 비어 있습니다: {LABEL_FILE}")
    return rows


def prepare_ppocrv6(device):
    require_device(device)
    from paddleocr import TextRecognition

    model = TextRecognition(model_name="PP-OCRv6_medium_rec", device=device)

    def run(image_path):
        raw = model.predict(input=str(image_path))
        return _first_prediction_text(raw), jsonable(raw), {
            "model": "PP-OCRv6_medium_rec", "device": device,
            "language_coverage": "50 languages; Korean not supported",
        }

    def close():
        model.close()
    return run, close


def prepare_paddleocr_vl(device):
    require_device(device)
    from paddleocr import PaddleOCRVL

    try:
        model = PaddleOCRVL(
            pipeline_version="v1.6", device=device,
            use_doc_orientation_classify=False, use_doc_unwarping=False,
            use_layout_detection=False,
        )
    except RuntimeError as error:
        if "dependency error" in str(error).lower():
            from importlib.metadata import version
            paddlex_version = version("paddlex")
            raise RuntimeError(
                "PaddleOCR-VL-1.6 의존성이 없습니다. 현재 설치 버전에 맞춰 먼저 설치하세요: "
                f'python -m pip install "paddlex[ocr]=={paddlex_version}"'
            ) from error
        raise
    def run(image_path):
        raw = model.predict(input=str(image_path))
        return _first_prediction_text(raw), jsonable(raw), {
            "model": "PaddleOCR-VL-1.6", "device": device,
            "mode": "local document parser, layout detection disabled for word crops",
        }

    def close():
        close = getattr(model, "close", None)
        if callable(close):
            close()
    return run, close


def run_clova(image_path):
    invoke_url = os.environ.get("CLOVA_OCR_INVOKE_URL", "").strip()
    secret = os.environ.get("CLOVA_OCR_SECRET", "").strip()
    if not invoke_url or not secret:
        raise RuntimeError("CLOVA_OCR_INVOKE_URL과 CLOVA_OCR_SECRET 환경변수가 필요합니다.")
    suffix = image_path.suffix.lower().lstrip(".")
    if suffix not in {"jpg", "jpeg", "png", "bmp", "tiff", "tif", "pdf"}:
        raise ValueError(f"CLOVA OCR에서 지원하지 않는 확장자입니다: .{suffix}")
    request_body = {
        "images": [{
            "format": "jpg" if suffix == "jpeg" else suffix,
            "name": image_path.stem,
            "data": base64.b64encode(image_path.read_bytes()).decode("ascii"),
        }],
        "lang": "ko", "requestId": f"hybridocr-{time.time_ns()}",
        "resultType": "string", "timestamp": int(time.time() * 1000), "version": "V2",
    }
    request = urllib.request.Request(
        invoke_url,
        data=json.dumps(request_body).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-OCR-SECRET": secret},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"CLOVA OCR HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"CLOVA OCR 연결 오류: {error.reason}") from error
    texts = []
    for image in raw.get("images", []):
        if image.get("inferResult") != "SUCCESS":
            raise RuntimeError(f"CLOVA OCR 응답 오류: {image.get('message', image.get('inferResult'))}")
        texts.extend(field.get("inferText", "") for field in image.get("fields", []))
    return " ".join(text for text in texts if text), raw, {
        "model": "NAVER CLOVA OCR General OCR", "language": "ko", "mode": "remote API",
    }


def prepare_qwen(_device):
    from nlp.qwen import MODEL_ID, QwenModel
    from nlp.qwen import CACHE_DIR

    cached_model = CACHE_DIR / f"models--{MODEL_ID.replace('/', '--')}"
    if not cached_model.is_dir():
        raise RuntimeError(
            f"{MODEL_ID} 가중치가 {CACHE_DIR}에 없습니다. 예상치 못한 대용량 자동 다운로드를 막기 위해 건너뜁니다."
        )

    # An image at initialization selects AutoProcessor (with image support).
    active = {"model": QwenModel(image=Image.new("RGB", (64, 64), "white"), max_new_tokens=128)}

    def run(image_path):
        with Image.open(image_path) as source:
            active["model"].image = ImageOps.exif_transpose(source).convert("RGB")
        raw = active["model"].generate_json(
            "Read all visible handwriting in this single cropped word image. Preserve Korean and Latin text exactly. "
            "Do not translate, correct, or infer missing characters. Return JSON: {\"text\": \"transcription\"}.",
            {"task": "word_crop_transcription"},
        )
        text = raw.get("text", "")
        if not isinstance(text, str):
            raise ValueError("Qwen 응답의 text 필드가 문자열이 아닙니다.")
        return text.strip(), raw, {"model": MODEL_ID, "mode": "local VLM"}

    def close():
        model = active.pop("model", None)
        if model is not None:
            model.model.to("cpu")
            del model

    return run, close


def prepare_clova(_device):
    if not os.environ.get("CLOVA_OCR_INVOKE_URL", "").strip():
        raise RuntimeError("CLOVA_OCR_INVOKE_URL 환경변수가 필요합니다.")
    if not os.environ.get("CLOVA_OCR_SECRET", "").strip():
        raise RuntimeError("CLOVA_OCR_SECRET 환경변수가 필요합니다.")
    return run_clova, lambda: None


PREPARERS = {
    "ppocrv6": prepare_ppocrv6,
    "paddleocr_vl_1_6": prepare_paddleocr_vl,
    "clova": prepare_clova,
    "qwen": prepare_qwen,
}


def edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for row_index, left_char in enumerate(left, 1):
        current = [row_index]
        for col_index, right_char in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[col_index] + 1,
                previous[col_index - 1] + (left_char != right_char),
            ))
        previous = current
    return previous[-1]


def score(rows):
    if not rows:
        return {"samples": 0, "exact_accuracy": None, "cer": None}
    chars = sum(len(row["target"]) for row in rows)
    errors = sum(edit_distance(row["target"], row["prediction"]) for row in rows)
    korean_rows = [row for row in rows if any("\uac00" <= char <= "\ud7a3" for char in row["target"])]
    latin_rows = [row for row in rows if row["target"] and all(ord(char) < 128 for char in row["target"])]
    return {
        "samples": len(rows),
        "exact_accuracy": sum(row["target"] == row["prediction"] for row in rows) / len(rows),
        "cer": errors / chars if chars else 0.0,
        "exact_accuracy_ignore_spaces": sum(
            row["target"].replace(" ", "") == row["prediction"].replace(" ", "") for row in rows
        ) / len(rows),
        "hangul_or_mixed": _basic_metrics(korean_rows),
        "latin_only": _basic_metrics(latin_rows),
    }


def _basic_metrics(rows):
    if not rows:
        return {"samples": 0, "exact_accuracy": None, "cer": None}
    total_chars = sum(len(row["target"]) for row in rows)
    return {
        "samples": len(rows),
        "exact_accuracy": sum(row["target"] == row["prediction"] for row in rows) / len(rows),
        "cer": sum(edit_distance(row["target"], row["prediction"]) for row in rows) / total_chars if total_chars else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engines", nargs="+", choices=ENGINES, default=list(DEFAULT_ENGINES),
        help="후보 엔진 (CLOVA는 유료 API 설정이 있을 때 --engines clova로 별도 지정)",
    )
    parser.add_argument("--device", default="gpu:0", help="PaddleOCR device, e.g. gpu:0 or cpu")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--limit", type=int, help="Limit samples for an initial smoke run")
    args = parser.parse_args()
    labels = load_labels()
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("--limit은 양수여야 합니다.")
        labels = labels[:args.limit]
    missing = [str(row["image"]) for row in labels if not row["image"].is_file()]
    if missing:
        raise FileNotFoundError(f"평가 이미지가 없습니다: {missing[:3]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "benchmark": "recognition-only pilot on fixed word crops",
        "labels_sha256": hashlib.sha256(LABEL_FILE.read_bytes()).hexdigest(),
        "dataset": str(LABEL_FILE.relative_to(ROOT)),
        "sample_count": len(labels),
        "input_images": [str(row["image"].relative_to(ROOT)) for row in labels],
        "flowchart_structure_evaluated": False,
        "engines": {},
    }
    print(f"평가 이미지: {len(labels)}개 단어 크롭 ({LABEL_FILE.relative_to(ROOT)})", flush=True)
    for engine in args.engines:
        init_started = time.perf_counter()
        try:
            runner, cleanup = PREPARERS[engine](args.device)
            init_seconds = time.perf_counter() - init_started
            init_error = None
        except Exception as error:
            runner, cleanup = None, lambda: None
            init_seconds = time.perf_counter() - init_started
            init_error = f"{type(error).__name__}: {error}"
        rows, engine_meta = [], None
        print(f"\n=== {engine} ===", flush=True)
        if init_error:
            print(f"초기화 실패: {init_error}", flush=True)
        for index, sample in enumerate(labels, 1):
            if init_error:
                rows.append({"image": str(sample["image"].relative_to(ROOT)),
                             "target": sample["target"], "error": f"model initialization failed: {init_error}"})
                continue
            started = time.perf_counter()
            try:
                prediction, raw, meta = runner(sample["image"])
                elapsed = time.perf_counter() - started
                engine_meta = meta
                rows.append({
                    "image": str(sample["image"].relative_to(ROOT)),
                    "target": sample["target"], "prediction": prediction,
                    "latency_seconds": round(elapsed, 4),
                })
                print(f"[{index}/{len(labels)}] {sample['target']} => {prediction} ({elapsed:.2f}s)", flush=True)
            except Exception as error:
                rows.append({"image": str(sample["image"].relative_to(ROOT)),
                             "target": sample["target"], "error": f"{type(error).__name__}: {error}"})
                print(f"[{index}/{len(labels)}] ERROR: {error}", flush=True)
        cleanup()
        successful = [row for row in rows if "prediction" in row]
        report["engines"][engine] = {
            "metadata": engine_meta or {}, "model_initialization_seconds": round(init_seconds, 3),
            "initialization_error": init_error, "metrics": score(successful),
            "mean_latency_seconds": (
                sum(row["latency_seconds"] for row in successful) / len(successful) if successful else None
            ),
            "predictions": rows,
        }
        (args.output_dir / f"{engine}.json").write_text(
            json.dumps(report["engines"][engine], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    output = args.output_dir / "summary.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nSummary: {output}")


if __name__ == "__main__":
    main()
