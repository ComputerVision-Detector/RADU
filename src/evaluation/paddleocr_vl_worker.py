"""Isolated PaddleOCR-VL 1.6 spotting worker used by the visual benchmark."""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

from PIL import Image

# Paddle's GPU auto_growth allocator can stall indefinitely in the BF16
# embedding kernel used by PaddleOCR-VL 1.6. Set this before importing Paddle.
# Respect an explicit operator override when diagnosing other allocator issues.
os.environ.setdefault("FLAGS_allocator_strategy", "naive_best_fit")


def _unwrap(value):
    if isinstance(value, list):
        return [_unwrap(item) for item in value]
    if hasattr(value, "json"):
        data = value.json
        value = data() if callable(data) else data
    if isinstance(value, dict) and isinstance(value.get("res"), dict):
        return value["res"]
    return value


def prepare_model(device: str):
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    # Keep this immediately before the lazy Paddle import as well, in case a
    # caller imports this worker after changing the process environment.
    os.environ.setdefault("FLAGS_allocator_strategy", "naive_best_fit")
    import paddle
    from packaging.version import Version

    if Version(paddle.__version__) < Version("3.2.1"):
        raise RuntimeError(f"PaddleOCR-VL-1.6 requires PaddlePaddle >= 3.2.1; found {paddle.__version__}")
    if device.startswith("gpu") and (
        not paddle.is_compiled_with_cuda() or paddle.device.cuda.device_count() < 1
    ):
        raise RuntimeError(f"Requested Paddle device {device} is unavailable; refusing CPU fallback")
    paddle.set_device(device)
    if device.startswith("gpu"):
        import subprocess
        gpu_index = device.partition(":")[2] or "0"
        try:
            result = subprocess.run(
                ["nvidia-smi", f"--id={gpu_index}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=True,
            )
            free_mb = int(result.stdout.strip().splitlines()[0])
        except (OSError, subprocess.SubprocessError, ValueError, IndexError) as error:
            raise RuntimeError("Could not verify free GPU memory with nvidia-smi") from error
        if free_mb < 10_000:
            raise RuntimeError(f"PaddleOCR-VL needs at least 10000 MiB free VRAM; found {free_mb} MiB")

    from paddlex.utils.cache import CACHE_DIR
    model_root = Path(CACHE_DIR) / "official_models"
    required_files = {
        "PaddleOCR-VL-1.6": (
            "config.json", "generation_config.json", "model.safetensors",
            "processor_config.json", "tokenizer.json",
        ),
        "PP-DocLayoutV3": ("inference.yml", "inference.json", "inference.pdiparams"),
    }
    missing = [
        str(model_root / model_name / filename)
        for model_name, filenames in required_files.items()
        for filename in filenames
        if not (model_root / model_name / filename).is_file()
    ]
    if missing:
        raise RuntimeError("Required cached PaddleOCR-VL files are missing: " + ", ".join(missing))
    try:
        import ml_dtypes  # noqa: F401
    except ImportError as error:
        raise RuntimeError("PaddleOCR-VL BF16 loading requires ml-dtypes") from error

    import numpy as np
    import safetensors
    original_safe_open = safetensors.safe_open

    class PaddleSlice:
        def __init__(self, view):
            self.view = view
            self.shape = view.get_shape()

        def get_shape(self):
            return self.shape

        def __getitem__(self, selector):
            array = self.view[selector]
            if str(array.dtype) == "bfloat16":
                return paddle.to_tensor(array.astype(np.float32), dtype="bfloat16")
            return paddle.to_tensor(array)

    class NumpySafeOpen:
        def __init__(self, filename, **kwargs):
            kwargs.pop("framework", None)
            kwargs.pop("device", None)
            self.file = original_safe_open(filename, framework="numpy", **kwargs)

        def __enter__(self):
            self.file.__enter__()
            return self

        def __exit__(self, *args):
            return self.file.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.file, name)

        def get_slice(self, name):
            return PaddleSlice(self.file.get_slice(name))

    safetensors.safe_open = NumpySafeOpen
    try:
        from paddlex import create_model
        model = create_model(
            "PaddleOCR-VL-1.6-0.9B",
            model_dir=str(model_root / "PaddleOCR-VL-1.6"),
            device=device,
        )
    finally:
        safetensors.safe_open = original_safe_open
    return model


def run(image_path: Path, device: str):
    print("[VL] Local model loading started", flush=True)
    init_started = time.perf_counter()
    model = prepare_model(device)
    init_seconds = time.perf_counter() - init_started
    print(f"[VL] Model loaded ({init_seconds:.2f}s); spotting inference started", flush=True)
    from paddlex.inference.pipelines.paddleocr_vl.uilts import post_process_for_spotting

    inference_started = time.perf_counter()
    max_new_tokens = int(os.environ.get("RADU_VL_MAX_NEW_TOKENS", "2048"))
    import faulthandler
    faulthandler.dump_traceback_later(20, repeat=True)
    try:
        predictions = model.predict(
            [{"image": str(image_path), "query": "Spotting:"}],
            min_pixels=112_896, max_pixels=602_112, max_new_tokens=max_new_tokens,
            use_cache=True, skip_special_tokens=False,
        )
        result = next(iter(predictions), None)
    finally:
        faulthandler.cancel_dump_traceback_later()
    if result is None:
        raise RuntimeError("PaddleOCR-VL returned no prediction")
    result = _unwrap(result)
    with Image.open(image_path) as image:
        width, height = image.size
    text, spotting = post_process_for_spotting(result["result"], width, height)
    metadata = {
        "model": "PaddleOCR-VL-1.6", "device": device,
        "mode": "isolated direct spotting inference",
        "paddle_allocator_strategy": os.environ.get("FLAGS_allocator_strategy"),
        "max_pixels": 602_112, "max_new_tokens": max_new_tokens,
        "model_initialization_seconds": round(init_seconds, 4),
        "inference_seconds": round(time.perf_counter() - inference_started, 4),
    }
    print(f"[VL] Inference completed ({metadata['inference_seconds']:.2f}s)", flush=True)
    return text, {"spotting_res": spotting}, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("--device", default="gpu:0")
    args = parser.parse_args()
    try:
        prediction, raw, metadata = run(args.image, args.device)
        payload = {"prediction": prediction, "raw": raw, "metadata": metadata}
    except Exception as error:
        payload = {"error": f"{type(error).__name__}: {error}"}
    args.result.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    try:
        import paddle
        if args.device.startswith("gpu"):
            paddle.device.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


if __name__ == "__main__":
    main()
