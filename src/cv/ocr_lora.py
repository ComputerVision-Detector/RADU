"""Optional, fixed selected LoRA OCR; existing OCR engines stay independent."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
ENGINE = 'paddleocrvl16lora'
ADAPTER = Path('data/experiments/ocr_v2_vl_lora/train/best-adapter')
ADAPTER_SHA256 = 'ab1bfcfba5c0711ebfc82e83432eca13b47fbb48ca0800d9e7b4d590e15421e5'
ADAPTER_CONFIG_SHA256 = 'f68fe0b7694f220dee10845075cbabfaed88cab46906a5ce58bab5304b5f729d'
BINDING_SHA256 = '73c19d2ffe91db34a84427174a9bb0a3356f629e147cdab0ca4b4cdbaf213c55'


def check_digest(path: Path, expected: str) -> str:
    if not path.is_file():
        raise RuntimeError(f'고정 LoRA 구성 파일이 없습니다: {path.name}')
    h = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1048576), b''):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected:
        raise RuntimeError(f'고정 모델 SHA-256 불일치: {path.name}. 다른 모델로 전환하지 않았습니다.')
    return actual


def gpu_device(device: str | None) -> str:
    device = device or os.getenv('RADU_PADDLE_DEVICE', 'cpu')
    if not re.fullmatch(r'gpu(?::\d+)?', device):
        raise ValueError('LoRA OCR은 검증된 GPU 장치(gpu:0 등)를 선택해 주세요.')
    return 'cuda:' + (device.partition(':')[2] or '0')


def verify_adapter(root: Path = ROOT) -> dict:
    return {
        'adapter_sha256': check_digest(root / ADAPTER / 'adapter_model.safetensors', ADAPTER_SHA256),
        'adapter_config_sha256': check_digest(root / ADAPTER / 'adapter_config.json', ADAPTER_CONFIG_SHA256),
    }


def fixed_binding() -> dict:
    path = ROOT / 'configs/ocr_lora_backend.lock.json'
    check_digest(path, BINDING_SHA256)
    binding = json.loads(path.read_text())
    for name, expected in binding['files'].items():
        check_digest(ROOT / name, expected)
    verify_adapter()
    return binding


def run_stage(stage: str, image: Path, output: Path, device: str, workdir: Path,
              *, detections: Path | None = None) -> None:
    variable = 'RADU_LORA_DETECTOR_PYTHON' if stage == 'detect' else 'RADU_LORA_PYTHON'
    relative = '.cache/venvs/ocr-v2/bin/python' if stage == 'detect' else '.cache/venvs/ocr-vl-lora/bin/python'
    python = Path(os.environ.get(variable, str(ROOT / relative)))
    if not python.is_file():
        raise RuntimeError(f'LoRA OCR 실행 환경이 없습니다. {variable} 설정을 확인해 주세요.')
    command = [str(python), '-m', 'src.cv.ocr_lora_worker', stage, str(image), str(output), '--device', device]
    if detections is not None:
        command += ['--detections', str(detections)]
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK='True')
    with (workdir / (stage + '.log')).open('w') as log:
        try:
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, timeout=60 if stage == 'detect' else 600)
        except subprocess.TimeoutExpired as error:
            raise TimeoutError(f'LoRA OCR {stage} 단계의 실행 시간을 초과했습니다.') from error
    if not output.is_file():
        tail = (workdir / (stage + '.log')).read_text(errors='replace')[-1600:]
        raise RuntimeError(f'LoRA OCR {stage} 실패: {tail}')
    payload = json.loads(output.read_text())
    if completed.returncode != 0 or payload.get('error'):
        raise RuntimeError(f"LoRA OCR {stage} 실패: {payload.get('error', 'worker failed')}")


def recognize_lora(image, device: str | None = None) -> tuple[str, dict]:
    from PIL import Image

    cuda = gpu_device(device)
    device = 'gpu:' + cuda.partition(':')[2]
    verified = verify_adapter()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='radu-fixed-lora-') as directory:
        workdir = Path(directory)
        image_path = workdir / 'input.png'
        if image.ndim == 3 and image.shape[2] == 3:
            Image.fromarray(image[:, :, ::-1]).save(image_path)
        elif image.ndim == 2:
            Image.fromarray(image).convert('RGB').save(image_path)
        else:
            raise ValueError('LoRA OCR에는 BGR 또는 회색조 이미지가 필요합니다.')
        detection_path = workdir / 'detections.json'
        run_stage('detect', image_path, detection_path, device, workdir)
        detections = json.loads(detection_path.read_text())
        if detections['regions']:
            result_path = workdir / 'recognition.json'
            run_stage('recognize', image_path, result_path, device, workdir, detections=detection_path)
            result = json.loads(result_path.read_text())
            lines, metadata = result['lines'], result['metadata']
        else:
            lines = []
            metadata = {'recognizer_called': False, 'generation_limit_regions': [], 'device': cuda}
        metadata.update(verified, model='PaddleOCR-VL-1.6 + fixed LoRA',
                        detector='PP-OCRv6_medium_det', detector_seconds=detections['seconds'],
                        detection_count=len(detections['regions']), input_sha256=detections['image_sha256'],
                        width=detections['width'], height=detections['height'],
                        total_seconds=time.perf_counter()-started, engine_path='PP-OCRv6 detection + fixed crop + LoRA OCR')
        return '\n'.join(line['text'] for line in lines), {
            'lines': lines, 'metadata': metadata, 'model': metadata['model'], 'device': cuda, 'resource': 'local',
        }
