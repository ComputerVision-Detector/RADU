"""Local HybridOCR API. Run with uvicorn src.hybrid.app:app."""
from __future__ import annotations
import io
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from uuid import uuid4
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel, Field
from PIL import Image, ImageOps
from src.web.backend.app import _atomic_write, _validate_document
from fastapi.staticfiles import StaticFiles
from datetime import datetime, timezone
from copy import deepcopy

ROOT = Path(__file__).resolve().parents[2]
STORE = ROOT / 'data' / 'hybrid'
STORE.mkdir(parents=True, exist_ok=True)
LOCK = threading.Lock()
app = FastAPI(title='HybridOCR')
app.mount('/static', StaticFiles(directory=str(ROOT / 'src' / 'web' / 'frontend')), name='static')
VECTOR_LOCK = threading.Lock()

class Edit(BaseModel):
    markdown: str = Field(max_length=500_000)

class Dataset(BaseModel):
    dataset: str = 'hakuuung/HybridOCR_CustomNotes'
    split: str | None = None
    seed: int = 42


def folder(job_id):
    if not re.fullmatch(r'[a-f0-9]{32}', job_id):
        raise HTTPException(400, '잘못된 작업 ID입니다.')
    path = STORE / job_id
    if not (path / 'state.json').exists():
        raise HTTPException(404, '작업을 찾을 수 없습니다.')
    return path


def write_state(path, status, **extra):
    _atomic_write(path / 'state.json', json.dumps({'id': path.name, 'status': status, **extra}, ensure_ascii=False).encode())


def process(path, dataset=None):
    try:
        if dataset:
            write_state(path, 'loading_dataset')
            from src.hybrid.dataset import load_sample
            image, provenance = load_sample(dataset.dataset, dataset.split, dataset.seed)
            ImageOps.exif_transpose(image).convert('RGB').save(path / 'source.png')
            (path / 'dataset.json').write_text(json.dumps(provenance), encoding='utf-8')
        with LOCK:
            write_state(path, 'running')
            # One process group lets timeout cleanup reach Paddle and Qwen children.
            with (path / 'run.log').open('wb') as log:
                worker = subprocess.Popen([sys.executable, '-m', 'src.hybrid.pipeline', 'run',
                    str(path / 'source.png'), str(path)], cwd=ROOT, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    worker.wait(timeout=int(os.getenv('HYBRIDOCR_JOB_TIMEOUT', '1200')))
                except subprocess.TimeoutExpired:
                    import signal
                    os.killpg(worker.pid, signal.SIGKILL)
                    worker.wait()
                    raise RuntimeError('처리 시간이 초과되었습니다. run.log를 확인하세요.')
                if worker.returncode:
                    raise RuntimeError('모델 실행에 실패했습니다. CUDA·모델 설치 상태와 작업 폴더의 run.log를 확인하세요.')
            result = json.loads((path / 'review.json').read_text(encoding='utf-8'))
            _atomic_write(path / 'final.json', json.dumps({'markdown': result['markdown'], 'human_reviewed': False}, ensure_ascii=False).encode())
            from src.hybrid.document import build_document
            ocr = json.loads((path / 'ocr.json').read_text(encoding='utf-8'))
            document = build_document(path / 'source.png', ocr, result, path.name)
            _atomic_write(path / 'vector.json', json.dumps(document, ensure_ascii=False).encode())
            write_state(path, 'ready')
    except Exception as exc:
        write_state(path, 'failed', error=str(exc))


def create_job(image=None, dataset=None):
    path = STORE / uuid4().hex
    path.mkdir()
    if image:
        image.save(path / 'source.png')
    write_state(path, 'queued')
    threading.Thread(target=process, args=(path, dataset), daemon=True).start()
    return {'id': path.name}


@app.get('/', response_class=HTMLResponse)
def home():
    return (Path(__file__).parent / 'index.html').read_text(encoding='utf-8')


@app.post('/api/jobs')
async def upload(request: Request):
    payload = await request.body()
    try:
        with Image.open(io.BytesIO(payload)) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
    except (OSError, ValueError) as exc:
        raise HTTPException(400, '지원되는 래스터 이미지를 업로드하세요.') from exc
    return create_job(image=image)


@app.post('/api/dataset')
def dataset_job(body: Dataset):
    return create_job(dataset=body)


@app.get('/api/jobs/{job_id}')
def status(job_id: str):
    path = folder(job_id)
    result = json.loads((path / 'state.json').read_text(encoding='utf-8'))
    if (path / 'stage.txt').exists():
        result['stage'] = (path / 'stage.txt').read_text(encoding='utf-8')
    for key in ('ocr', 'review', 'final'):
        if (path / f'{key}.json').exists():
            result[key] = json.loads((path / f'{key}.json').read_text(encoding='utf-8'))
    return result


@app.get('/api/jobs/{job_id}/image')
def image(job_id: str):
    path = folder(job_id) / 'source.png'
    if not path.exists():
        raise HTTPException(404, '이미지 로딩 중입니다.')
    return FileResponse(path)


@app.put('/api/jobs/{job_id}')
def save(job_id: str, body: Edit):
    path = folder(job_id)
    if json.loads((path / 'state.json').read_text())['status'] != 'ready':
        raise HTTPException(409, '검수 결과가 준비되지 않았습니다.')
    _atomic_write(path / 'final.json', json.dumps({'markdown': body.markdown, 'human_reviewed': True}, ensure_ascii=False).encode())
    return {'saved': True}


@app.get('/api/jobs/{job_id}/pdf')
def pdf(job_id: str):
    path = folder(job_id)
    if (path / 'vector.json').exists():
        with VECTOR_LOCK:
            visual = _vector(path)
        if not visual.get('human_reviewed'):
            raise HTTPException(409, '벡터 검수 후 저장해 주세요.')
        from src.hybrid.vector_export import render_vector
        return Response(render_vector(visual, 'pdf'), media_type='application/pdf',
                        headers={'Content-Disposition': 'attachment; filename="HybridOCR.pdf"'})
    if not (path / 'final.json').exists():
        raise HTTPException(409, '결과가 준비되지 않았습니다.')
    document = json.loads((path / 'final.json').read_text(encoding='utf-8'))
    if not document['human_reviewed']:
        raise HTTPException(409, '사용자 검수 후 저장해 주세요.')
    from src.hybrid.pdf import render_pdf
    try:
        payload = render_pdf(document['markdown'])
    except (RuntimeError, ImportError, ValueError) as exc:
        raise HTTPException(503, str(exc)) from exc
    return Response(payload, media_type='application/pdf',
                    headers={'Content-Disposition': 'attachment; filename="HybridOCR.pdf"'})


# Vector documents are the primary review format. JSON is an internal transport only.
def _vector(path):
    if json.loads((path / 'state.json').read_text())['status'] != 'ready':
        raise HTTPException(409, '검수 결과가 준비되지 않았습니다.')
    output = path / 'vector.json'
    if not output.exists():
        from src.hybrid.document import build_document
        ocr = json.loads((path / 'ocr.json').read_text(encoding='utf-8'))
        review = json.loads((path / 'review.json').read_text(encoding='utf-8'))
        doc = build_document(path / 'source.png', ocr, review, path.name)
        _atomic_write(output, json.dumps(doc, ensure_ascii=False).encode())
    return json.loads(output.read_text(encoding='utf-8'))


def _valid_vector(path, document):
    doc = deepcopy(_validate_document(document))
    from PIL import Image
    with Image.open(path / 'source.png') as source:
        size = source.size
    if (doc['image']['width'], doc['image']['height']) != size or doc.get('pages'):
        raise HTTPException(422, '벡터 문서는 원본 한 페이지의 크기를 유지해야 합니다.')
    doc['image'] = {'width': size[0], 'height': size[1], 'source_url': f'/api/jobs/{path.name}/image'}
    doc['session_id'] = path.name
    return doc


@app.get('/review/{job_id}', response_class=HTMLResponse)
def visual_review(job_id: str):
    path = folder(job_id)
    with VECTOR_LOCK:
        _vector(path)
    html = (ROOT / 'src' / 'web' / 'frontend' / 'index.html').read_text(encoding='utf-8')
    html = html.replace('<title>RADU · 순서도 복원 작업실</title>', '<title>HybridOCR · 벡터 검수</title>')
    html = html.replace('<head>', '<head><script>window.HYBRID_REVIEW=' + json.dumps({'jobId': job_id}) + ';</script>')
    html = html.replace('</head>', '<link rel="stylesheet" href="/hybrid-review.css"></head>')
    html = html.replace('<strong>RADU</strong>', '<strong>HybridOCR</strong>').replace('FLOWCHART RESTORATION', 'VISUAL VECTOR REVIEW')
    html = html.replace('<div class="top-actions">', '<div class="top-actions"><a class="button" href="/">이미지 분석으로</a><button class="button" id="hybrid-table">표 추가</button><button class="button primary" id="hybrid-pdf">PDF 다운로드</button>')
    html = html.replace('id="save">저장', 'id="save">검수 완료 · 저장')
    html = html.replace('자동 검출은 후보 결과입니다.', '텍스트를 더블클릭해 수정하고, 요소를 드래그하거나 핸들로 크기를 조절하세요. 표는 셀별로 수정합니다. 자동 검출은 후보 결과입니다.')
    return html


@app.get('/hybrid-review.css')
def review_styles():
    return FileResponse(Path(__file__).parent / 'review.css', media_type='text/css')


@app.get('/api/jobs/{job_id}/vector')
def vector(job_id: str):
    with VECTOR_LOCK:
        return _vector(folder(job_id))


@app.put('/api/jobs/{job_id}/vector')
def save_vector(job_id: str, document: dict, reviewed: bool = Query(False)):
    path = folder(job_id)
    with VECTOR_LOCK:
        current = _vector(path)
        valid = _valid_vector(path, document)
        if valid.get('revision', 0) != current.get('revision', 0):
            raise HTTPException(409, '다른 창에서 수정한 문서입니다. 새로고침하여 최신 편집본을 확인하세요.')
        valid['revision'] = current.get('revision', 0) + 1
        valid['human_reviewed'] = reviewed
        valid['updated_at'] = datetime.now(timezone.utc).isoformat()
        _atomic_write(path / 'vector.json', json.dumps(valid, ensure_ascii=False).encode())
        return {'saved': True, 'session_id': job_id, 'revision': valid['revision'], 'updated_at': valid['updated_at']}


@app.post('/api/jobs/{job_id}/vector/export/{fmt}')
def export_vector(job_id: str, fmt: str, document: dict):
    path = folder(job_id)
    if fmt not in {'pdf', 'svg', 'png'}:
        raise HTTPException(400, 'PDF, SVG, PNG로 출력할 수 있습니다.')
    with VECTOR_LOCK:
        _vector(path)
    valid = _valid_vector(path, document)
    from src.hybrid.vector_export import render_vector
    try:
        payload = render_vector(valid, fmt)
    except (ImportError, RuntimeError, ValueError, OSError) as exc:
        raise HTTPException(503, f'벡터 출력 실패: {exc}') from exc
    return Response(payload, media_type={'pdf':'application/pdf', 'svg':'image/svg+xml', 'png':'image/png'}[fmt],
                    headers={'Content-Disposition': f'attachment; filename="HybridOCR.{fmt}"'})


@app.post('/api/analyze-relations')
def visual_relations(document: dict):
    from src.cv.relation import analyze_relations
    return analyze_relations(_validate_document(document))
