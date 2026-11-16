"""Opt-in RADU bbox routing; existing engines/detector/crops remain independent."""
from __future__ import annotations
import json,os,pathlib,subprocess,tempfile,time
from src.cv.ocr_lora import ROOT,gpu_device,run_stage,verify_adapter

def recognize_routed(image,device=None):
    from PIL import Image
    if gpu_device(device)!='cuda:0':raise ValueError('두 경로 OCR은 검증된 GPU 0을 선택해 주세요.')
    device='gpu:0';verified=verify_adapter();started=time.perf_counter()
    deployment=ROOT/'data/experiments/ocr_radu_dual_route_20261113_01/deployment.json'
    if not deployment.is_file():raise RuntimeError('두 경로 OCR의 학습·검증 binding이 아직 준비되지 않았습니다.')
    with tempfile.TemporaryDirectory(prefix='radu-dual-ocr-') as folder:
        work=pathlib.Path(folder);source=work/'input.png'
        if image.ndim==3 and image.shape[2]==3:Image.fromarray(image[:,:,::-1]).save(source)
        elif image.ndim==2:Image.fromarray(image).convert('RGB').save(source)
        else:raise ValueError('BGR 또는 회색조 이미지가 필요합니다.')
        det=work/'detections.json';run_stage('detect',source,det,device,work);d=json.loads(det.read_text())
        if d['regions']:
            output=work/'recognition.json';python=pathlib.Path(os.environ.get('RADU_LORA_PYTHON',str(ROOT/'.cache/venvs/ocr-vl-lora/bin/python')))
            command=[str(python),'-m','src.cv.ocr_dual_route_worker',str(source),str(output),'--detections',str(det),'--device',device]
            env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
            with (work/'recognition.log').open('w') as log:
                try:r=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
                except subprocess.TimeoutExpired as e:raise TimeoutError('두 경로 OCR 인식 시간이 초과됐습니다.') from e
            if not output.is_file():raise RuntimeError('두 경로 OCR worker가 결과를 반환하지 않았습니다: '+(work/'recognition.log').read_text(errors='replace')[-1600:])
            result=json.loads(output.read_text())
            if r.returncode or result.get('error'):raise RuntimeError(result.get('error','두 경로 OCR worker 실패'))
            lines,metadata=result['lines'],result['metadata']
        else:lines=[];metadata=dict(device='cuda:0',recognizer_called=False,generation_limit_regions=[],route_counts=dict(parent=0,diagram=0))
        metadata.update(verified,model='PaddleOCR-VL-1.6 + bbox-routed LoRA',detector='PP-OCRv6_medium_det',detector_seconds=d['seconds'],detection_count=len(d['regions']),
            input_sha256=d['image_sha256'],width=d['width'],height=d['height'],total_seconds=time.perf_counter()-started,
            engine_path='existing PP-OCRv6 detection + fixed crop + conservative bbox route',text_task='plain-text OCR; LaTeX structure not validated')
        return '\n'.join(r['text'] for r in lines),dict(lines=lines,metadata=metadata,model=metadata['model'],device='cuda:0',resource='local')
