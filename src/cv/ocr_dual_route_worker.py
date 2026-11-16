"""One selected expert per existing bbox; preserve RADU's text/bbox contract."""
from __future__ import annotations
import argparse,hashlib,json,pathlib,sys,time

FROZEN_DEPLOYMENT_SHA256='ada51f815b41e3396a8bcd1bd02a1ca979421807fb95ca2009c0f13d3c57550f'

def _sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as source:
        for chunk in iter(lambda:source.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def verify_deployment(root,experiment):
    """Approve this web release while preserving the original model ledger."""
    root=pathlib.Path(root);experiment=pathlib.Path(experiment);original=experiment/'deployment.json'
    if _sha(original)!=FROZEN_DEPLOYMENT_SHA256:raise ValueError('원래 두 경로 OCR 배포 기록이 변경됐습니다.')
    deploy=json.loads(original.read_text())
    for name,expected in deploy['files'].items():
        if _sha(experiment/name)!=expected:raise ValueError('두 경로 OCR 모델·정책 binding이 변경됐습니다.')
    release=root/'configs/ocr_cv_final.lock.json'
    if release.is_file():
        approved=json.loads(release.read_text())
        if approved.get('schema')!=1 or approved.get('base_deployment_sha256')!=FROZEN_DEPLOYMENT_SHA256:raise ValueError('CV 최종 배포 기록이 원래 모델과 다릅니다.')
        for key in ['engine','parent_sha256','candidate_sha256']:
            if approved.get(key)!=deploy[key]:raise ValueError('CV 최종 배포가 다른 OCR 모델을 지정했습니다.')
        expected_names={str(pathlib.Path(name).relative_to(root)) for name in deploy['installed_files']}
        if set(approved.get('installed_files',{}))!=expected_names:raise ValueError('CV 최종 배포의 설치 파일 범위가 다릅니다.')
        for name,expected in approved['installed_files'].items():
            if _sha(root/name)!=expected:raise ValueError('CV 최종 설치 코드가 승인한 버전과 다릅니다: '+name)
        deploy.update(verification_release=approved.get('release','cv-final'),verification_binding_sha256=_sha(release))
    else:
        for name,expected in deploy.get('installed_files',{}).items():
            if _sha(name)!=expected:raise ValueError('두 경로 OCR 설치 코드가 변경됐습니다.')
        deploy.update(verification_release='legacy',verification_binding_sha256=_sha(original))
    return deploy

def merge_outputs(regions,predictions):
    if len(regions)!=len(predictions):raise ValueError('라우팅 OCR bbox 출력 수가 다릅니다.')
    output=[];limits=[];counts=dict(parent=0,diagram=0)
    for r,p in zip(regions,predictions):
        if r['region_id']!=p['id'] or r['crop_sha256']!=p['crop_sha256']:raise ValueError('라우팅 OCR bbox 순서 또는 crop이 다릅니다.')
        if p.get('error'):raise RuntimeError(f"bbox {p['id']}: {p['error']}")
        if p['expert'] not in counts:raise ValueError('검증하지 않은 OCR expert입니다.')
        counts[p['expert']]+=1
        if p.get('limit'):limits.append(p['id'])
        output.append(dict(id=r['region_id'],text=p['text'],bbox=r['bbox'],crop_sha256=r['crop_sha256'],confidence=None,
            ocr_route=p['expert'],route_score=p.get('score')))
    return output,limits,counts

def run(image,detections,device):
    from src.cv.ocr_lora import ROOT,gpu_device
    experiment=ROOT/'data/experiments/ocr_radu_dual_route_20261113_01';sys.path.insert(0,str(experiment))
    from worker import Engine,guard,sha,read
    deploy=verify_deployment(ROOT,experiment)
    if gpu_device(device)!='cuda:0':raise ValueError('두 경로 OCR은 검증된 GPU 0을 선택해 주세요.')
    data=json.loads(detections.read_text())
    if sha(image)!=data['image_sha256']:raise ValueError('검출·인식 원본이 다릅니다.')
    jobs=[dict(id=r['region_id'],image=r['image'],crop_sha256=r['crop_sha256']) for r in data['regions']]
    guard(True);engine=Engine();pred,timing=engine.run_page(jobs,'router',read('policy.json'));output,limits,counts=merge_outputs(data['regions'],pred)
    return dict(lines=output,metadata=dict(device='cuda:0',recognizer_called=bool(jobs),generation_limit_regions=limits,route_counts=counts,
        route_policy=read('policy.json'),validation_qualified=deploy['validation_qualified'],selected_epoch=deploy['selected_epoch'],
        deployment_release=deploy['verification_release'],release_binding_sha256=deploy['verification_binding_sha256'],
        parent_adapter_sha256=read('binding.json')['parent_sha256'],english_adapter_sha256=read('binding.json')['selected_adapter_sha256'],
        model='PaddleOCR-VL-1.6 + bbox-routed LoRA',text_task='plain-text OCR; LaTeX structure not validated',
        recognizer_seconds=timing['seconds']+engine.load['seconds'],routing_runtime=timing,load=engine.load,
        peak_gpu_bytes=engine.torch.cuda.max_memory_allocated()))

def main():
    p=argparse.ArgumentParser();p.add_argument('image',type=pathlib.Path);p.add_argument('output',type=pathlib.Path);p.add_argument('--detections',type=pathlib.Path,required=True);p.add_argument('--device',required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('기존 출력은 보존합니다.')
    try:result=run(a.image,a.detections,a.device);code=0
    except Exception as e:result=dict(error=f'{type(e).__name__}: {e}');code=1
    a.output.write_text(json.dumps(result,ensure_ascii=False));return code
if __name__=='__main__':raise SystemExit(main())
