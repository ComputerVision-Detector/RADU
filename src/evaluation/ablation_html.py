"""Generate an offline-readable visual comparison from frozen ablation artifacts."""
from __future__ import annotations
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import quote


def differences(before, after):
    left, right = [], []
    for tag, a, b, c, d in difflib.SequenceMatcher(None, before, after, autojunk=False).get_opcodes():
        if a != b:
            left.append({'text':before[a:b], 'changed':tag != 'equal'})
        if c != d:
            right.append({'text':after[c:d], 'changed':tag != 'equal'})
    return left, right


def bbox(raw):
    if not raw:
        return None
    if isinstance(raw[0], (list,tuple)):
        return [min(p[0] for p in raw),min(p[1] for p in raw),max(p[0] for p in raw),max(p[1] for p in raw)]
    return raw if len(raw)==4 else None


def build_payload(manifest_path, results_path, output, references=None):
    manifest_path, results_path, output = map(Path, (manifest_path,results_path,output))
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    results=json.loads(results_path.read_text(encoding='utf-8'))
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != results['manifest_sha256']:
        raise ValueError('Results do not match the input manifest')
    summary_path=results_path.parent/'summary.json'
    summary=json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.exists() else {}
    reference_path=Path(references) if references else manifest_path.parent/'references.json'
    reference_rows={}
    if references or (summary.get('accuracy_available') and reference_path.exists()):
        from src.evaluation.ablation import load_references
        load_references(reference_path,results['manifest_sha256'],{item['id'] for item in manifest['samples']})
        if (summary.get('reference_provenance') or {}).get('sha256') != hashlib.sha256(reference_path.read_bytes()).hexdigest():
            raise ValueError('Reference file changed after scoring; rerun score before rendering HTML')
        reference_rows={item['id']:item for item in json.loads(reference_path.read_text(encoding='utf-8'))['samples'] if item.get('status')=='verified'}
    score_path=results_path.parent/'sample_scores.json'
    score_rows={item['id']:item for item in json.loads(score_path.read_text(encoding='utf-8'))['samples']} if score_path.exists() else {}
    indexed={sample['id']:sample for sample in results['samples']}
    payload={'dataset':manifest['dataset'],'revision':manifest['revision'],'seed':manifest['seed'],
             'population':manifest['population_rows'],'summary':summary,
             'environment':results.get('environment',{}),'samples':[],
             'csv':quote(os.path.relpath(results_path.parent/'comparison.csv',output.parent).replace(os.sep,'/')),
             'accuracy_plot':quote(os.path.relpath(results_path.parent/'accuracy_comparison.png',output.parent).replace(os.sep,'/'))}
    for sample in manifest['samples']:
        result=indexed.get(sample['id'])
        if result is None:
            continue
        folder=results_path.parent/sample['id']
        ocr=json.loads((folder/'ocr.json').read_text(encoding='utf-8')) if (folder/'ocr.json').exists() else {}
        review=json.loads((folder/'review.json').read_text(encoding='utf-8')) if (folder/'review.json').exists() and result.get('qwen',{}).get('status')=='ok' else {}
        edits={item['id']:item for item in review.get('vector_edits',[])
               if isinstance(item,dict) and isinstance(item.get('id'),str) and isinstance(item.get('text'),str)}
        rows=[]
        for number, line in enumerate(ocr.get('lines',[]),1):
            line_id=line.get('id') or f'ocr-text-{number:04d}'
            before=str(line.get('text',''))
            after=edits.get(line_id,{}).get('text',before) if result.get('qwen',{}).get('status')=='ok' else None
            left,right=differences(before,after) if after is not None else ([{'text':before,'changed':False}],[])
            rows.append({'id':line_id,'number':number,'before':before,'after':after,
                         'changed':after is not None and before!=after,'bbox':bbox(line.get('bbox')),
                         'left':left,'right':right,'uncertain':bool(edits.get(line_id,{}).get('uncertain',False))})
        image_path=manifest_path.parent/sample['image']
        if hashlib.sha256(image_path.read_bytes()).hexdigest()!=sample['sha256']:
            raise ValueError(f'Input image changed: {sample["id"]}')
        payload['samples'].append({'id':sample['id'],'image':quote(os.path.relpath(image_path,output.parent).replace(os.sep,'/')),
            'width':sample['width'],'height':sample['height'],'shard':sample['shard'],'row':sample['row'],
            'rows':rows,'paddle':result.get('paddle',{}),'qwen':result.get('qwen',{}),
            'total_seconds':result.get('hybrid_total_seconds'),'changes':result.get('changes',{}),
            'baseline_text':result.get('baseline_text',''),'hybrid_text':result.get('hybrid_text'),
            'uncertainties':review.get('uncertainties',[]), 'reference':reference_rows.get(sample['id']),
            'scores':score_rows.get(sample['id']) if sample['id'] in reference_rows else None})
    return payload


def generate(manifest, results, output, references=None):
    output=Path(output).resolve()
    data=build_payload(Path(manifest).resolve(),Path(results).resolve(),output,references)
    template=(Path(__file__).parent/'templates/ablation.html').read_text(encoding='utf-8')
    # OCR/model text remains inert even when it contains HTML or script delimiters.
    encoded=json.dumps(data,ensure_ascii=False,allow_nan=False).replace('&','\\u0026').replace('<','\\u003c').replace('>','\\u003e')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(template.replace('__REPORT_DATA__',encoded),encoding='utf-8')
    print(f'HTML comparison: {output} ({len(data["samples"])} samples)')
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--references',type=Path)
    args=parser.parse_args()
    generate(args.manifest,args.results,args.output or args.manifest.parent/'index.html',args.references)
