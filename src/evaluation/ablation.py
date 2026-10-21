"""Paired PaddleOCR-VL vs PaddleOCR-VL + Qwen evaluation on fixed images.

Usage: python -m src.evaluation.ablation prepare|run|score --help
No reference labels => no accuracy metrics. Never treat agreement/change rate as accuracy.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import signal
import statistics
import subprocess
import sys
import time

from src.evaluation.ocr_metrics import NORMALIZATION, aggregate, apply_review, score

ROOT = Path(__file__).resolve().parents[2]


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(args):
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as parquet
    from PIL import ImageOps
    from src.evaluation.benchmark_ocr_visual import extract_image
    out = args.output.resolve()
    if (out/'manifest.json').exists():
        raise ValueError('Manifest exists; select another output directory to preserve the fixed evaluation set')
    info = HfApi().dataset_info(args.dataset, revision=args.revision, timeout=30)
    files = sorted(item.rfilename for item in info.siblings
                   if item.rfilename.startswith(f'data/{args.split}-') and item.rfilename.endswith('.parquet'))
    if not files:
        raise ValueError('Preparation currently supports HF data/<split>-*.parquet repositories')
    shards = []
    total = 0
    for name in files:
        path = hf_hub_download(args.dataset, name, repo_type='dataset', revision=info.sha)
        reader = parquet.ParquetFile(path)
        shards.append((name, reader, total, reader.metadata.num_rows))
        total += reader.metadata.num_rows
    if not 1 <= args.samples <= total:
        raise ValueError(f'--samples must be between 1 and {total}')
    selected = random.Random(args.seed).sample(range(total), args.samples)
    (out/'images').mkdir(parents=True, exist_ok=True)
    manifest = {'version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
                'dataset': args.dataset, 'revision': info.sha, 'split': args.split,
                'seed': args.seed, 'population_rows': total,
                'selection': 'uniform global row sample without replacement, frozen before inference', 'samples': []}
    cached = {}
    for global_row in selected:
        name, reader, start, _ = next(item for item in shards if item[2] <= global_row < item[2]+item[3])
        row = global_row-start
        offset = row
        for group in range(reader.num_row_groups):
            count = reader.metadata.row_group(group).num_rows
            if offset < count:
                key = (name, group)
                if key not in cached:
                    cached[key] = reader.read_row_group(group)
                record = cached[key].slice(offset, 1).to_pylist()[0]
                _, image = extract_image(record)
                image = ImageOps.exif_transpose(image).convert('RGB')
                break
            offset -= count
        sample_id = f'hf-{global_row:04d}'
        filename = f'images/{sample_id}.png'
        image.save(out/filename)
        manifest['samples'].append({'id': sample_id, 'image': filename, 'sha256': digest(out/filename),
             'width': image.width, 'height': image.height, 'shard': name, 'row': row, 'global_row': global_row})
    save_json(out/'manifest.json', manifest)
    references = {'version': 1, 'manifest_sha256': digest(out/'manifest.json'),
        'instructions': 'Transcribe images independently of model predictions. Preserve case, punctuation and reading order. Set status=verified only after human checking. Blank text is not a label.',
        'samples': [{'id': sample['id'], 'text': None, 'status': 'unlabeled', 'annotator': '', 'notes': ''}
                    for sample in manifest['samples']]}
    save_json(out/'references.template.json', references)
    (out/'ANNOTATION.md').write_text(
        '# 전사 정답 작성\n\nimages 폴더의 원본만 보고 references.template.json의 text에 전사합니다. '
        '모델 출력은 정답 작성 중 보지 않습니다. 한국어·영문·숫자·문장부호와 읽기 순서를 유지하고, '
        '실제 사용한 표기 규칙을 notes에 기록하세요. 확인한 항목만 status를 verified로 바꿉니다. '
        '판독 불가능한 항목은 unlabeled로 남기세요. 초안은 draft로 표시하면 공식 지표에서 제외합니다.\n\n'
        '표·화살표 구조 평가는 전사 문자열만으로 평가하지 않습니다. 별도 셀/연결 정답이 필요합니다.\n', encoding='utf-8')
    print(json.dumps({'manifest': str(out/'manifest.json'), 'population': total, 'selected': selected}, ensure_ascii=False), flush=True)


def gpu_used():
    try:
        result = subprocess.run(['nvidia-smi', '--id=0', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, timeout=5, check=True)
        return float(result.stdout.strip().splitlines()[0])
    except (OSError, ValueError, subprocess.SubprocessError, IndexError):
        return None


def run_stage(command, log_path, timeout):
    start_memory = gpu_used()
    peak = start_memory
    started = time.perf_counter()
    status, error = 'ok', None
    with log_path.open('wb') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                if time.perf_counter()-started > timeout:
                    raise TimeoutError(f'Stage exceeded {timeout} seconds')
                memory = gpu_used()
                if memory is not None:
                    peak = memory if peak is None else max(peak, memory)
                time.sleep(.5)
            if process.returncode:
                status, error = 'failed', f'Exit code {process.returncode}; see {log_path.name}'
        except BaseException as exc:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            if not isinstance(exc, Exception):
                raise
            status, error = 'failed', f'{type(exc).__name__}: {exc}'
    return {'status': status, 'error': error, 'wall_seconds': time.perf_counter()-started,
            'gpu_start_used_mib': start_memory, 'gpu_peak_used_mib': peak,
            'gpu_incremental_peak_mib': max(0, peak-start_memory) if peak is not None and start_memory is not None else None,
            'gpu_sampling_interval_seconds': .5}


def environment():
    from src.hybrid.pipeline import PROMPT
    versions = {}
    for package in ['torch', 'transformers', 'paddlepaddle-gpu', 'paddleocr', 'paddlex', 'rapidfuzz']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
                            capture_output=True, text=True, timeout=10)
    source_paths = ['src/hybrid/pipeline.py', 'src/hybrid/document.py', 'src/cv/ocr_engine.py',
                    'src/evaluation/paddleocr_vl_worker.py', 'src/evaluation/ablation_ocr_worker.py',
                    'src/evaluation/ablation.py', 'src/evaluation/ocr_metrics.py',
                    'src/cv/detector.py', 'src/cv/structure_rules.py']
    return {'python': sys.version, 'platform': platform.platform(), 'packages': versions,
            'gpu': result.stdout.strip(), 'prompt_sha256': hashlib.sha256(PROMPT.encode()).hexdigest(),
            'source_sha256': {path:digest(ROOT/path) for path in source_paths},
            'settings': {name:os.getenv(name, default) for name,default in [
                ('HYBRIDOCR_QWEN_MODEL', str(ROOT/'models/vision_language/Qwen2.5-VL-7B-Instruct')),
                ('HYBRIDOCR_MAX_TOKENS','6144'), ('HYBRIDOCR_MAX_PIXELS',str(2048*28*28)),
                ('RADU_VL_MAX_NEW_TOKENS','2048')]},
            'decoding': 'Qwen do_sample=False; one image per request; process/model reload for every stage',
            'memory_note': 'nvidia-smi samples whole GPU, including desktop/other processes; sampled peak may miss short spikes. Increment is relative to pre-stage device usage.'}


def run(args):
    import cv2
    from src.hybrid.document import layout_candidates
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    out = args.output.resolve() if args.output else manifest_path.parent/'results'
    out.mkdir(parents=True, exist_ok=True)
    path = out/'results.json'
    if path.exists() and not args.resume:
        raise ValueError('Results exist; use --resume to reuse completed sample artifacts or choose another output')
    report = json.loads(path.read_text(encoding='utf-8')) if args.resume and path.exists() else {
        'version': 1, 'started_at': datetime.now(timezone.utc).isoformat(), 'manifest_sha256': digest(manifest_path),
        'environment': environment(), 'arms': ['paddle_only','paddle_plus_qwen'], 'samples': []}
    if report['manifest_sha256'] != digest(manifest_path):
        raise ValueError('Manifest differs from the existing run')
    if args.resume and report['environment']['source_sha256'] != environment()['source_sha256']:
        raise ValueError('Code changed since this run. Start a new results directory to avoid mixing configurations')
    completed = {item['id'] for item in report['samples']}
    for index, sample in enumerate(manifest['samples'], 1):
        if sample['id'] in completed:
            continue
        image = (manifest_path.parent/sample['image']).resolve()
        if digest(image) != sample['sha256']:
            raise ValueError(f'Image changed: {sample["id"]}')
        folder = out/sample['id']
        folder.mkdir(exist_ok=True)
        result = {'id': sample['id'], 'image_sha256': sample['sha256']}
        print(f'[{index}/{len(manifest["samples"])}] {sample["id"]}: PaddleOCR-VL', flush=True)
        stage = run_stage([sys.executable,'-m','src.evaluation.ablation_ocr_worker',str(image),str(folder/'ocr.json')],
                          folder/'ocr.log', args.ocr_timeout)
        result['paddle'] = stage
        if stage['status'] == 'ok':
            try:
                ocr = json.loads((folder/'ocr.json').read_text(encoding='utf-8'))
                result['ocr_metadata'] = ocr.get('metadata', {})
                result['baseline_text'] = '\n'.join(line.get('text','') for line in ocr['lines'])
                started = time.perf_counter()
                ocr['layout'] = layout_candidates(cv2.imread(str(image)), ocr['lines'])
                result['layout_seconds'] = time.perf_counter()-started
                save_json(folder/'ocr.json', ocr)
                print(f'[{index}/{len(manifest["samples"])}] {sample["id"]}: Qwen review', flush=True)
                review_stage = run_stage([sys.executable,'-m','src.hybrid.pipeline','review',str(image),str(folder)],
                                         folder/'qwen.log', args.qwen_timeout)
                result['qwen'] = review_stage
                if review_stage['status'] == 'ok':
                    review = json.loads((folder/'review.json').read_text(encoding='utf-8'))
                    baseline, corrected, change = apply_review(ocr, review)
                    result.update(baseline_text=baseline, hybrid_text=corrected, changes=change,
                                  qwen_metadata=review.get('metadata', {}), qwen_model=review.get('model'))
                    (folder/'paddle.txt').write_text(baseline, encoding='utf-8')
                    (folder/'hybrid.txt').write_text(corrected, encoding='utf-8')
                result['hybrid_total_seconds'] = stage['wall_seconds']+result['layout_seconds']+review_stage['wall_seconds']
            except Exception as exc:
                result['qwen'] = {'status':'failed','error':f'{type(exc).__name__}: {exc}'}
        else:
            result['qwen'] = {'status':'not_run','error':'Baseline failed'}
        report['samples'].append(result)
        report['updated_at'] = datetime.now(timezone.utc).isoformat()
        save_json(path, report)
        write_report(manifest_path, path, args.references)
        print(f'[{index}/{len(manifest["samples"])}] saved; OCR={stage["status"]}, Qwen={result["qwen"]["status"]}', flush=True)
    report['completed_at'] = datetime.now(timezone.utc).isoformat()
    save_json(path, report)
    write_report(manifest_path, path, args.references)
    print(f'Report: {out / "REPORT.md"}', flush=True)


def distribution(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    position = .95*(len(values)-1)
    lo, hi = math.floor(position), math.ceil(position)
    return {'n':len(values), 'mean':statistics.mean(values), 'median':statistics.median(values),
            'p95':values[lo]+(values[hi]-values[lo])*(position-lo), 'min':values[0], 'max':values[-1]}


def load_references(path, manifest_sha256, ids):
    if path is None:
        return {}, {'verified':0,'draft':0,'unlabeled':len(ids)}
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('manifest_sha256') != manifest_sha256:
        raise ValueError('Reference manifest hash differs')
    counts = Counter()
    references, seen = {}, set()
    for item in data['samples']:
        if item['id'] not in ids or item['id'] in seen:
            raise ValueError('Unknown or duplicate reference ID')
        seen.add(item['id'])
        status = item.get('status','unlabeled')
        counts[status] += 1
        if status == 'verified':
            if not isinstance(item.get('text'),str) or not item['text'].strip():
                raise ValueError('Verified references need non-empty text')
            references[item['id']] = item['text']
    counts['unlabeled'] += len(ids-seen)
    return references, dict(counts)


def write_report(manifest_path, result_path, reference_path=None):
    import csv
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    report = json.loads(Path(result_path).read_text(encoding='utf-8'))
    ids = {s['id'] for s in manifest['samples']}
    refs, reference_counts = load_references(reference_path, digest(manifest_path), ids)
    reference_document = json.loads(Path(reference_path).read_text(encoding='utf-8')) if reference_path else {}
    samples = report['samples']
    pairs = [s for s in samples if s['paddle']['status']=='ok' and s['qwen']['status']=='ok']
    baseline_ok = [s for s in samples if s['paddle']['status']=='ok']
    baseline_scores, hybrid_scores, fallback_scores, all_baseline_scores = [], [], [], []
    rows = []
    detailed_scores = []
    for sample in samples:
        paired = sample in pairs
        ref = refs.get(sample['id'])
        before = score(ref,sample['baseline_text']) if paired and ref is not None else None
        after = score(ref,sample['hybrid_text']) if paired and ref is not None else None
        if before is not None:
            baseline_scores.append(before); hybrid_scores.append(after)
        if ref is not None and sample['paddle']['status']=='ok':
            all_baseline_scores.append(score(ref,sample['baseline_text']))
            effective = sample.get('hybrid_text',sample['baseline_text']) if paired else sample['baseline_text']
            fallback_scores.append(score(ref,effective))
        detailed_scores.append({'id':sample['id'], 'reference_verified':ref is not None, 'paddle':before, 'hybrid':after})
        rows.append({'id':sample['id'],'paddle_status':sample['paddle']['status'],'qwen_status':sample['qwen']['status'],
             'paddle_seconds':sample['paddle'].get('wall_seconds'), 'qwen_seconds':sample['qwen'].get('wall_seconds'),
             'hybrid_total_seconds':sample.get('hybrid_total_seconds'),
             'paddle_peak_gpu_mib':sample['paddle'].get('gpu_peak_used_mib'),
             'qwen_peak_gpu_mib':sample['qwen'].get('gpu_peak_used_mib'),
             'baseline_lines':sample.get('changes',{}).get('baseline_lines'),
             'changed_lines':sample.get('changes',{}).get('changed_lines'),
             'changed_line_rate':sample.get('changes',{}).get('changed_line_rate'),
             'text_change_ratio':sample.get('changes',{}).get('text_change_ratio'),
             'reference_verified':ref is not None,'baseline_cer':before['cer'] if before else None,
             'hybrid_cer':after['cer'] if after else None,'baseline_wer':before['wer'] if before else None,
             'hybrid_wer':after['wer'] if after else None,
             'baseline_cer_no_space':before['cer_no_space'] if before else None,
             'hybrid_cer_no_space':after['cer_no_space'] if after else None,
             'cer_change_pp':100*(after['cer']-before['cer']) if before and after else None})
    summary = {'requested_samples':len(manifest['samples']), 'finished_samples':len(samples),
        'baseline_success':len(baseline_ok),'paired_success':len(pairs),
        'baseline_failures':sum(s['paddle']['status']!='ok' for s in samples),
        'qwen_failures':sum(s['qwen']['status']=='failed' for s in samples),
        'reference_counts':reference_counts,'normalization':NORMALIZATION,
        'paddle_seconds_all_success':distribution([s['paddle']['wall_seconds'] for s in baseline_ok]),
        'paddle_seconds_paired':distribution([s['paddle']['wall_seconds'] for s in pairs]),
        'qwen_seconds_paired':distribution([s['qwen']['wall_seconds'] for s in pairs]),
        'hybrid_total_seconds_paired':distribution([s['hybrid_total_seconds'] for s in pairs]),
        'paddle_gpu_mib_paired':distribution([s['paddle'].get('gpu_peak_used_mib') for s in pairs]),
        'qwen_gpu_mib_paired':distribution([s['qwen'].get('gpu_peak_used_mib') for s in pairs]),
        'changed_lines':sum(s['changes']['changed_lines'] for s in pairs),
        'baseline_lines':sum(s['changes']['baseline_lines'] for s in pairs),
        'reference_paired_baseline':aggregate(baseline_scores), 'reference_paired_hybrid':aggregate(hybrid_scores),
        'reference_baseline_all_success':aggregate(all_baseline_scores),
        'reference_hybrid_with_baseline_fallback':aggregate(fallback_scores),
        'accuracy_available':bool(baseline_scores),
        'limitations':[('Provided user transcriptions are compared verbatim with unmasked full-page predictions; omitted regions and notation conventions can affect errors.' if refs else 'No ground truth => CER/WER/accuracy are not computed; change rate is not accuracy.'),
          'Five-image pilot does not establish population-level accuracy or latency.',
          'Timing includes fresh process/model initialization per request; not steady-state throughput.',
          'Hybrid reuses identical OCR lines/order; correction cannot recover text with no OCR anchor.',
          'Both arms share OCR geometry; table/arrow structure accuracy requires separate ground truth.',
          'GPU sampling includes desktop/other workloads and can miss brief peaks.']}
    summary['reference_provenance'] = {'file':Path(reference_path).name, 'sha256':digest(reference_path),
        'source':reference_document.get('source'), 'id_mapping':reference_document.get('id_mapping', {}),
        'scoring_scope':reference_document.get('scoring_scope'),
        'notes':{item['id']:item.get('notes') for item in reference_document.get('samples',[])}} if reference_path else None
    summary['scoring_source_sha256'] = {'ablation.py':digest(Path(__file__)), 'ocr_metrics.py':digest(Path(__file__).with_name('ocr_metrics.py'))}
    summary['cer_sample_outcomes'] = {'improved':sum(b['cer']<a['cer'] for a,b in zip(baseline_scores,hybrid_scores)),
        'worsened':sum(b['cer']>a['cer'] for a,b in zip(baseline_scores,hybrid_scores)),
        'unchanged':sum(b['cer']==a['cer'] for a,b in zip(baseline_scores,hybrid_scores))}
    summary['baseline_failure_rate'] = summary['baseline_failures']/len(samples) if samples else None
    summary['qwen_failure_rate_given_ocr_success'] = summary['qwen_failures']/len(baseline_ok) if baseline_ok else None
    summary['hybrid_failure_rate'] = 1-len(pairs)/len(samples) if samples else None
    count=summary['baseline_lines']
    summary['changed_line_rate'] = summary['changed_lines']/count if count else None
    out = Path(result_path).parent
    save_json(out/'summary.json', summary)
    save_json(out/'sample_scores.json', {'normalization':NORMALIZATION, 'references_sha256':digest(reference_path) if reference_path else None, 'samples':detailed_scores})
    with (out/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        if rows:
            writer=csv.DictWriter(stream,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    def number(value):
        return '—' if value is None else f'{value:.2f}'
    text = ['# PaddleOCR-VL 단독 vs Qwen 검수 비교', '',
        f'- 데이터셋: `{manifest["dataset"]}` / `{manifest["split"]}`, revision `{manifest["revision"]}`',
        f'- 고정 표본: {len(manifest["samples"])}장 / 전체 {manifest["population_rows"]}장, seed {manifest["seed"]}',
        f'- 실행 완료 {len(samples)}장, OCR 성공 {len(baseline_ok)}장, 두 조건 성공 {len(pairs)}장',
        f'- 정답 상태: {reference_counts}. 사람이 확인한 정답만 정확도 계산에 사용합니다.', '',
        ('**사용자 제공 전사 정답으로 재채점했습니다. CER/WER는 낮을수록 좋으며, 제외 영역·표기 기준 차이를 함께 확인하세요.**' if refs else '**정답이 없으면 정확도 개선 여부는 판단할 수 없습니다. 교정률은 출력 변화량입니다.**'), '',
        '|표본|OCR(s)|Qwen 추가(s)|Hybrid 합계(s)|교정 줄/전체 줄|OCR peak MiB|Qwen peak MiB|상태|',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for row in rows:
        text.append('|'+ '|'.join([row['id'],number(row['paddle_seconds']),number(row['qwen_seconds']),number(row['hybrid_total_seconds']),
            f'{row["changed_lines"]}/{row["baseline_lines"]}',number(row['paddle_peak_gpu_mib']),number(row['qwen_peak_gpu_mib']),row['qwen_status']])+'|')
    text += ['', '## 집계', '', '|지표|Paddle 단독|Paddle + Qwen|', '|---|---:|---:|']
    for title, left, right in [('평균 응답 시간(s)',summary['paddle_seconds_paired'],summary['hybrid_total_seconds_paired']),
                               ('p95 응답 시간(s)',summary['paddle_seconds_paired'],summary['hybrid_total_seconds_paired'])]:
        field = 'p95' if title.startswith('p95') else 'mean'
        text.append(f'|{title}|{number(left[field] if left else None)}|{number(right[field] if right else None)}|')
    for metric in ['cer_micro','cer_no_space_micro','wer_micro']:
        a,b=summary['reference_paired_baseline'],summary['reference_paired_hybrid']
        text.append(f'|{metric} (%)|{number(100*a[metric] if a else None)}|{number(100*b[metric] if b else None)}|')
    text += ['', f'교정된 줄: {summary["changed_lines"]}/{summary["baseline_lines"]}. 정확도를 의미하지 않습니다.', '',
      '## 재현 및 해석', '',
      'OCR는 각 이미지에서 한 번 실행하고 두 조건이 같은 OCR 파일을 공유합니다. 원본 RGB PNG·좌표·읽기 순서를 동일하게 유지하며 Qwen 교정문만 OCR id에 반영합니다. 수동 편집이나 정답은 모델 입력에 포함하지 않습니다.', '',
      '시간은 프로세스·모델 초기화를 포함한 단일 요청 응답 시간입니다. Hybrid 합계는 OCR + 구조 후보 처리 + Qwen입니다. 메모리는 GPU 전체 사용량을 0.5초 간격으로 측정하며 바탕화면 등도 포함합니다.', '',
      'CER/WER는 NFC 정규화·공백 연속 축약 후 계산합니다. 대소문자와 문장부호를 유지합니다. cer_no_space는 공백을 모두 제거합니다. micro는 오류 합/정답 길이 합, macro는 표본별 비율 평균입니다. 오류율은 1을 넘을 수 있습니다.', '',
      '정답이 생기면 references.template.json을 복사해 text와 status=verified를 작성하고 score 명령만 실행하면 됩니다. 실패 표본을 숨기지 않고 기록하며, 정답이 있는 성공 쌍의 비교와 OCR 원문 fallback 정책 결과를 구분합니다.', '',
      '표·화살표 정확도는 별도 셀·연결 정답이 필요하므로 현재 수치화하지 않습니다. 모델·입력·프롬프트·소스 해시는 results.json에 보존합니다.', '']
    if refs:
        text += ['## 표본별 전사 오류율', '', '|표본|Paddle CER (%)|Hybrid CER (%)|Paddle WER (%)|Hybrid WER (%)|CER 변화(%p)|',
                 '|---|---:|---:|---:|---:|---:|']
        for row in rows:
            if row['baseline_cer'] is not None:
                text.append('|'+ '|'.join([row['id'],number(100*row['baseline_cer']),number(100*row['hybrid_cer']),
                    number(100*row['baseline_wer']),number(100*row['hybrid_wer']),number(row['cer_change_pp'])])+'|')
        a,b=summary['reference_paired_baseline'],summary['reference_paired_hybrid']
        text += ['', f'문자 오류 합: {a["cer_errors"]} → {b["cer_errors"]}, 정답 문자 수 {a["cer_reference_units"]}.',
                 f'CER 표본 결과: {summary["cer_sample_outcomes"]}. 정확도(1−CER)로 환산하지 않습니다.', '',
                 '## 전사 반영 및 해석 범위', '',
                 '원본 모델 예측과 시간 측정은 바꾸지 않고 저장된 결과를 재채점했습니다. 전사 문구·철자·기호·주석은 제공 내용대로 유지했습니다.', '',
                 '포스트잇·괄호 등 주석에서 제외한 영역을 전체 페이지 예측에서 자동으로 제거하지 않았습니다. 그 영역이 인식되면 삽입 오류로 집계될 수 있습니다. 기호 X/×, 부등호, 목록 기호, 대소문자 및 문장부호 차이도 오류에 포함합니다. 영역을 맞춘 평가에는 별도 영역 지정이 필요합니다.', '']
        if reference_document.get('id_mapping'):
            text += ['제공 ID의 내용 대조에 따른 매핑: '+json.dumps(reference_document['id_mapping'],ensure_ascii=False), '']
        for item in reference_document.get('samples',[]):
            text.append(f'- {item["id"]} / 작성자: {item.get("annotator", "")} / 주석: {item.get("notes") or "없음"}')
        text += ['', 'sample_scores.json에 치환·삭제·삽입 오류 수와 정답 분모를 보존합니다. 원본 추론 당시 소스 해시와 현재 채점 코드 해시는 별도로 기록합니다.', '']
    (out/'REPORT.md').write_text('\n'.join(text),encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare')
    prep.add_argument('--dataset',default='hakuuung/HybridOCR_CustomNotes')
    prep.add_argument('--revision')
    prep.add_argument('--split',default='train')
    prep.add_argument('--samples',type=int,default=5)
    prep.add_argument('--seed',type=int,default=42)
    prep.add_argument('--output',type=Path,required=True)
    infer=sub.add_parser('run')
    infer.add_argument('--manifest',type=Path,required=True)
    infer.add_argument('--output',type=Path)
    infer.add_argument('--references',type=Path)
    infer.add_argument('--ocr-timeout',type=int,default=180)
    infer.add_argument('--qwen-timeout',type=int,default=900)
    infer.add_argument('--resume',action='store_true')
    evaluate=sub.add_parser('score')
    evaluate.add_argument('--manifest',type=Path,required=True)
    evaluate.add_argument('--results',type=Path,required=True)
    evaluate.add_argument('--references',type=Path)
    args=parser.parse_args()
    if args.command=='prepare':
        prepare(args)
    elif args.command=='run':
        run(args)
    else:
        print(json.dumps(write_report(args.manifest,args.results,args.references),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
