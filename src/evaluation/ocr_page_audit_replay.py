"""Revalidate frozen visual decisions and cached inference in a new exclusive run.

This replays an existing AI audit; it does not certify new human visual review.
No GPU, optimizer, threshold selection or model inference is invoked.
"""
import argparse
import contextlib
import io
import platform
import shutil
import subprocess
import time
from pathlib import Path
from collections import Counter

from src.evaluation import ocr_page_audit as a


def check_hashes(root, hashes):
    for name, expected in hashes.items():
        path = Path(root) / name
        if not path.is_file() or a.sha256(path) != expected:
            raise ValueError(f'cached artifact drift: {path}')


def main(output):
    start = time.perf_counter()
    source = a.ROOT
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    manifest = a.read(source / 'completion-manifest.json')
    check_hashes(source, manifest['artifacts'])
    a.verify_lock()
    frozen = a.read(source / 'frozen-fp201.json')
    review = a.read(source / 'fp201-audit-v2.json')
    a.validate_reviews([r['prediction_id'] for r in frozen], review)
    if len(frozen) != 201 or len({r['prediction_id'] for r in frozen}) != 201:
        raise ValueError('frozen FP identity invalid')
    a.ROOT = out
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        a.prepare()
        a.audit()
        from src.evaluation import ocr_page_audit_revision as revision
        revision.ROOT = out
        revision.main()
    metadata_differences = []
    for name in ['audited-gt-v1.json', 'audited-gt-v2.json',
                 'fp201-audit-v2.json', 'frozen-fp201.json', 'legacy-reproduced.json']:
        before, after = a.read(source / name), a.read(out / name)
        if name == 'audited-gt-v2.json' and before.get('revision') != after.get('revision'):
            metadata_differences.append(dict(file=name, field='revision', before=before['revision'], after=after['revision']))
            before = dict(before, revision=after['revision'])
        if name == 'fp201-audit-v2.json':
            for old, new in zip(before, after):
                if old['reason'] != new['reason']:
                    metadata_differences.append(dict(file=name, prediction_id=old['prediction_id'], field='reason', before=old['reason'], after=new['reason']))
                    old['reason'] = new['reason']
        if before != after:
            raise ValueError(f'replay mismatch: {name}')
    a.write_new(out / 'replay-metadata-differences.json', metadata_differences)
    # Preserve original batch membership and crop identity, including source paths.
    for name in ['M.json', 'M-excluded.json', 'P1-rows.json', 'inference-lock.json']:
        shutil.copyfile(source / name, out / name)
    shutil.copytree(source / 'inference', out / 'inference')
    for name in ['M.json', 'P1-rows.json']:
        for row in a.read(out / name):
            if a.sha256(row['image']) != row['crop_sha256']:
                raise ValueError('GT crop drift')
            if 'pred_image' in row and a.sha256(row['pred_image']) != row['pred_crop_sha256']:
                raise ValueError('prediction crop drift')
    from src.evaluation import ocr_page_audit_report as report
    report.ROOT = out
    with contextlib.redirect_stdout(log):
        report.summarize_results()
    results = a.read(out / 'summary/results.json')
    # Recompute, rather than trust, stored audited-scope scores.
    fresh = a.score_scope(a.read(out / 'audited-gt-v2.json')['pages'], a.read(a.PRED))
    if fresh != results['new_GT_with_ignore']:
        raise ValueError('scope scoring mismatch')
    a.write_new(out / 'fresh-scope-score.json', fresh)
    a.write_new(out / 'cache-provenance.json', dict(
        source=str(source), source_manifest_sha256=a.sha256(source / 'completion-manifest.json'),
        validated_cached_files=len(manifest['artifacts']), visual_review='replayed frozen AI decisions; no new independent review',
        new_inference_inputs=0, optimizer_steps=0, counts=dict(Counter(r['primary'] for r in review))))
    modules = ['tests.test_ocr_page_audit_replay', 'tests.test_ocr_page_audit',
               'tests.test_ocr_clean_vision', 'tests.test_ocr_train_audit',
               'tests.test_ocr_lora_round2', 'tests.test_ocr_vl_lora',
               'tests.test_ocr_v2', 'tests.test_ocr_finetune', 'tests.test_ocr_finetune_page']
    import sys
    command = [sys.executable, '-m', 'unittest', *modules, '-v']
    test = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out / 'tests.log').write_text(test.stdout)
    (out / 'execution.log').write_text(log.getvalue())
    a.verify_lock()
    a.write_new(out / 'execution-receipt.json', dict(
        seconds=time.perf_counter()-start, python=platform.python_version(),
        test_command=command, test_exit_code=test.returncode,
        git_head=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        new_inference_inputs=0, source_audit=str(source)))
    a.write_new(out / 'manifest.json', {str(p.relative_to(out)):a.sha256(p)
        for p in sorted(out.rglob('*')) if p.is_file()})
    if test.returncode:
        raise RuntimeError('tests failed; see tests.log')
    print(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    main(parser.parse_args().output)
