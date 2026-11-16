"""Freeze read-only GitHub V2 bytes and image provenance, never live.json."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
from src.ocr_data.audit import sha256


def git(*args):
    return subprocess.check_output(['git',*args])


def snapshot(output, commit=None):
    branch='refs/heads/RADU/labeling-data'
    remote=git('ls-remote','origin',branch).decode().strip().split()[0]
    revision=commit or remote
    if not re.fullmatch('[0-9a-f]{40}',revision): raise ValueError('full commit required')
    if subprocess.run(['git','cat-file','-e',revision+'^{commit}'],capture_output=True).returncode:
        git('fetch','--no-tags','origin',revision)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    files={}
    for name in ['live-v2.json','dataset.catalog.json','assets.manifest.json','legacy_images.manifest.json']:
        ref=revision if name=='live-v2.json' else 'HEAD'
        content=git('show',f'{ref}:labeling/data/{name}') if name!='legacy_images.manifest.json' else git('show','HEAD:data/benchmark/qwen_ablation_pilot/additional_images.manifest.json')
        (output/name).write_bytes(content);files[name]=sha256(output/name)
    data=json.loads((output/'live-v2.json').read_text())
    manifest=dict(schema=1,repository='https://github.com/ComputerVision-Detector/RADU',branch=branch,
                  checked_at=datetime.now(timezone.utc).isoformat(),latest_remote_commit=remote,labels_commit=revision,
                  reference_commit='50aca983da9885d9e7a965c1ef4b05055d47b252',code_commit=git('rev-parse','HEAD').decode().strip(),
                  image_revision=data['revision'],files_sha256=files,source_format='items / bbox_annotation',
                  catalog_source='local HEAD; cross-checked with label provenance during build')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--commit')
    a=p.parse_args();print(json.dumps(snapshot(a.output,a.commit),indent=2))
if __name__=='__main__':main()
