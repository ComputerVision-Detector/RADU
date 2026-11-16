"""Reproduce the bounded experiment into a NEW output root, never overwrite."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/ocr_v2_korean_ctc.json');p.add_argument('--output',required=True);p.add_argument('--with-vl',action='store_true');a=p.parse_args()
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    config=json.loads(Path(a.config).read_text());config.update(prepared=str(root/'prepared'),trained=str(root/'train'))
    path=root/'effective-config.json';path.write_text(json.dumps(config,indent=2)+'\n')
    def run(module,args,log):
        with (root/log).open('x') as f:
            command=[sys.executable,'-m',module,'--config',str(path),*args]
            print(' '.join(command),flush=True)
            subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
    for stage in ['prepare','smoke','train','resume-check']:
        run('src.training.ocr_finetune',['--stage',stage,'--output',str(root/stage if stage!='prepare' else root/'prepared')],stage+'.log')
    # The selection receipt exists before any test prediction. No tuner is run.
    from src.ocr_data.audit import sha256
    (root/'final-lock.json').write_text(json.dumps(dict(config_sha256=sha256(path),selection=json.loads((root/'train/selection.json').read_text()),selected_checkpoint_sha256=sha256(root/'train/best.pdparams'),test_used_for_selection=False),indent=2))
    for split in ['validation','test']:
        run('src.evaluation.ocr_finetune_page',['--split',split,'--stage','detect','--output',str(root/f'det-{split}')],f'det-{split}.log')
        run('src.evaluation.ocr_finetune_page',['--split',split,'--stage','recognize','--detections',str(root/f'det-{split}/detections.json'),'--output',str(root/f'page-{split}')],f'page-{split}.log')
        if split=='validation':continue
        run('src.training.ocr_finetune',['--stage','final','--output',str(root/'final')],'final.log')
    if a.with_vl:run('src.evaluation.ocr_v2_vl_baseline',['--output',str(root/'vl-validation'),'--timeout','90'],'vl-validation.log')

if __name__=='__main__':main()
