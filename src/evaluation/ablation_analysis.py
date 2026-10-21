"""Render completed paired-run plots and line-level differences without changing inference."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path


def render(result_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    result_path=Path(result_path)
    out=result_path.parent
    results=json.loads(result_path.read_text(encoding='utf-8'))
    summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    pairs=[s for s in results['samples'] if s['paddle']['status']=='ok' and s['qwen']['status']=='ok']
    if not pairs:
        raise ValueError('No successful pairs to visualize')
    changes=[]
    for sample in pairs:
        folder=out/sample['id']
        ocr=json.loads((folder/'ocr.json').read_text(encoding='utf-8'))
        review=json.loads((folder/'review.json').read_text(encoding='utf-8'))
        edits={item['id']:item for item in review['vector_edits']
               if isinstance(item,dict) and isinstance(item.get('id'),str) and isinstance(item.get('text'),str)}
        for line in ocr['lines']:
            edit=edits.get(line['id'])
            if edit and line.get('text','') != edit['text']:
                changes.append({'sample':sample['id'],'line_id':line['id'],'paddle':line.get('text',''),
                                'qwen':edit['text'],'model_uncertain':edit.get('uncertain',False)})
    with (out/'line_changes.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['sample','line_id','paddle','qwen','model_uncertain'])
        writer.writeheader();writer.writerows(changes)
    lines=['# 출력 변경 사례', '', '**아래는 모델 간 출력 차이이며, 정답 판정이나 정확도 개선 사례가 아닙니다.**', '']
    for sample in pairs:
        own=[row for row in changes if row['sample']==sample['id']]
        lines += [f'## {sample["id"]}', '', f'변경된 줄 {len(own)}개. 전체 차이는 line_changes.csv에서 확인합니다.', '']
        for row in own[:8]:
            lines += [f'### {row["line_id"]}', '', 'Paddle:', '```text', row['paddle'], '```', '',
                      'Qwen:', '```text', row['qwen'], '```', '']
    (out/'OUTPUT_CHANGES.md').write_text('\n'.join(lines),encoding='utf-8')
    x=np.arange(len(pairs));width=.36
    labels=[s['id'] for s in pairs]
    ocr=np.array([s['paddle']['wall_seconds'] for s in pairs])
    qwen=np.array([s['qwen']['wall_seconds'] for s in pairs])
    layout=np.array([s['layout_seconds'] for s in pairs])
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    axes[0].bar(x-width/2,ocr,width,label='Paddle only',color='#5276b8')
    axes[0].bar(x+width/2,ocr,width,color='#5276b8')
    axes[0].bar(x+width/2,layout,width,bottom=ocr,label='Layout overhead',color='#a3aab6')
    axes[0].bar(x+width/2,qwen,width,bottom=ocr+layout,label='Qwen review',color='#35a187')
    axes[0].set_ylabel('Seconds / request');axes[0].set_title('Response time including model loading')
    axes[0].legend(fontsize=8)
    axes[1].bar(x-width/2,[s['paddle']['gpu_peak_used_mib']/1024 for s in pairs],width,label='Paddle',color='#5276b8')
    axes[1].bar(x+width/2,[s['qwen']['gpu_peak_used_mib']/1024 for s in pairs],width,label='Qwen',color='#35a187')
    axes[1].set_ylabel('GiB, whole GPU');axes[1].set_title('Sampled peak GPU memory (0.5 s)');axes[1].legend(fontsize=8)
    axes[2].bar(x,[s['changes']['changed_line_rate']*100 for s in pairs],color='#ba8d4d')
    axes[2].set_ylabel('Changed OCR lines (%)');axes[2].set_title('Output intervention rate (not accuracy)');axes[2].set_ylim(0,105)
    for ax in axes:
        ax.set_xticks(x,labels,rotation=35);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('PaddleOCR-VL 1.6 vs PaddleOCR-VL + Qwen2.5-VL-7B | fixed pilot samples')
    fig.text(.5,.015,('User-provided reference transcripts are scored separately. GPU memory includes desktop/other processes.' if summary.get('accuracy_available') else 'No reference labels: CER/WER and accuracy improvement are NOT measured. GPU memory includes desktop/other processes.'),ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,.93]);fig.savefig(out/'comparison.png',dpi=160);fig.savefig(out/'comparison.svg');plt.close(fig)
    baseline=summary['paddle_seconds_paired']['mean'];hybrid=summary['hybrid_total_seconds_paired']['mean']
    facts={'successful_pairs':len(pairs),'mean_latency_ratio':hybrid/baseline,'mean_added_seconds':hybrid-baseline,
           'changed_lines':len(changes),
           'paddle_gpu_peak_max_mib': max(s['paddle']['gpu_peak_used_mib'] for s in pairs),
           'hybrid_gpu_peak_max_mib': max(max(s['paddle']['gpu_peak_used_mib'],s['qwen']['gpu_peak_used_mib']) for s in pairs),
           'mean_ocr_inference_seconds': sum(s['ocr_metadata']['inference_seconds'] for s in pairs)/len(pairs),
           'mean_qwen_generation_seconds': sum(s['qwen_metadata']['inference_seconds'] for s in pairs)/len(pairs),
           'accuracy_conclusion':('Measured against user-provided transcripts; omitted-region and notation conventions are not masked.' if summary.get('accuracy_available') else 'not measurable without independently verified references')}
    if summary.get('accuracy_available'):
        a,b=summary['reference_paired_baseline'],summary['reference_paired_hybrid']
        facts['reference_metrics']={'paddle':a,'hybrid':b,'cer_sample_outcomes':summary['cer_sample_outcomes'],
            'cer_reduction_pp':100*(a['cer_micro']-b['cer_micro']), 'wer_reduction_pp':100*(a['wer_micro']-b['wer_micro'])}
        detailed=json.loads((out/'sample_scores.json').read_text(encoding='utf-8'))['samples']
        detailed=[item for item in detailed if item.get('paddle') is not None and item.get('hybrid') is not None]
        fig,axes=plt.subplots(1,2,figsize=(13,4.8))
        x=np.arange(3)
        metrics=['cer_micro','cer_no_space_micro','wer_micro']
        axes[0].bar(x-.18,[100*a[k] for k in metrics],.36,label='Paddle only',color='#5276b8')
        axes[0].bar(x+.18,[100*b[k] for k in metrics],.36,label='Paddle + Qwen',color='#35a187')
        axes[0].set_xticks(x,['CER','CER without spaces','WER']);axes[0].set_title('Micro error rates, all 5 references')
        x=np.arange(len(detailed))
        axes[1].bar(x-.18,[100*item['paddle']['cer'] for item in detailed],.36,label='Paddle only',color='#5276b8')
        axes[1].bar(x+.18,[100*item['hybrid']['cer'] for item in detailed],.36,label='Paddle + Qwen',color='#35a187')
        axes[1].set_xticks(x,[item['id'] for item in detailed],rotation=30);axes[1].set_title('Per-image CER')
        for ax in axes:
            ax.set_ylabel('Error rate (%) — lower is better');ax.legend(fontsize=9);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
            for container in ax.containers: ax.bar_label(container,fmt='%.1f',padding=3,fontsize=8)
            ax.margins(y=.15)
        fig.suptitle('Frozen predictions rescored with user-provided reference transcripts')
        fig.text(.5,.012,'NFC + collapsed whitespace. Unmasked full-page predictions; omitted regions and notation conventions can affect errors.',ha='center',fontsize=9)
        fig.tight_layout(rect=[0,.045,1,.93]);fig.savefig(out/'accuracy_comparison.png',dpi=160);fig.savefig(out/'accuracy_comparison.svg');plt.close(fig)
    (out/'interpretation.json').write_text(json.dumps(facts,indent=2),encoding='utf-8')
    print(json.dumps(facts,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    render(parser.parse_args().results)
