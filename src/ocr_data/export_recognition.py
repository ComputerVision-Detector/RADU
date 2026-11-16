"""Explicitly screened PP-OCRv6 recognition lists; never normalize labels silently."""
import argparse
from collections import Counter
import json
from pathlib import Path
from src.ocr_data.build import jsonl,write
from src.ocr_data.audit import sha256


def export(dataset, config, max_length, output):
    import yaml
    if max_length<1:raise ValueError('positive max length required')
    dataset=Path(dataset);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    conf=yaml.safe_load(Path(config).read_text())
    chars=set(conf['PostProcess']['character_dict']);chars.add(' ')
    rejected=[];counts={}
    for split in ['train','validation','test']:
        rows=[json.loads(x) for x in (dataset/f'recognition_candidates.{split}.jsonl').read_text().splitlines()]
        count=0
        with (output/f'{split}.txt').open('x',encoding='utf-8') as out:
            for r in rows:
                text=r['text'];reasons=[]
                if any(c in text for c in '\n\r\t'):reasons.append('unsupported_line_or_tab')
                if len(text)>max_length:reasons.append('maximum_length_exceeded')
                unknown=sorted(set(text)-chars)
                if unknown:reasons.append('unsupported_characters')
                if reasons:rejected.append(dict(page_id=r['page_id'],region_id=r['region_id'],split=split,reasons=reasons,unsupported=unknown,text=text));continue
                out.write(str((dataset/r['image']).resolve())+'\t'+text+'\n');count+=1
        counts[split]=count
    jsonl(output/'rejected.jsonl',rejected)
    write(output/'manifest.json',dict(model='PP-OCRv6_medium_rec',purpose='recognition-only training/evaluation lists; no training started',
         max_text_length=max_length,character_config_sha256=sha256(config),dataset_manifest_sha256=sha256(dataset/'manifest.json'),
         charset_policy='exact codepoints; space enabled; no normalization, truncation or replacement',counts=counts,rejected=len(rejected)))
    return dict(counts=counts,rejected=len(rejected))


def export_vl(dataset, tokenizer, max_length, output):
    """PaddleOCR-VL direct-recognition task manifest; labels are never truncated.

    This is an input/task/target list for a future training adapter, not Qwen
    structure SFT and not a claim of upstream trainer integration.
    """
    from tokenizers import Tokenizer
    if max_length<1:raise ValueError('positive max label-token length required')
    dataset=Path(dataset);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    tok=Tokenizer.from_file(str(tokenizer));vocab=tok.get_vocab()
    unknown_ids={vocab[x] for x in ['[UNK]','<unk>'] if x in vocab}
    rejected=[];counts={};maximum=0
    for split in ['train','validation','test']:
        accepted=[]
        for line in (dataset/f'recognition_candidates.{split}.jsonl').read_text().splitlines():
            r=json.loads(line);text=r['text'];encoded=tok.encode(text,add_special_tokens=False);reasons=[]
            maximum=max(maximum,len(encoded.ids))
            if len(encoded.ids)>max_length:reasons.append('maximum_label_tokens_exceeded')
            if any(x in unknown_ids for x in encoded.ids):reasons.append('unknown_token')
            if any(c in text for c in '\n\r\t'):reasons.append('requires_multiline_or_tab_review')
            if reasons:rejected.append(dict(page_id=r['page_id'],region_id=r['region_id'],split=split,text=text,reasons=reasons,label_tokens=len(encoded.ids)));continue
            row=dict(r);row.update(image=str((dataset/r['image']).resolve()),query='OCR:',label_tokens=len(encoded.ids),model='PaddleOCR-VL-1.6',task='GT-crop recognition only');accepted.append(row)
        jsonl(output/f'{split}.jsonl',accepted);counts[split]=len(accepted)
    jsonl(output/'rejected.jsonl',rejected)
    write(output/'manifest.json',dict(model='PaddleOCR-VL-1.6',purpose='direct recognition task lists for future training adapter; no training started',counts=counts,rejected=len(rejected),
        max_label_tokens=max_length,maximum_observed_label_tokens=maximum,tokenizer_sha256=sha256(tokenizer),dataset_manifest_sha256=sha256(dataset/'manifest.json'),
        length_scope='label tokens only; future training preflight must budget image/prompt tokens separately',semantic_types='suggestions pending human confirmation'))
    return dict(counts=counts,rejected=len(rejected),maximum_observed_label_tokens=maximum)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',required=True);p.add_argument('--model',choices=['ppocrv6','paddleocrvl16'],default='ppocrv6');p.add_argument('--config');p.add_argument('--tokenizer');p.add_argument('--max-length',type=int,required=True);p.add_argument('--output',required=True)
    a=p.parse_args()
    if a.model=='ppocrv6':
        if not a.config:p.error('--config is required for ppocrv6')
        result=export(a.dataset,a.config,a.max_length,a.output)
    else:
        if not a.tokenizer:p.error('--tokenizer is required for paddleocrvl16')
        result=export_vl(a.dataset,a.tokenizer,a.max_length,a.output)
    print(json.dumps(result))
if __name__=='__main__':main()
