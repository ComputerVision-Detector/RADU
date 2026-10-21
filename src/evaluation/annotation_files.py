"""Create per-image human labels without overwriting edits; merge for scoring."""
import argparse
import hashlib
import json
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def create(inventory, references, directory):
    inventory=load(inventory); references=load(references)
    verified={item['id']:item for item in references['samples']}
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    created=0
    for sample in inventory['samples']:
        identity=sample['id']; path=directory/f'{identity}.json'
        if path.exists(): continue
        label=verified.get(identity, {'id':identity,'text':None,'status':'unlabeled','annotator':'','notes':''})
        with path.open('x',encoding='utf-8') as out:
            json.dump(label,out,ensure_ascii=False,indent=2); out.write('\n')
        created+=1
    return created


def check(directory, manifest, strict=False):
    manifest=Path(manifest); data=load(manifest)
    identities=[s['id'] for s in data['samples']]
    if len(identities)!=len(set(identities)): raise ValueError('Duplicate manifest IDs')
    samples=[]
    for identity in identities:
        path=Path(directory)/f'{identity}.json'; label=load(path)
        if label.get('id')!=identity: raise ValueError(f'ID mismatch: {path}')
        if label.get('status') not in ('unlabeled','draft','verified'): raise ValueError(f'Invalid status: {path}')
        if label.get('text') is not None and not isinstance(label['text'],str): raise ValueError(f'Invalid text: {path}')
        if label['status']=='verified' and not (label.get('text') or '').strip(): raise ValueError(f'Empty verified text: {path}')
        for key in ('annotator','notes'):
            if not isinstance(label.get(key),str): raise ValueError(f'Invalid {key}: {path}')
        samples.append(label)
    if strict:
        extra={p.stem for p in Path(directory).glob('*.json')}-set(identities)
        if extra: raise ValueError(f'Unknown label files: {sorted(extra)}')
    return samples


def merge(directory, manifest, output):
    manifest=Path(manifest)
    samples=check(directory,manifest)
    out=Path(output)
    # The input label files must never be overwritten by an aggregate.
    if out.resolve().parent==Path(directory).resolve(): raise ValueError('Write merged output outside labels directory')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'version':1,'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
                               'source':'per-image human labels','samples':samples},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return len(samples)


def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('create'); p.add_argument('--inventory',required=True); p.add_argument('--references',required=True); p.add_argument('--directory',required=True)
    p=sub.add_parser('merge'); p.add_argument('--directory',required=True); p.add_argument('--manifest',required=True); p.add_argument('--output',required=True)
    p=sub.add_parser('check'); p.add_argument('--directory',required=True); p.add_argument('--manifest',required=True)
    args=parser.parse_args()
    if args.command=='create': print(f'Created {create(args.inventory,args.references,args.directory)} labels (existing files preserved)')
    elif args.command=='check':
        from collections import Counter
        samples=check(args.directory,args.manifest,strict=True)
        print(json.dumps({'files':len(samples),'statuses':dict(Counter(s['status'] for s in samples))},ensure_ascii=False))
    else: print(f'Merged {merge(args.directory,args.manifest,args.output)} labels')

if __name__=='__main__': main()
