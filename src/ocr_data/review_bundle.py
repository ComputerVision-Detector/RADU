"""Local append-only human review bundle. Source labels and model banks stay read-only."""
import argparse, copy, datetime, hashlib, json, math, mimetypes, threading, uuid, time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
from src.evaluation.ocr_page_audit import read, write_new, sha256

BASE=Path('data/experiments')
SCOPE=BASE/'ocr_page_audit_scope_recovery_v1'
BOUND=BASE/'ocr_page_audit_boundary_v1'
AUDIT=BASE/'ocr_page_audit_v1'
FIELDS=['bbox','text','type','include','ignore_reason','order','uncertainty','active']

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def build(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    gt=read(BOUND/'audited-gt-v4.json');queue=read(SCOPE/'review-queue.json');geometry=read(BOUND/'geometry-review.json')['decisions'];fps=[r for r in read(AUDIT/'fp201-audit-v2.json') if r['primary']=='uncertain']
    assert (len(queue),len(geometry),len(fps))==(37,23,13)
    links={};unresolved=[]
    def add(image,id,source,entry):
        links.setdefault((image,id),[]).append(dict(source=source,entry=entry))
    for r in queue:add(r['image_id'],r['annotation_id'],'scope37',r)
    for r in geometry:add(r['page_id'],r['annotation_id'],'geometry23',r)
    for r in fps:
        matches=[q for q in queue if q.get('fp_index')==r['index'] and q['image_id']==r['page_id']]
        if len(matches)==1:add(matches[0]['image_id'],matches[0]['annotation_id'],'FP13',r)
        else:unresolved.append(dict(source='FP13',prediction_id=r['prediction_id'],reason='no unique explicit fp_index link'))
    keys={(p['id'],r['annotation_id']) for p in gt['pages'] for r in p['regions']}
    for k in links:
        if k not in keys:unresolved.append(dict(key=k,reason='source annotation ID absent in GTv4'))
    risks=[dict(image_id=k[0],annotation_id=k[1],sources=v,link_basis='exact image+annotation ID; FP linked by explicit fp_index, never coordinate proximity') for k,v in sorted(links.items())]
    original={(p['id'],g['id']):g for p in read(AUDIT/'source-pages.json') for g in p['regions']}
    pred=read('data/experiments/ocr_page_audit_scope_recovery_v1/recognition/predictions500.json') if (SCOPE/'recognition/predictions500.json').exists() else read('data/experiments/ocr_v2_vl_lora/pages/after.test.predictions.json')
    predmap={p['id']:p for p in pred['pages']}
    pages=[]
    for p in gt['pages']:
        regions=[]
        for r in p['regions']:
            regions.append(dict(image_id=p['id'],annotation_id=r['annotation_id'],bbox=r['bbox'],text='',type='unclassified',include=None,ignore_reason=None,order=r['order'],uncertainty=None,active=True,review_status='unreviewed',reviewer=None,reviewed_at=None,human_verified=False,source_version='GT_v4_AI_draft',source_gt_ids=r.get('source_gt_ids',[]),ai_proposal=r,original_gt=original.get((p['id'],r['annotation_id'])),risk_sources=[x['source'] for x in links.get((p['id'],r['annotation_id']),[])],lineage=[]))
        pages.append(dict(image_id=p['id'],image=p['image'],width=p['width'],height=p['height'],regions=regions,review_status='unreviewed',reviewer=None,reviewed_at=None,human_verified=False,completeness={k:False for k in ['full_canvas_scan','missing_text_checked','inline_math_checked','tables_cancellations_illegible_checked']},predictions=predmap.get(p['id'],{}).get('regions',[])))
    sourcepaths=[BOUND/'audited-gt-v4.json',SCOPE/'review-queue.json',BOUND/'geometry-review.json',AUDIT/'fp201-audit-v2.json',AUDIT/'source-pages.json']+[Path(p['image']) for p in pages]
    lock=dict(hashes={str(p):sha256(p) for p in sourcepaths},policy='No model predictions automatically copied; human draft is not evaluation GT')
    write_new(root/'input-lock.json',lock)
    bundle=dict(schema_version=1,bundle_id=uuid.uuid4().hex,input_lock_sha256=sha256(root/'input-lock.json'),revision=0,test_mode=False,pages=pages,history=[])
    write_new(root/'draft.json',bundle);write_new(root/'risk-items.json',dict(source_counts=dict(scope37=37,geometry23=23,FP13=13),input_references=73,unique_items=len(risks),items=risks,unresolved_links=unresolved))
    write_new(root/'page-completeness.json',[dict(image_id=p['image_id'],review_status='unreviewed',human_verified=False,checks=p['completeness']) for p in pages])
    write_new(root/'schema.json',{'$schema':'https://json-schema.org/draft/2020-12/schema','title':'RADU independent review export','type':'object','required':['schema_version','bundle_id','input_lock_sha256','revision','test_mode','pages','history'],'properties':{'schema_version':{'const':1},'bundle_id':{'type':'string'},'pages':{'type':'array','items':{'type':'object','required':['image_id','regions','completeness','review_status'],'properties':{'regions':{'type':'array','items':{'type':'object','required':['image_id','annotation_id',*FIELDS,'review_status','reviewer','reviewed_at','source_version','human_verified']}}}}}},'additional_validation':'Runtime validates IDs, bounds, provenance, human attestation and page completion; see review_bundle.py'})
    shutil_path=Path(__file__).with_name('review_bundle.html');(root/'index.html').write_bytes(shutil_path.read_bytes())
    write_new(root/'schema-v2.json',read(Path(__file__).with_name('review_bundle.schema.json')))
    print(json.dumps(dict(pages=len(pages),regions=sum(len(p['regions']) for p in pages),unique_risks=len(risks),unresolved=unresolved)))


def validate(state, baseline):
    if state.get('bundle_id')!=baseline['bundle_id'] or state.get('input_lock_sha256')!=baseline['input_lock_sha256'] or state.get('schema_version')!=1:raise ValueError('wrong bundle/source fingerprint')
    if {p['image_id'] for p in state['pages']}!={p['image_id'] for p in baseline['pages']} or len(state['pages'])!=len(baseline['pages']):raise ValueError('page identity drift')
    if type(state.get('test_mode')) is not bool or not isinstance(state.get('history'),list):raise ValueError('invalid session metadata')
    seen=set()
    baselinepages={p['image_id']:p for p in baseline['pages']}
    for p in state['pages']:
        bp=baselinepages[p['image_id']]
        for k in ['image','width','height','predictions']:
            if p[k]!=bp[k]:raise ValueError('source page drift')
        if set(p['completeness'])!=set(bp['completeness']) or not all(type(v) is bool for v in p['completeness'].values()):raise ValueError('invalid full-page checklist')
        orig={r['annotation_id']:r for r in bp['regions']}
        if not set(orig)<={r['annotation_id'] for r in p['regions']}:raise ValueError('source region deleted; use supersede')
        for r in p['regions']:
            k=(p['image_id'],r['annotation_id'])
            if k in seen or r['image_id']!=p['image_id']:raise ValueError('duplicate/cross-page ID')
            if r['review_status'] not in ['unreviewed','draft','hold','verified'] or type(r['active']) is not bool or type(r['human_verified']) is not bool or r['include'] is not None and type(r['include']) is not bool:raise ValueError('invalid region state')
            seen.add(k);b=r['bbox']
            if len(b)!=4 or not all(type(x) in (int,float) and math.isfinite(x) for x in b) or not (0<=b[0]<b[2]<=p['width'] and 0<=b[1]<b[3]<=p['height']):raise ValueError('invalid bbox')
            if not isinstance(r['text'],str) or type(r['order']) not in (int,float) or not math.isfinite(r['order']) or r['include'] not in (None,True,False):raise ValueError('invalid fields')
            if r['annotation_id'] in orig:
                for field in ['ai_proposal','original_gt','source_version','source_gt_ids','risk_sources']:
                    if r[field]!=orig[r['annotation_id']][field]:raise ValueError('immutable provenance drift')
            elif not r['annotation_id'].startswith('human-') or r['source_version']!='human_added_proposal':raise ValueError('invalid new region provenance')
            if r['review_status']=='verified':
                events=[e for e in state['history'] if e['action']=='verify_region' and e['image_id']==p['image_id'] and r['annotation_id'] in e['annotation_ids']]
                if not events or not events[-1].get('reason') or not events[-1].get('human_attested') or not r['reviewer'] or not r['reviewed_at'] or r['include'] is None or r['uncertainty'] or (r['include'] and not r['text'].strip()) or (not r['include'] and not r['ignore_reason']):raise ValueError('verification without review/evidence/decision')
            if r['human_verified'] and (state['test_mode'] or r['review_status']!='verified' or not any(e.get('human_attested') and e.get('reviewer')==r['reviewer'] for e in state['history'] if e['action']=='verify_region' and r['annotation_id'] in e['annotation_ids'] and e['image_id']==p['image_id'])):raise ValueError('human_verified without human input')
            if r['review_status']!='verified' and (r['reviewer'] or r['reviewed_at'] or r['human_verified']):raise ValueError('unreviewed human attribution')
        if p['review_status'] not in ['unreviewed','in_progress','complete']:raise ValueError('invalid page state')
        if p['review_status']=='complete':
            orders=[r['order'] for r in p['regions'] if r['active'] and r['include']]
            if len(orders)!=len(set(orders)):raise ValueError('resolve reading-order ties before completing page')
            if not all(p['completeness'].values()) or any(r['active'] and r['review_status']!='verified' for r in p['regions']) or not p['reviewer'] or not p['reviewed_at']:raise ValueError('page incomplete')
            if not any(e['action']=='complete_page' and e['image_id']==p['image_id'] and e.get('human_attested') for e in state['history']):raise ValueError('no full-page attestation')
            if p['human_verified'] and (state['test_mode'] or not all(r['human_verified'] for r in p['regions'] if r['active'])):raise ValueError('page verification mismatch')
    return state


class Store:
    def __init__(self,root,test_mode=False):
        self.root=Path(root);self.baseline=read(self.root/'draft.json');self.test_mode=test_mode
        for p,h in read(self.root/'input-lock.json')['hashes'].items():
            if sha256(p)!=h:raise ValueError('source drift '+p)
        self.sessions=self.root/('test-sessions' if test_mode else 'sessions');self.sessions.mkdir(exist_ok=True)
        saved=sorted(self.sessions.glob('*.json'));self.state=read(saved[-1]) if saved else copy.deepcopy(self.baseline)
        self.state['test_mode']=test_mode;validate(self.state,self.baseline);self.mutex=threading.Lock()
    def save(self):
        validate(self.state,self.baseline);path=self.sessions/f'{time.time_ns():020d}-r{self.state["revision"]:06d}-{uuid.uuid4().hex}.json';write_new(path,self.state);return path.name
    def action(self,req):
        with self.mutex:
            if req.get('revision')!=self.state['revision']:raise ValueError('stale revision: reload before editing')
            before=copy.deepcopy(self.state);s=copy.deepcopy(self.state);p=next(p for p in s['pages'] if p['image_id']==req['image_id']);ids=req.get('annotation_ids',[]);selected=[r for r in p['regions'] if r['annotation_id'] in ids]
            if len(selected)!=len(set(ids)):raise ValueError('unknown/duplicate selected ID')
            action=req['action'];reason=req.get('reason','').strip();reviewer=req.get('reviewer','').strip();attested=req.get('human_attested') is True
            event=dict(action=action,image_id=p['image_id'],annotation_ids=ids,reason=reason,reviewer=reviewer or None,human_attested=attested,test_mode=self.test_mode,at=now(),before=copy.deepcopy(selected))
            def reset(r):r.update(review_status='hold' if r.get('uncertainty') else 'draft',reviewer=None,reviewed_at=None,human_verified=False)
            if action=='edit':
                if len(selected)!=1 or not reason:raise ValueError('edit needs one region and reason')
                values=req['values']
                if set(values)-set(FIELDS):raise ValueError('forbidden attribution/source edit')
                selected[0].update(values);reset(selected[0])
            elif action in ('add','split','merge'):
                if not reason:raise ValueError('lineage reason required')
                values=req['regions']
                if action=='add' and (selected or len(values)!=1):raise ValueError('add one region')
                if action=='split' and (len(selected)!=1 or len(values)<2):raise ValueError('split needs one source and >=2 children')
                if action=='merge' and (len(selected)<2 or len(values)!=1):raise ValueError('merge needs >=2 sources and one child')
                for r in selected:r['active']=False;reset(r)
                for v in values:
                    if set(v)-set(FIELDS):raise ValueError('forbidden new fields')
                    r=dict(image_id=p['image_id'],annotation_id='human-'+uuid.uuid4().hex,bbox=[0,0,1,1],text='',type='unclassified',include=None,ignore_reason=None,order=0,uncertainty=None,active=True,source_version='human_added_proposal',source_gt_ids=ids,lineage=[dict(action=action,parents=ids,reason=reason)],ai_proposal=None,original_gt=None,risk_sources=[]);r.update(v);reset(r);p['regions'].append(r);event['annotation_ids']=event['annotation_ids']+[r['annotation_id']]
            elif action=='verify_region':
                if len(selected)!=1 or not reviewer or not reason or not attested:raise ValueError('explicit reviewer, evidence, independent-image attestation required')
                selected[0].update(review_status='verified',reviewer=reviewer,reviewed_at=event['at'],human_verified=not self.test_mode)
            elif action=='check_page':
                if not reason:raise ValueError('page scan evidence required')
                if set(req['checks'])!=set(p['completeness']) or not all(type(x) is bool for x in req['checks'].values()):raise ValueError('invalid completeness checks')
                p['completeness']=req['checks']
            elif action=='complete_page':
                if not reviewer or not reason or not attested:raise ValueError('full-page attestation required')
                p.update(review_status='complete',reviewer=reviewer,reviewed_at=event['at'],human_verified=not self.test_mode)
            else:raise ValueError('unknown action')
            if action!='complete_page':p.update(review_status='in_progress',reviewer=None,reviewed_at=None,human_verified=False)
            event['after']=copy.deepcopy([r for r in p['regions'] if r['annotation_id'] in event['annotation_ids']]);event['page_checks_before']=next(q for q in before['pages'] if q['image_id']==p['image_id'])['completeness'];event['page_checks_after']=copy.deepcopy(p['completeness'])
            s['history'].append(event);s['revision']+=1;validate(s,self.baseline);self.state=s;self.save();return s
    def restore(self,state):
        with self.mutex:
            if state['test_mode']!=self.test_mode:raise ValueError('test/real import mixing forbidden')
            validate(state,self.baseline);self.state=copy.deepcopy(state);self.save();return self.state


def serve(root,port,test_mode):
    store=Store(root,test_mode);allowed={p['image_id']:p['image'] for p in store.baseline['pages']}
    class Handler(BaseHTTPRequestHandler):
        def reply(self,data,code=200,kind='application/json'):
            raw=data if isinstance(data,bytes) else json.dumps(data,ensure_ascii=False).encode();self.send_response(code);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(raw)
        def do_GET(self):
            path=urlparse(self.path).path
            if path=='/':return self.reply((Path(root)/'index.html').read_bytes(),kind='text/html; charset=utf-8')
            if path in ['/api/state','/api/export']:return self.reply(store.state)
            if path=='/api/risks':return self.reply(read(Path(root)/'risk-items.json'))
            if path.startswith('/image/') and path[7:] in allowed:return self.reply(Path(allowed[path[7:]]).read_bytes(),kind='image/png')
            self.reply({'error':'not found'},404)
        def do_POST(self):
            try:
                if self.headers.get('Origin') not in (None,f'http://127.0.0.1:{port}',f'http://localhost:{port}'):raise ValueError('local origin required')
                length=int(self.headers.get('Content-Length','0'))
                if length>20000000:raise ValueError('payload too large')
                req=json.loads(self.rfile.read(length) or b'{}');path=urlparse(self.path).path
                if path=='/api/action':return self.reply(store.action(req))
                if path=='/api/import':return self.reply(store.restore(req))
                if path=='/api/save':return self.reply(dict(snapshot=store.save()))
                self.reply({'error':'not found'},404)
            except (ValueError,KeyError,TypeError,StopIteration) as e:self.reply(dict(error=str(e)),400)
    print(f'http://127.0.0.1:{port} test_mode={test_mode}',flush=True);ThreadingHTTPServer(('127.0.0.1',port),Handler).serve_forever()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--stage',choices=['build','serve'],required=True);p.add_argument('--port',type=int,default=8765);p.add_argument('--test-mode',action='store_true');args=p.parse_args()
    if args.stage=='build':build(args.root)
    else:serve(args.root,args.port,args.test_mode)
