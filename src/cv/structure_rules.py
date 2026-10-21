"""Conservative, CPU-only ruled-table and straight-arrow rules (original pixels).

Scores are heuristic evidence, never calibrated probabilities. Unsupported or
ambiguous structures remain review candidates rather than invented connections.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import cv2
import numpy as np

RULE_VERSION = '1.0'

@dataclass(frozen=True)
class RuleConfig:
    max_side: int = 1600
    line_fraction: float = .025
    gap_fraction: float = .004
    boundary_support: float = .65
    min_arrow_fraction: float = .035
    connection_fraction: float = .065
    ambiguity_fraction: float = .012


def _bbox(raw):
    if raw is None or len(raw) == 0:
        return None
    a = np.asarray(raw, dtype=float)
    if a.ndim == 2:
        a = np.array([a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max()])
    if a.shape != (4,) or not np.isfinite(a).all() or a[2] <= a[0] or a[3] <= a[1]:
        return None
    return a.tolist()


def _runs(values):
    indices = np.flatnonzero(values)
    return [int(round(float(g.mean()))) for g in np.split(indices, np.where(np.diff(indices) > 1)[0]+1) if len(g)]


def _support(mask, x1, y1, x2, y2, vertical=False):
    if vertical:
        roi = mask[max(0,y1+3):y2-2, max(0,x1-2):x1+3]
        return float(np.mean(np.any(roi > 0, axis=1))) if roi.size else 0.
    roi = mask[max(0,y1-2):y1+3, max(0,x1+3):x2-2]
    return float(np.mean(np.any(roi > 0, axis=0))) if roi.size else 0.


def _tables(binary, cfg):
    h, w = binary.shape
    gap = max(3, round(min(h,w)*cfg.gap_fraction))
    hk = max(15, round(w*cfg.line_fraction)); vk = max(15, round(h*cfg.line_fraction))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((1,gap),np.uint8))
    horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_OPEN, np.ones((1,hk),np.uint8))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((gap,1),np.uint8))
    vertical = cv2.morphologyEx(vertical, cv2.MORPH_OPEN, np.ones((vk,1),np.uint8))
    grid = cv2.bitwise_or(horizontal, vertical)
    count, _, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(grid,np.ones((3,3),np.uint8)))
    tables=[]
    for x,y,cw,ch,area in stats[1:]:
        if cw < 2*hk or ch < 2*vk:
            continue
        xs = [x+i for i in _runs(np.count_nonzero(vertical[y:y+ch,x:x+cw],axis=0) >= ch*.38)]
        ys = [y+i for i in _runs(np.count_nonzero(horizontal[y:y+ch,x:x+cw],axis=1) >= cw*.38)]
        if len(xs)<2 or len(ys)<2 or (len(xs)-1)*(len(ys)-1)<2:
            continue  # A lone box is not a table.
        # Require a closed outer boundary; ruled paper has no vertical enclosure.
        outer=[_support(horizontal,xs[0],ys[0],xs[-1],ys[0]),
               _support(horizontal,xs[0],ys[-1],xs[-1],ys[-1]),
               _support(vertical,xs[0],ys[0],xs[0],ys[-1],True),
               _support(vertical,xs[-1],ys[0],xs[-1],ys[-1],True)]
        if min(outer)<cfg.boundary_support:
            continue
        nr,nc=len(ys)-1,len(xs)-1
        parent=list(range(nr*nc)); uncertain=False
        def find(i):
            while parent[i]!=i:
                parent[i]=parent[parent[i]]; i=parent[i]
            return i
        def join(i,j): parent[find(i)]=find(j)
        for r in range(nr):
            for c in range(nc):
                for rr,cc,support in ([(r,c+1,_support(vertical,xs[c+1],ys[r],xs[c+1],ys[r+1],True))] if c+1<nc else []) + ([(r+1,c,_support(horizontal,xs[c],ys[r+1],xs[c+1],ys[r+1]))] if r+1<nr else []):
                    if support<.15: join(r*nc+c,rr*nc+cc)
                    elif support<cfg.boundary_support: uncertain=True
        groups={}
        for r in range(nr):
            for c in range(nc): groups.setdefault(find(r*nc+c),[]).append((r,c))
        cells=[]
        for points in groups.values():
            r0=min(r for r,c in points); r1=max(r for r,c in points)+1
            c0=min(c for r,c in points); c1=max(c for r,c in points)+1
            rectangular=len(points)==(r1-r0)*(c1-c0)
            # Nonrectangular missing boundaries: keep atomic cells, never invent L-shaped merges.
            chunks=[(r0,r1,c0,c1)] if rectangular else [(r,r+1,c,c+1) for r,c in points]
            uncertain |= not rectangular
            for ra,rb,ca,cb in chunks:
                cells.append({'bbox':[xs[ca],ys[ra],xs[cb],ys[rb]],'row':ra,'col':ca,
                              'rowspan':rb-ra,'colspan':cb-ca,'text_ids':[],'text':''})
        tables.append({'type':'Table','bbox':[xs[0],ys[0],xs[-1],ys[-1]],'rows':nr,'cols':nc,
                       'cells':sorted(cells,key=lambda a:(a['row'],a['col'])),
                       'source':'structure-rules','needs_review':True,
                       'evidence':{'outer_support':round(min(outer),3),'ambiguous_boundary':bool(uncertain)}})
    table_grid=np.zeros_like(grid)
    for table in tables:
        x,y,r,b=table['bbox']
        table_grid[max(0,y-3):b+4,max(0,x-3):r+4]=grid[max(0,y-3):b+4,max(0,x-3):r+4]
    return tables,table_grid


def _arrows(binary, grid, cfg):
    h,w=binary.shape; size=min(h,w)
    # Table borders cannot become shafts or arrowheads.
    ink=cv2.subtract(binary,cv2.dilate(grid,np.ones((3,3),np.uint8)))
    lines=cv2.HoughLinesP(ink,1,np.pi/180,threshold=max(10,round(size*.018)),
                         minLineLength=max(6,round(size*.008)),maxLineGap=max(3,round(size*.006)))
    if lines is None: return []
    segments=[]
    for raw in lines[:,0]:
        a=raw[:2].astype(float); b=raw[2:].astype(float); length=float(np.linalg.norm(b-a))
        segments.append((a,b,length))
    arrows=[]; tolerance=max(3,size*.004)
    for a,b,length in sorted(segments,key=lambda s:-s[2]):
        if length<size*cfg.min_arrow_fraction: continue
        def head(tip,tail):
            back=(tail-tip)/np.linalg.norm(tail-tip); arms=[]
            for p,q,n in segments:
                if not 4<=n<=min(length*.5,size*.055): continue
                near,far=(p,q) if np.linalg.norm(p-tip)<np.linalg.norm(q-tip) else (q,p)
                if np.linalg.norm(near-tip)>tolerance: continue
                v=far-tip; norm=np.linalg.norm(v)
                if norm<4: continue
                cosine=float(np.dot(v/norm,back)); cross=float(back[0]*v[1]-back[1]*v[0])
                if .4<cosine<.94 and abs(cross)>2:
                    arms.append((1 if cross>0 else -1,n,cosine))
            return any(sa != sb and .5 < na/nb < 2 and abs(ca-cb)<.25
                       for sa,na,ca in arms for sb,nb,cb in arms)
        # Page edges and spiral bindings are frequent false shafts.
        if any(p[0]<w*.025 or p[0]>w*.975 or p[1]<h*.015 or p[1]>h*.985 for p in (a,b)):
            continue
        ha,hb=head(a,b),head(b,a)
        if not ha and not hb: continue
        start,end=(a,b) if hb else (b,a)
        if any(np.linalg.norm(end-np.array(old['end']))<tolerance*2 and np.dot((end-start)/length,(np.array(old['end'])-old['start'])/max(1,np.linalg.norm(np.array(old['end'])-old['start'])))>.95 for old in arrows): continue
        if any(min(np.linalg.norm(start-np.array(old['start']))+np.linalg.norm(end-np.array(old['end'])),
                   np.linalg.norm(start-np.array(old['end']))+np.linalg.norm(end-np.array(old['start'])))<tolerance*4 for old in arrows): continue
        arrows.append({'type':'Arrow','bbox':[float(min(a[0],b[0])-1),float(min(a[1],b[1])-1),float(max(a[0],b[0])+1),float(max(a[1],b[1])+1)],
                       'start':start.tolist(),'end':end.tolist(),'path':[start.tolist(),end.tolist()],
                       'direction':'bidirectional' if ha and hb else 'forward','source_id':None,'target_id':None,
                       'source':'structure-rules','needs_review':True,'evidence':{'arrowhead_detected':True,'both_heads':bool(ha and hb)}})
    return arrows


def connect_arrows(arrows, targets, image_size, cfg=None):
    """Connect by boundary distance + outward direction; ties stay null."""
    cfg=cfg or RuleConfig(); size=min(image_size)
    targets=[(t['id'],_bbox(t.get('bbox'))) for t in targets if t.get('id')]
    targets=[(i,b) for i,b in targets if b]
    for arrow in arrows:
        start=np.array(arrow['start']); end=np.array(arrow['end']); direction=end-start
        direction=direction/max(1e-9,np.linalg.norm(direction))
        for key,point,outward in [('source_id',start,-direction),('target_id',end,direction)]:
            candidates=[]
            for identity,box in targets:
                nearest=np.clip(point,box[:2],box[2:]); distance=float(np.linalg.norm(nearest-point))
                center=(np.array(box[:2])+box[2:])/2
                delta=center-point
                if distance>size*cfg.connection_fraction or np.dot(delta,outward)<=0: continue
                cosine=float(np.dot(delta,outward)/max(1e-9,np.linalg.norm(delta)))
                if cosine<.5: continue
                candidates.append((distance+size*.02*(1-cosine),identity))
            candidates.sort()
            arrow[key]=candidates[0][1] if candidates and (len(candidates)==1 or candidates[1][0]-candidates[0][0]>size*cfg.ambiguity_fraction) else None
        if arrow['source_id']==arrow['target_id']:
            arrow['source_id']=arrow['target_id']=None
        arrow['connection_status']='candidate' if arrow['source_id'] and arrow['target_id'] else 'unresolved'
    return arrows


def detect_structure(image, ocr_lines=(), targets=(), config=None):
    """Return tables/arrows, original-pixel geometry and explicit unresolved cases."""
    cfg=config or RuleConfig()
    if image is None or image.size==0: raise ValueError('Empty image')
    height,width=image.shape[:2]; scale=min(1.,cfg.max_side/max(height,width))
    small=cv2.resize(image,(round(width*scale),round(height*scale))) if scale<1 else image
    gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY) if small.ndim==3 else small
    binary=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,12)
    tables,grid=_tables(binary,cfg)
    arrow_ink=binary.copy()
    # Text strokes must not supply shafts or arrowheads. Use original OCR boxes
    # only for a separate structure mask; never modify OCR input pixels.
    for line in ocr_lines:
        box=_bbox(line.get('bbox'))
        if box:
            x,y,r,b=[int(round(v*scale)) for v in box]
            arrow_ink[max(0,y-2):min(arrow_ink.shape[0],b+3),max(0,x-2):min(arrow_ink.shape[1],r+3)]=0
    arrows=_arrows(arrow_ink,grid,cfg)
    for index,table in enumerate(sorted(tables,key=lambda t:(t['bbox'][1],t['bbox'][0]))):
        table['id']=f'rule-table-{index+1:04d}'
        for j,cell in enumerate(table['cells']): cell['id']=f"{table['id']}-cell-{j+1:04d}"
    for index,arrow in enumerate(arrows): arrow['id']=f'rule-arrow-{index+1:04d}'
    for item in [*tables,*arrows]:
        item['rule_version']=RULE_VERSION
        item['rule_config']=asdict(cfg)
        item['bbox']=[round(max(0,min(width if i%2==0 else height,v/scale)),2) for i,v in enumerate(item['bbox'])]
        for cell in item.get('cells',[]): cell['bbox']=[round(v/scale,2) for v in cell['bbox']]
        for key in ('start','end'):
            if key in item: item[key]=[round(v/scale,2) for v in item[key]]
        if 'path' in item: item['path']=[item['start'],item['end']]
    cells=[c for t in tables for c in t['cells']]
    unassigned=[]
    for line in ocr_lines:
        box=_bbox(line.get('bbox'))
        if not box: continue
        x,y,r,b=box; area=(r-x)*(b-y); matches=[]
        for cell in cells:
            cx,cy,cr,cb=cell['bbox']; overlap=max(0,min(r,cr)-max(x,cx))*max(0,min(b,cb)-max(y,cy))/area
            if overlap>=.8: matches.append(cell)
        if len(matches)==1:
            cell=matches[0]; cell['text_ids'].append(line['id']); cell['text']+='\n' if cell['text'] else ''; cell['text']+=str(line.get('text',''))
        elif any(max(x,t['bbox'][0])<min(r,t['bbox'][2]) and max(y,t['bbox'][1])<min(b,t['bbox'][3]) for t in tables): unassigned.append(line.get('id'))
    connect_arrows(arrows,[*targets,*tables,*[dict(line,type='Text') for line in ocr_lines]],(width,height),cfg)
    return {'rule_version':RULE_VERSION,'config':asdict(cfg),'image':{'width':width,'height':height},
            'tables':tables,'arrows':arrows,'unassigned_text_ids':unassigned,
            'limitations':['Ruled axis-aligned tables and straight arrows only; all detections require visual review.',
                           'Curved/bent arrows, borderless tables and automatic perspective correction are not implemented.']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image',type=Path); parser.add_argument('--ocr',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args(); lines=[]
    if args.ocr: lines=json.loads(args.ocr.read_text())['lines']
    result=detect_structure(cv2.imread(str(args.image)),lines)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__': main()
