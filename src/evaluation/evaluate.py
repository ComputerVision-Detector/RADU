"""Element matching and relation metrics for RADU and VLM predictions."""
from __future__ import annotations

from collections import Counter
from typing import Any


def _iou(a: list[float], b: list[float]) -> float:
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    area_a=max(0,a[2]-a[0])*max(0,a[3]-a[1]); area_b=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/max(1e-12, area_a+area_b-inter)


def _prf(tp: int, fp: int, fn: int) -> dict[str,float|int]:
    p=tp/max(1,tp+fp); r=tp/max(1,tp+fn)
    return {"tp":tp,"fp":fp,"fn":fn,"precision":p,"recall":r,"f1":2*p*r/max(1e-12,p+r)}


def evaluate(prediction: dict[str,Any], ground_truth: dict[str,Any], iou_threshold: float=.5) -> dict[str,Any]:
    """Greedy type-aware IoU matching, then relation endpoint/direction scoring."""
    preds=prediction.get("elements",[]); refs=ground_truth.get("elements",[])
    candidates=sorted(((_iou(p["bbox"],g["bbox"]),i,j) for i,p in enumerate(preds) for j,g in enumerate(refs)
                       if p.get("type")==g.get("type") and "bbox" in p and "bbox" in g), reverse=True)
    mapping={}; used_p=set(); used_g=set()
    for score,i,j in candidates:
        if score < iou_threshold: break
        if i not in used_p and j not in used_g:
            mapping[preds[i]["id"]]=refs[j]["id"]; used_p.add(i); used_g.add(j)
    element_metrics={}
    for kind in sorted({e.get("type") for e in preds+refs}):
        tp=sum(1 for i in used_p if preds[i].get("type")==kind)
        element_metrics[kind]=_prf(tp,sum(1 for e in preds if e.get("type")==kind)-tp,
                                   sum(1 for e in refs if e.get("type")==kind)-tp)
    pred_rel=[r for r in prediction.get("relations",[]) if r.get("type")=="flows_to"]
    true_rel=[r for r in ground_truth.get("relations",[]) if r.get("type")=="flows_to"]
    def key(rel:dict[str,Any], transform) -> tuple:
        return (rel.get("type"),transform(rel.get("source")),transform(rel.get("target")),rel.get("direction"))
    transform_pred=lambda x:mapping.get(x,"__unmatched__")
    pk=Counter(key(r,transform_pred) for r in pred_rel); tk=Counter(key(r,lambda x:x) for r in true_rel)
    exact_tp=sum((pk & tk).values())
    direction={"all":_prf(exact_tp,sum(pk.values())-exact_tp,sum(tk.values())-exact_tp)}
    pk2=Counter((k[0],k[1],k[2]) for k,n in pk.items() for _ in range(n)); tk2=Counter((k[0],k[1],k[2]) for k,n in tk.items() for _ in range(n))
    topology_tp=sum((pk2 & tk2).values())
    direction["target_and_relation_type"]=_prf(topology_tp,sum(pk2.values())-topology_tp,sum(tk2.values())-topology_tp)
    return {"iou_threshold":iou_threshold,"element_metrics":element_metrics,"relation_metrics":direction,
            "matched_elements":len(mapping),"predicted_elements":len(preds),"ground_truth_elements":len(refs)}
