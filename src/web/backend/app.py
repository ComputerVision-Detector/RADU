"""RADU prototype API: upload, analyze, edit-save, and export."""
from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.cv.detector import analyze_image
from src.cv.exporter import export_document
from src.cv.ocr_engine import recognize
from src.cv.relation import analyze_relations

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"data"/"processed"/"sessions"; OUT.mkdir(parents=True,exist_ok=True)
app=FastAPI(title="RADU",version="0.1.0",description="Relation-aware handwritten flowchart restoration prototype")
app.mount("/static",StaticFiles(directory=str(Path(__file__).resolve().parents[1]/"frontend")),name="static")


@app.get("/",response_class=HTMLResponse)
def home(): return (Path(__file__).resolve().parents[1]/"frontend"/"index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health(): return {"status":"ok","version":"0.1.0"}


@app.post("/api/analyze")
async def analyze(file: UploadFile=File(...), ocr_engine: str="none"):
    payload=await file.read()
    if len(payload)>20*1024*1024: raise HTTPException(413,"Maximum upload size is 20 MB")
    image=cv2.imdecode(np.frombuffer(payload,np.uint8),cv2.IMREAD_COLOR)
    if image is None: raise HTTPException(400,"Upload must be a readable image")
    document=analyze_image(image)
    if ocr_engine.lower() not in {"none",""}:
        try:
            text,details=recognize(image,ocr_engine)
            document["ocr"]={"engine":ocr_engine,"text":text,"lines":details.get("lines",[])}
            # OCR lines remain separate Text elements and can overlap detected Box elements.
            for i,line in enumerate(details.get("lines",[])):
                raw=line.get("bbox") or []
                if raw and isinstance(raw[0],(list,tuple)):
                    xs=[p[0] for p in raw]; ys=[p[1] for p in raw]; bbox=[min(xs),min(ys),max(xs),max(ys)]
                elif len(raw)==4: bbox=raw
                else: continue
                document["elements"].append({"id":f"ocr-text-{i+1:04d}","type":"Text","bbox":bbox,"text":line["text"],"confidence":line.get("confidence"),"source":ocr_engine})
        except Exception as exc: document.setdefault("warnings",[]).append({"code":"ocr_unavailable","message":str(exc)})
    document=analyze_relations(document)
    document["session_id"]=uuid4().hex
    return document


@app.post("/api/export/{fmt}")
async def export(fmt: str, document: dict):
    if fmt not in {"json","svg","png","pdf"}: raise HTTPException(400,"Supported formats: json, svg, png, pdf")
    path=OUT/f"{uuid4().hex}.{fmt}"
    try: export_document(document,path,fmt)
    except (ValueError,RuntimeError) as exc: raise HTTPException(400,str(exc)) from exc
    media={"json":"application/json","svg":"image/svg+xml","png":"image/png","pdf":"application/pdf"}[fmt]
    return FileResponse(path,media_type=media,filename=f"radu-flowchart.{fmt}")


@app.post("/api/save")
async def save(document: dict):
    session=document.get("session_id") or uuid4().hex
    path=OUT/f"{session}.json"
    path.write_text(json.dumps(document,ensure_ascii=False,indent=2),encoding="utf-8")
    return {"session_id":session,"saved":True}
