"""Read-only local OCR engine comparison on identical preprocessed pixels."""
from __future__ import annotations
import asyncio
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

from src.cv.ocr_engine import recognize, release_paddleocrv6
from src.cv.preprocess import preprocess_image

ROOT=Path(__file__).resolve().parents[3]
LOCAL_BASELINES={'paddleocrv6','paddleocrvl16'}


def compare_image(image,baseline,device):
    if baseline not in LOCAL_BASELINES:raise ValueError('비교 기준은 로컬 OCR 엔진만 지원합니다.')
    processed=preprocess_image(image)
    fingerprint=hashlib.sha256(processed.tobytes()).hexdigest()
    results=[]
    for engine in [baseline,'paddleocrvl16lora']:
        tick=time.perf_counter();record={'engine':engine,'input_sha256':fingerprint}
        try:
            text,details=recognize(processed.copy(),engine,device=device)
            record.update(text=text,lines=details.get('lines',[]),metadata=details.get('metadata',{}))
        except Exception as error:record['error']=f'{type(error).__name__}: {error}'
        finally:
            if engine=='paddleocrv6':
                try:release_paddleocrv6()
                except Exception as error:record['error']=f'기준 엔진 GPU 자원 해제 실패: {error}'
        record['seconds']=time.perf_counter()-tick;results.append(record)
    return {'results':results,'input_sha256':fingerprint,'width':processed.shape[1],
            'height':processed.shape[0],'successful_engines':sum('error' not in r for r in results),
            'comparison_kind':'complete engine paths; same preprocessing, different detection/recognition paths',
            'accuracy_metrics':None}


async def run_process(command,log_path,timeout):
    with Path(log_path).open('wb') as log:
        process=await asyncio.create_subprocess_exec(*command,cwd=ROOT,stdout=log,
                    stderr=asyncio.subprocess.STDOUT,start_new_session=True)
        try:
            return await asyncio.wait_for(process.wait(),timeout)
        except (asyncio.TimeoutError,asyncio.CancelledError) as error:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            await process.wait()
            if isinstance(error,asyncio.CancelledError):raise
            raise TimeoutError('OCR 비교 시간이 초과되어 비교 작업을 종료했습니다.') from error


async def run_comparison(source,workdir,baseline,device):
    output=workdir/'comparison.json';log=workdir/'comparison.log'
    command=[sys.executable,'-m','src.web.backend.ocr_compare',str(source),str(output),baseline,device]
    code=await run_process(command,log,900)
    if code!=0 or not output.is_file():
        raise RuntimeError('OCR 비교 작업에 실패했습니다: '+log.read_text(errors='replace')[-1800:])
    return json.loads(output.read_text())


def main():
    import cv2
    import numpy as np
    from PIL import Image,ImageOps
    source,output,baseline,device=sys.argv[1:]
    with Image.open(source) as image:
        bgr=cv2.cvtColor(np.asarray(ImageOps.exif_transpose(image).convert('RGB')),cv2.COLOR_RGB2BGR)
    result=compare_image(bgr,baseline,device)
    Path(output).write_text(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
