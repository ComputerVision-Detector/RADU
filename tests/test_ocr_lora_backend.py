import asyncio
import contextlib
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from src.web.backend import app as api


def optional_module(name):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name == name:
            return None
        raise


adapter = optional_module('src.cv.ocr_lora')
worker = optional_module('src.cv.ocr_lora_worker')
comparison = optional_module('src.web.backend.ocr_compare')


def image_bytes():
    stream = io.BytesIO()
    Image.new('RGB', (20, 12), 'white').save(stream, 'PNG')
    return stream.getvalue()


class LoraApiTests(unittest.TestCase):
    def test_lora_engine_is_selectable_without_changing_existing_default(self):
        async def fake_run(image, result, engine, device, session_id, name):
            return {'ocr': {'engine': engine}, 'device': device}
        with tempfile.TemporaryDirectory() as directory, patch.object(api, 'OUT', Path(directory)), patch.object(api, '_run_analysis_worker', fake_run), TestClient(api.app) as client:
            response = client.post('/api/analyze?ocr_engine=paddleocrvl16lora&device=gpu:0', content=image_bytes(), headers={'content-type':'image/png'})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()['ocr']['engine'], 'paddleocrvl16lora')
            default = client.post('/api/analyze', content=image_bytes(), headers={'content-type':'image/png'})
            self.assertEqual(default.status_code, 200)
            self.assertEqual(default.json()['ocr']['engine'], 'none')

    def test_lora_rejects_unverified_cpu_before_creating_session_but_v6_keeps_cpu(self):
        async def fake_run(*args): return {'ok': True}
        with tempfile.TemporaryDirectory() as directory, patch.object(api, 'OUT', Path(directory)), patch.object(api, '_run_analysis_worker', fake_run), TestClient(api.app) as client:
            response = client.post('/api/analyze?ocr_engine=paddleocrvl16lora&device=cpu', content=image_bytes())
            self.assertEqual(response.status_code, 400)
            self.assertIn('GPU', response.json()['detail'])
            self.assertEqual(list(Path(directory).iterdir()), [])
            response = client.post('/api/analyze?ocr_engine=paddleocrv6&device=cpu', content=image_bytes())
            self.assertEqual(response.status_code, 200)

    def test_compare_keeps_saved_document_byte_identical(self):
        session = 'a'*32
        result = {'results':[{'engine':'paddleocrv6','text':'base'},{'engine':'paddleocrvl16lora','text':'lora'}]}
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);saved=folder/(session+'.json');saved.write_bytes(b'{"user_edit":"preserve"}')
            source=folder/(session+'.source.png');source.write_bytes(image_bytes())
            runner = patch.object(comparison, 'run_comparison', AsyncMock(return_value=result)) if comparison else contextlib.nullcontext()
            with patch.object(api,'OUT',folder), runner, TestClient(api.app) as client:
                response=client.post('/api/sessions/'+session+'/compare-ocr?baseline=paddleocrv6&device=gpu:0')
            self.assertEqual(response.status_code,200,response.text)
            self.assertEqual(response.json()['results'][1]['text'],'lora')
            self.assertEqual(saved.read_bytes(),b'{"user_edit":"preserve"}')

    def test_comparison_rejects_paid_engine_and_missing_source(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(api,'OUT',Path(directory)), TestClient(api.app) as client:
            response=client.post('/api/sessions/'+('a'*32)+'/compare-ocr?baseline=clova&device=gpu:0')
            self.assertEqual(response.status_code,400)
            response=client.post('/api/sessions/'+('a'*32)+'/compare-ocr?baseline=paddleocrv6&device=gpu:0')
            self.assertEqual(response.status_code,404)


class LoraAdapterTests(unittest.TestCase):
    def test_corrupt_or_missing_weights_are_rejected_before_loading_models(self):
        self.assertIsNotNone(adapter,'fixed LoRA adapter is missing')
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'weight';p.write_bytes(b'abc')
            self.assertEqual(adapter.check_digest(p,'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'), 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')
            with self.assertRaisesRegex(RuntimeError,'SHA-256'):adapter.check_digest(p,'0'*64)
            with self.assertRaises(RuntimeError):adapter.check_digest(p.with_name('missing'),'0'*64)

    def test_float_polygons_use_frozen_axis_aligned_crop_and_original_pixels(self):
        self.assertIsNotNone(worker,'LoRA worker is missing')
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);source=folder/'source.png'
            image=Image.new('RGB',(6,4));image.putdata([(x*20,y*30,10) for y in range(4) for x in range(6)]);image.save(source)
            regions=worker.make_crops(source,[[[-1.2,1.1],[2.2,1.1],[2.2,4.8],[-1.2,4.8]],[[3.2,.2],[5.8,.2],[5.8,1.8],[3.2,1.8]]],folder/'crops')
            self.assertEqual([r['bbox'] for r in regions],[[3,0,6,2],[0,1,3,4]])
            with Image.open(regions[0]['image']) as crop:
                self.assertEqual(crop.size,(3,2));self.assertEqual(crop.getpixel((0,0)),(60,0,10))

    def test_empty_detection_does_not_load_torch_or_call_recognizer(self):
        self.assertIsNotNone(adapter,'fixed LoRA adapter is missing')
        def fake_stage(stage,image,output,device,workdir,**kwargs):
            if stage!='detect': raise AssertionError('recognizer must not run without text detections')
            Path(output).write_text(json.dumps({'regions':[],'width':20,'height':12,'image_sha256':'input','seconds':.1}))
        with patch.object(adapter,'verify_adapter',return_value={'adapter_sha256':'approved'}), patch.object(adapter,'run_stage',side_effect=fake_stage):
            text,details=adapter.recognize_lora(np.full((12,20,3),255,dtype=np.uint8),'gpu:0')
        self.assertEqual(text,'');self.assertEqual(details['lines'],[])
        self.assertFalse(details['metadata']['recognizer_called'])

    def test_invalid_cpu_does_not_initialize_models(self):
        self.assertIsNotNone(adapter,'fixed LoRA adapter is missing')
        with self.assertRaisesRegex(ValueError,'GPU'):
            adapter.recognize_lora(np.zeros((2,2,3),dtype=np.uint8),'cpu')


class ComparisonTests(unittest.TestCase):
    def test_baseline_gpu_model_is_released_before_lora_starts(self):
        memory={'allocated':False}
        def recognize(frame,engine,device=None):
            if engine=='paddleocrv6':memory['allocated']=True;return 'baseline',{'lines':[]}
            if memory['allocated']:raise RuntimeError('baseline GPU allocation still held')
            return 'lora',{'lines':[]}
        def release():memory['allocated']=False
        with patch.object(comparison,'recognize',side_effect=recognize), patch.object(comparison,'release_paddleocrv6',side_effect=release,create=True):
            result=comparison.compare_image(np.full((12,20,3),255,dtype=np.uint8),'paddleocrv6','gpu:0')
        self.assertEqual(result['successful_engines'],2)
        self.assertEqual(result['results'][1]['text'],'lora')

    def test_failed_baseline_releases_gpu_before_lora_starts(self):
        memory={'allocated':False}
        def recognize(frame,engine,device=None):
            if engine=='paddleocrv6':memory['allocated']=True;raise RuntimeError('baseline failed after allocating')
            if memory['allocated']:raise RuntimeError('baseline GPU allocation still held')
            return 'lora',{'lines':[]}
        def release():memory['allocated']=False
        with patch.object(comparison,'recognize',side_effect=recognize), patch.object(comparison,'release_paddleocrv6',side_effect=release,create=True):
            result=comparison.compare_image(np.full((12,20,3),255,dtype=np.uint8),'paddleocrv6','gpu:0')
        self.assertEqual(result['successful_engines'],1)
        self.assertEqual(result['results'][1]['text'],'lora')

    def test_each_engine_gets_identical_pixels_even_if_first_mutates_its_copy(self):
        self.assertIsNotNone(comparison,'comparison module is missing')
        image=np.full((12,20,3),255,dtype=np.uint8)
        def recognize(frame,engine,device=None):
            value=str(int(frame[0,0,0]));frame[:]=0
            return value,{'lines':[],'metadata':{'device':device}}
        with patch.object(comparison,'recognize',side_effect=recognize):
            result=comparison.compare_image(image,'paddleocrv6','gpu:0')
        self.assertEqual([r['text'] for r in result['results']],['255','255'])
        self.assertEqual(int(image[0,0,0]),255)
        self.assertEqual(result['results'][0]['input_sha256'],result['results'][1]['input_sha256'])

    def test_one_engine_failure_is_explicit_and_does_not_hide_other_result(self):
        self.assertIsNotNone(comparison,'comparison module is missing')
        def recognize(frame,engine,device=None):
            if engine=='paddleocrvl16lora':raise RuntimeError('selected adapter changed')
            return 'baseline',{'lines':[]}
        with patch.object(comparison,'recognize',side_effect=recognize):
            result=comparison.compare_image(np.full((12,20,3),255,dtype=np.uint8),'paddleocrv6','gpu:0')
        self.assertEqual(result['results'][0]['text'],'baseline')
        self.assertIn('selected adapter changed',result['results'][1]['error'])
        self.assertEqual(result['successful_engines'],1)

    def test_timeout_reaps_only_its_own_process_group(self):
        self.assertIsNotNone(comparison,'comparison module is missing')
        with tempfile.TemporaryDirectory() as directory:
            log=Path(directory)/'child.log'
            command=[sys.executable,'-c','import os,time; print(os.getpid(),flush=True); time.sleep(60)']
            with self.assertRaises(TimeoutError):asyncio.run(comparison.run_process(command,log,.4))
            pid=int(log.read_text().strip())
            with self.assertRaises(ProcessLookupError):os.kill(pid,0)


if __name__=='__main__':unittest.main()
