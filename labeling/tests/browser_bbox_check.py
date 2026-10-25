"""V2 browser QA; owns a temporary server and never contacts production storage.

Run with a Python environment containing Playwright; node 22+ must be on PATH.
"""
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import time
from urllib.request import urlopen
import zipfile

from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]


def drag(page, start, end):
    page.mouse.move(*start)
    page.mouse.down()
    page.mouse.move(*end, steps=8)
    page.mouse.up()


with tempfile.TemporaryDirectory(prefix='radu-bbox-browser-') as temporary:
    scratch = Path(temporary)
    # A real V1-shaped fixture is present so a V2 write cannot hide a V1 reset.
    v1 = scratch / 'v1.json'
    v1.write_bytes((ROOT / 'data/seed.json').read_bytes())
    before = v1.read_bytes()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    env = {**os.environ, 'PORT': str(port), 'HOST': '127.0.0.1',
           'LABEL_LOCAL_STORE': str(v1), 'LABEL_LOCAL_STORE_V2': str(scratch / 'v2.json'),
           'LABEL_PROPOSALS_FILE': str(scratch / 'no-proposals.json'),
           'LABEL_BACKUP_DIR': str(scratch / 'backup-v1'),
           'LABEL_BACKUP_DIR_V2': str(scratch / 'backup-v2')}
    server = subprocess.Popen([shutil.which('node'), 'server.mjs'], cwd=ROOT, env=env,
                              stdout=subprocess.DEVNULL)
    try:
        for attempt in range(100):
            try:
                with urlopen(url + '/healthz', timeout=1):
                    break
            except OSError:
                time.sleep(.05)
        else:
            raise RuntimeError('Local QA server did not start')
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path='/usr/bin/google-chrome',
                                        headless=True, args=['--no-sandbox'])
            a = browser.new_context(viewport={'width': 1512, 'height': 1040}, accept_downloads=True)
            b = browser.new_context(viewport={'width': 1512, 'height': 1040})
            page, other = a.new_page(), b.new_page()
            errors = []
            for tab, name in [(page, 'V2 QA A'), (other, 'V2 QA B')]:
                tab.on('pageerror', lambda error: errors.append(str(error)))
                tab.on('dialog', lambda dialog: dialog.accept())
                tab.goto(url)
                tab.locator('#author-name').fill(name)
                tab.locator('#name-form button[type=submit]').click()
                expect(tab.locator('[data-version="v2"]')).to_have_class('active')
            page.locator('#next').click()
            expect(page.locator('#image-title')).to_have_text('hf-0000')
            expect(page.locator('#draw-box')).to_be_enabled()
            page.wait_for_function('document.querySelector("#source-image").naturalWidth > 0')
            other.locator('#refresh').click()
            other.locator('[data-id="hf-0000"]').click()
            expect(other.locator('#draw-box')).to_be_disabled()
            expect(other.locator('#claim')).to_be_disabled()

            page.locator('#draw-box').click()
            overlay = page.locator('#bbox-overlay').bounding_box()
            x, y, w, h = (overlay[k] for k in ['x', 'y', 'width', 'height'])
            drag(page, (x + w*.2, y + h*.2), (x + w*.7, y + h*.28))
            expect(page.locator('.bbox-box')).to_have_count(1)
            page.locator('#bbox-text').fill('한국어 English 123 + 기호')
            # Move and resize through actual pointer events, then compare persisted coords.
            rect = page.locator('.bbox-box')
            original = [float(rect.get_attribute(k)) for k in ['x', 'y', 'width', 'height']]
            bounds = rect.bounding_box()
            drag(page, (bounds['x']+bounds['width']/2, bounds['y']+bounds['height']/2),
                 (bounds['x']+bounds['width']/2+12, bounds['y']+bounds['height']/2+10))
            assert float(rect.get_attribute('x')) > original[0]
            width = float(rect.get_attribute('width'))
            handle = page.locator('.handle-se').bounding_box()
            drag(page, (handle['x']+handle['width']/2, handle['y']+handle['height']/2),
                 (handle['x']+handle['width']/2+15, handle['y']+handle['height']/2+10))
            assert float(rect.get_attribute('width')) > width
            page.locator('#bbox-uncertain').check()
            page.locator('#bbox-notes').fill('로컬 기능 검증용, 실제 정답 아님')
            page.locator('#save').click()
            expect(page.locator('#save-state')).to_have_text('서버 저장됨')
            page.reload()
            page.locator('[data-id="hf-0000"]').click()
            page.locator('.box-row').click()
            expect(page.locator('#bbox-text')).to_have_value('한국어 English 123 + 기호')
            expect(page.locator('#bbox-uncertain')).to_be_checked()
            page.locator('#bbox-uncertain').uncheck()
            page.locator('#bbox-reviewed').check()
            page.locator('#verify').click()
            expect(page.locator('#image-status')).to_have_text('검수 완료')

            page.locator('#export').click()
            with page.expect_download() as download:
                page.locator('#download-zip').click()
            with zipfile.ZipFile(io.BytesIO(Path(download.value.path()).read_bytes())) as archive:
                label = json.loads(archive.read('bbox-labels/hf-0000.json'))
                assert label['boxes'][0]['text'] == '한국어 English 123 + 기호'
                assert len(label['source_sha256']) == 64
            page.locator('[data-version="v1"]').click()
            expect(page.locator('#progress-value')).to_have_text('3%')
            page.locator('[data-version="v2"]').click()
            page.locator('#next').click()
            expect(page.locator('#image-title')).to_have_text('hf-0001')
            page.locator('#no-text').check()
            page.locator('#verify').click()
            expect(page.locator('#image-status')).to_have_text('검수 완료')
            page.locator('#next').click()
            expect(page.locator('#image-title')).to_have_text('hf-0002')
            image = page.request.get(url + '/api/labeling?action=item&version=v2&id=hf-0002').json()['image']
            proposal = {'schema_version': 1, 'engine': 'QA fixture', 'proposal_version': 'qa-only',
                        'images': {'hf-0002': {
                            'file': image['file'], 'input_sha256': image['asset_sha256'],
                            'asset_sha256': image['asset_sha256'], 'source_sha256': image['source_sha256'],
                            'source_size': {'width': image['source_width'], 'height': image['source_height']},
                            'lines': [{'id': 'candidate-1', 'bbox': [10, 10, 200, 70],
                                       'prediction': 'OCR fixture', 'order': 1}]}}}
            page.route('**/data/text-box-proposals.json', lambda route: route.fulfill(json=proposal))
            page.reload()
            page.locator('[data-id="hf-0002"]').click()
            expect(page.locator('#import-proposals')).to_be_enabled()
            page.locator('#import-proposals').click()
            expect(page.locator('#bbox-text')).to_have_value('OCR fixture')
            expect(page.locator('#bbox-reviewed')).not_to_be_checked()
            expect(page.locator('#prediction-text')).to_have_text('OCR fixture')
            expect(page.locator('#ocr-reference-text')).to_contain_text('OCR fixture')
            page.locator('#verify').click()
            expect(page.locator('#toast')).to_contain_text('원본과 대조')
            expect(page.locator('#image-status')).not_to_have_text('검수 완료')
            page.locator('#bbox-text').fill('사람이 수정한 문구')
            expect(page.locator('#prediction-text')).to_have_text('OCR fixture')
            expect(page.locator('#bbox-comparison')).to_have_text('OCR 원문을 수정한 상태')
            assert page.locator('#line-crop').evaluate('(canvas) => canvas.width > 0 && canvas.height > 0')
            page.locator('#copy-prediction').click()
            expect(page.locator('#bbox-text')).to_have_value('OCR fixture')
            page.locator('#bbox-text').fill('저장된 사람의 수정')
            page.locator('#save').click()
            expect(page.locator('#save-state')).to_have_text('서버 저장됨')
            page.reload()
            page.locator('[data-id="hf-0002"]').click()
            page.locator('.box-row').click()
            expect(page.locator('#bbox-text')).to_have_value('저장된 사람의 수정')
            expect(page.locator('#prediction-text')).to_have_text('OCR fixture')
            page.locator('#delete-box').click()
            expect(page.locator('.bbox-box')).to_have_count(0)
            page.locator('#save').click()
            expect(page.locator('#save-state')).to_have_text('서버 저장됨')
            proposal['images']['hf-0002']['source_sha256'] = '0'*64
            page.reload()
            page.locator('[data-id="hf-0002"]').click()
            expect(page.locator('#proposal-status')).to_contain_text('fingerprint 불일치')
            expect(page.locator('#import-proposals')).to_be_disabled()
            page.set_viewport_size({'width': 390, 'height': 844})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            assert not errors, errors
            browser.close()
        assert v1.read_bytes() == before, 'V2 browser workflow changed V1 bytes'
        print('PASS: V2 draw/move/resize, locking, uncertain draft/reload, verify/ZIP, V1 isolation, no-text, proposal import/delete/hash rejection, mobile')
    finally:
        server.terminate()
        server.wait(timeout=10)
