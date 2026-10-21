"""Browser regression: real SVG gestures, cell editing, arrow endpoints, save and PDF.
Run from project root with .venv/bin/python tests/browser/hybrid_vector_review.py.
Requires the local server at :8010 and playwright with Chrome.
"""
import json
from pathlib import Path
import shutil
from uuid import uuid4
import requests
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data' / 'hybrid-smoke'
source = ROOT / 'data/hybrid/e745fbc9af124029b3acd68b2f118197'
job = uuid4().hex
folder = ROOT / 'data/hybrid' / job
folder.mkdir(parents=True)
for name in ('source.png','ocr.json','review.json'):
    shutil.copyfile(source/name,folder/name)
(folder/'state.json').write_text(json.dumps({'id':job,'status':'ready'}))
base = f'http://127.0.0.1:8010/api/jobs/{job}/vector'

with sync_playwright() as p:
    browser = p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox'])
    page = browser.new_page(viewport={'width':1600,'height':1000},accept_downloads=True)
    errors = []
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto(f'http://127.0.0.1:8010/review/{job}')
    group = page.locator('#diagram [data-id="ocr-text-0003"]')
    group.wait_for()
    initial = requests.get(base).json()
    before = next(e for e in initial['elements'] if e['id']=='ocr-text-0003')['bbox']
    print('Testing inline text edit', flush=True)
    group.dblclick()
    page.locator('.inline-editor').fill('벡터 직접 수정')
    page.locator('.inline-editor').press('Control+Enter')
    shape = group.locator('.element-shape').bounding_box()
    x,y = shape['x']+shape['width']/2,shape['y']+shape['height']/2
    page.mouse.move(x,y);page.mouse.down();page.mouse.move(x+35,y+18,steps=6);page.mouse.up()
    handle=page.locator('#diagram .resize-handle[data-handle="se"]').bounding_box()
    x,y=handle['x']+handle['width']/2,handle['y']+handle['height']/2
    page.mouse.move(x,y);page.mouse.down();page.mouse.move(x+25,y+12,steps=6);page.mouse.up()
    print('Text drag and resize finished', flush=True)
    # Create a 2x2 table by drawing its bounds in source coordinates.
    page.locator('#hybrid-table').click()
    page.locator('#table-rows').fill('2');page.locator('#table-cols').fill('2')
    page.locator('#draw-table').click()
    bounds=page.locator('#diagram').bounding_box()
    x,y=bounds['x']+bounds['width']*.25,bounds['y']+bounds['height']*.55
    page.mouse.move(x,y);page.mouse.down();page.mouse.move(x+200,y+120,steps=8);page.mouse.up()
    cell=page.locator('#diagram [data-id^="cell-"]').first
    cell.dblclick()
    page.locator('.inline-editor').fill('셀 수정')
    page.locator('.inline-editor').press('Control+Enter')
    print('Table editing finished', flush=True)
    # Draw an arrow and move the end handle.
    page.locator('[data-tool="arrow"]').click()
    x,y=bounds['x']+bounds['width']*.15,bounds['y']+bounds['height']*.45
    page.mouse.move(x,y);page.mouse.down();page.mouse.move(x+150,y+35,steps=8);page.mouse.up()
    arrow=page.locator('#diagram [data-id^="manual-arrow-"]').last
    page.keyboard.press('v')
    arrow.locator('.element-shape').click()
    end=page.locator('#diagram .arrow-handle[data-handle="end"]').bounding_box()
    x,y=end['x']+end['width']/2,end['y']+end['height']/2
    page.mouse.move(x,y);page.mouse.down();page.mouse.move(x+25,y+20,steps=6);page.mouse.up()
    with page.expect_response(lambda r:r.request.method=='PUT' and 'reviewed=true' in r.url) as response:
        page.locator('#save').click()
    assert response.value.status==200,response.value.text()
    saved=requests.get(base).json()
    changed=next(e for e in saved['elements'] if e['id']=='ocr-text-0003')
    assert changed['text']=='벡터 직접 수정'
    assert changed['bbox']!=before
    assert changed['bbox'][2]-changed['bbox'][0]>before[2]-before[0]
    cells=[e for e in saved['elements'] if e.get('semantic_type')=='table-cell']
    assert len(cells)==4 and any(e['text']=='셀 수정' for e in cells)
    assert any(e['type']=='Arrow' and e.get('start') and e.get('end') for e in saved['elements'])
    page.locator('#undo').click();page.locator('#redo').click()
    with page.expect_download() as download:
        page.locator('#hybrid-pdf').click()
    download.value.save_as(OUT/'vector-browser.pdf')
    page.screenshot(path=str(OUT/'vector-desktop.png'),full_page=True)
    page.reload()
    page.locator('#diagram [data-id="ocr-text-0003"]').wait_for()
    assert '벡터 직접 수정' in page.locator('#diagram').text_content()
    page.set_viewport_size({'width':390,'height':844})
    page.screenshot(path=str(OUT/'vector-mobile.png'),full_page=True)
    print('Mobile overflow:',page.evaluate('document.documentElement.scrollWidth>innerWidth'))
    assert not errors,errors
    print('SVG gesture checks passed. Job:',job)
    browser.close()
(OUT/'vector-browser-job.json').write_text(json.dumps({'id':job}))
import subprocess
text = ''.join(subprocess.check_output(['pdftotext', str(OUT/'vector-browser.pdf'), '-']).decode().split())
assert '벡터직접수정' in text and '셀수정' in text
print('PDF text extraction confirms saved element and table cell edits')
