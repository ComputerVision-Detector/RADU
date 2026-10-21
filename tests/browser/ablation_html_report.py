"""Check static ablation comparison controls, source alignment, downloads and offline use."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'data/benchmark/qwen_ablation_pilot'
RESULTS=json.loads((REPORT/'results/results.json').read_text())
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1600,'height':1000},accept_downloads=True)
    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    response=page.goto('http://127.0.0.1:8013/')
    assert response.status==200
    page.locator('.line-row').first.wait_for()
    assert page.locator('.sample').count()==5
    assert page.locator('.line-row').count()==21
    assert page.locator('.line-text mark').count()>0
    page.locator('#changed-only').check()
    assert page.locator('.line-row').count()==17
    page.locator('#search').fill('RANSAC')
    assert page.locator('.line-row').count()==1
    page.locator('.line-row').click()
    page.wait_for_timeout(500)
    assert page.locator('.ocr-box.active').count()==1
    assert page.locator('#zoom-label').inner_text()=='2.4×'
    box=page.locator('.ocr-box.active').bounding_box()
    viewport=page.locator('#image-scroll').bounding_box()
    assert box['x']+box['width']>viewport['x'] and box['x']<viewport['x']+viewport['width']
    assert box['y']+box['height']>viewport['y'] and box['y']<viewport['y']+viewport['height']
    page.locator('#toggle-diff').click()
    assert page.locator('.line-text mark').count()==0
    page.locator('#toggle-diff').click()
    assert page.locator('.line-text mark').count()>0
    page.locator('#show-boxes').check()
    assert page.locator('.ocr-box').count()==21
    page.locator('#fit').click()
    assert page.locator('#zoom-label').inner_text()=='맞춤'
    page.locator('#search').fill('')
    page.locator('#changed-only').uncheck()
    with page.expect_download() as download:
        page.locator('#download-hybrid').click()
    path=REPORT/'html-download-test.txt';download.value.save_as(path)
    assert path.read_text()==RESULTS['samples'][0]['hybrid_text']
    page.locator('#method').click()
    assert page.locator('#modal').is_visible()
    summary=json.loads((REPORT/'results/summary.json').read_text())
    if summary.get('accuracy_available'):
        assert '사용자 제공 정답 전사' in page.locator('#method-content').inner_text()
        assert '63.26%' in page.locator('#summary-cer').inner_text()
        assert '61.49%' in page.locator('#summary-cer').inner_text()
        assert '제외된 영역' in page.locator('#method-content').inner_text()
    else:
        assert '전사 정답이 없는' in page.locator('#method-content').inner_text()
    page.keyboard.press('Escape')
    assert not page.locator('#modal').is_visible()
    page.locator('.sample[data-index="4"]').click()
    assert page.locator('.line-row').count()==6
    page.locator('#changed-only').check()
    assert page.locator('.line-row').count()==0
    assert '조건에 맞는 줄이 없습니다' in page.locator('#lines').inner_text()
    page.locator('#changed-only').uncheck()
    page.locator('#previous').click()
    assert 'hf-0094' in page.locator('#sample-title').inner_text()
    page.locator('.sample[data-index="0"]').click()
    page.locator('.line-row').first.click()
    page.wait_for_timeout(350)
    page.screenshot(path=str(REPORT/'html-desktop-selected.png'),full_page=True)
    page.set_viewport_size({'width':390,'height':844})
    page.locator('#fit').click()
    page.locator('#source-image').evaluate('(img)=>img.decode()')
    page.screenshot(path=str(REPORT/'html-mobile.png'),full_page=True)
    assert not page.evaluate('document.documentElement.scrollWidth>innerWidth')
    # file:// opens without a server or fetch requests; original images stay relative.
    offline=browser.new_page(viewport={'width':1440,'height':1000})
    offline.on('pageerror',lambda e:errors.append(str(e)))
    offline.goto((REPORT/'index.html').as_uri())
    offline.locator('.line-row').first.wait_for()
    offline.locator('#source-image').evaluate('(img)=>img.decode()')
    assert offline.locator('.line-row').count()==21
    assert not errors,errors
    browser.close()
print('HTML comparison: filtering, character diffs, source-box zoom alignment, TXT download, sample navigation, mobile and file:// checks passed')
