"""Two-browser collaboration, persistence, export, mobile and keyboard QA."""
import io,json,os,tempfile,zipfile
from pathlib import Path
from playwright.sync_api import sync_playwright,expect
ROOT=Path(__file__).resolve().parents[2]
URL=os.environ.get('LABEL_TEST_URL','http://127.0.0.1:8014')
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox'])
    a=browser.new_context(viewport={'width':1512,'height':1040},accept_downloads=True)
    b=browser.new_context(viewport={'width':1512,'height':1040})
    page=a.new_page();other=b.new_page();errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)));other.on('pageerror',lambda e:errors.append(str(e)))
    for tab,name in [(page,'브라우저 검증 A'),(other,'브라우저 검증 B')]:
        tab.goto(URL);tab.locator('#author-name').fill(name);tab.locator('#name-form button[type=submit]').click()
        expect(tab.locator('#next')).to_be_enabled()
    page.locator('#next').click();expect(page.locator('#image-title')).to_have_text('hf-0000')
    expect(page.locator('#transcription')).to_be_enabled();page.locator('#transcription').fill('한국어와 English\n첫 번째 셀\t두 번째 셀')
    page.locator('#notes').fill('실제 정답이 아닌 브라우저 기능 검증 기록')
    other.locator('#refresh').click();other.locator('[data-id="hf-0000"]').click()
    expect(other.locator('#transcription')).to_be_disabled();expect(other.locator('#claim')).to_be_disabled()
    page.locator('#save').click();expect(page.locator('#save-state')).to_have_text('서버 저장됨')
    page.reload();expect(page.locator('#identity')).to_have_text('브라우저 검증 A 님')
    page.locator('[data-id="hf-0000"]').click();expect(page.locator('#transcription')).to_have_value('한국어와 English\n첫 번째 셀\t두 번째 셀')
    page.locator('#zoom-in').click();page.locator('#rotate').click();page.locator('#fit').click()
    expect(page.locator('#verify')).to_be_enabled();page.locator('#verify').click()
    expect(page.locator('#image-status')).to_have_text('검수 완료')
    expect(page.locator('#remaining-note')).to_have_text(f"{len(json.loads((ROOT/'labeling/data/seed.json').read_text())['items'])-6}장 남음")
    page.locator('#export').click()
    with page.expect_download() as info:page.locator('#download-zip').click()
    payload=Path(info.value.path()).read_bytes()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert len(archive.namelist())==6
        label=json.loads(archive.read('labels/hf-0000.json'));assert label['status']=='verified' and label['annotator']=='브라우저 검증 A'
        assert label['text']=='한국어와 English\n첫 번째 셀\t두 번째 셀'
    page.locator('[data-filter="all"]').click();page.locator('[data-id="hf-0003"]').click();expect(page.locator('#claim')).to_be_disabled()
    page.locator('[data-filter="pending"]').click();page.locator('#next').click();expect(page.locator('#image-title')).to_have_text('hf-0001')
    page.locator('#transcription').fill('브라우저 미저장 초안')
    page.on('dialog',lambda d:d.accept());page.reload();page.locator('[data-id="hf-0001"]').click()
    expect(page.locator('#transcription')).to_have_value('브라우저 미저장 초안')
    page.locator('#save').click();expect(page.locator('#save-state')).to_have_text('서버 저장됨')
    page.locator('#source-image').wait_for(state='visible');page.wait_for_function('document.getElementById("source-image").complete && document.getElementById("source-image").naturalWidth>0')
    page.screenshot(path=str(ROOT/'labeling/qa/desktop.png'),full_page=True)
    page.set_viewport_size({'width':390,'height':844});page.wait_for_timeout(250)
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'Mobile horizontal overflow'
    for selector in ['#viewer','#transcription','#save','#verify']:
        box=page.locator(selector).bounding_box();assert box and box['width']<=390 and box['x']+box['width']<=391,(selector,box)
    image=page.locator('#source-image').bounding_box();view=page.locator('#viewer').bounding_box()
    assert image['x']<view['x']+view['width'] and image['x']+image['width']>view['x'],'Mobile image outside viewport'
    page.screenshot(path=str(ROOT/'labeling/qa/mobile.png'),full_page=True)
    assert not errors,errors
    browser.close()
print('Two-user locking, draft/reload, verify/export ZIP, protected references, local draft recovery, mobile and JS error checks passed')
