"""Exercise scroll reachability and inline zoom on desktop and narrow screens."""
import os
from playwright.sync_api import sync_playwright, expect
URL=os.environ.get('LABEL_TEST_URL','http://127.0.0.1:8028')
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1366,'height':768})
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto(URL)
    page.locator('#author-name').fill('스크롤 회귀 검증')
    page.locator('#name-form button[type=submit]').click()
    page.locator('#next').click()
    expect(page.locator('#transcription')).to_be_enabled()
    page.wait_for_function('document.getElementById("source-image").naturalWidth>0')
    for width,height in [(1512,1040),(1366,768),(1024,600),(820,650),(800,600),(390,844)]:
        page.set_viewport_size({'width':width,'height':height})
        page.locator('#fit').click()
        page.locator('.writing-help').evaluate('(e)=>e.open=true')
        page.wait_for_timeout(100)
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),(width,'page overflow')
        for selector in ['#save','#verify']:
            button=page.locator(selector)
            button.scroll_into_view_if_needed()
            expect(button).to_be_in_viewport()
            assert button.evaluate('e=>{const r=e.getBoundingClientRect();return e.contains(document.elementFromPoint(r.x+r.width/2,r.y+r.height/2))}'),(width,selector,'clipped')
        if width>800:
            assert page.locator('.editor-footer').evaluate('e=>{const r=e.getBoundingClientRect(),w=document.querySelector(".workspace").getBoundingClientRect();return r.bottom<=w.bottom+1}'),(width,'footer outside grid')
            assert page.locator('.editor-fields').evaluate('e=>{e.scrollTop=100000;return e.scrollHeight<=e.clientHeight || e.scrollTop>0}'),(width,'cannot scroll editor')
        viewer=page.locator('#viewer')
        before=viewer.evaluate('e=>[e.clientWidth,e.clientHeight]')
        page.locator('#open-image').click()
        expect(page.locator('#zoom-value')).to_have_text('100%')
        page.locator('#zoom-in').click()
        expect(page.locator('#zoom-value')).to_have_text('130%')
        assert viewer.evaluate('e=>[e.clientWidth,e.clientHeight]')==before,(width,'viewer expanded')
        assert viewer.evaluate('e=>{e.scrollTop=100;e.scrollLeft=100;return e.scrollTop>0 || e.scrollLeft>0}'),(width,'image cannot scroll')
        page.locator('#zoom-out').click()
        expect(page.locator('#zoom-value')).to_have_text('100%')
        page.locator('#rotate').click()
        page.locator('#open-image').click()
        assert page.locator('#source-image').evaluate('e=>{const r=e.getBoundingClientRect(),s=e.parentElement.getBoundingClientRect();return r.left>=s.left && r.top>=s.top && r.right<=s.right && r.bottom<=s.bottom}'),(width,'rotated image clipped')
        assert len(page.context.pages)==1,'zoom opened a new window'
        page.locator('#fit').click()
        print(f'{width}x{height}: buttons reachable, scroll and inline zoom passed',flush=True)
    assert not errors,errors
    browser.close()
