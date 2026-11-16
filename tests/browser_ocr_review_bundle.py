import json
from pathlib import Path
from playwright.sync_api import sync_playwright
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--url',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
OUT=Path(args.output);OUT.mkdir(parents=True,exist_ok=False)
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox'])
 page=browser.new_page(viewport={'width':1600,'height':1100},accept_downloads=True);errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto(args.url);page.wait_for_function('window.document.getElementById("summary").textContent.includes("479")')
 assert page.locator('#context').inner_text()=='전사·AI 제안·모델 출력은 숨김'
 assert page.locator('#text').input_value()==''
 page.locator('#text').fill('TEST 한글 English');page.locator('#include').select_option('true');page.locator('#type').select_option('text');page.locator('#reason').fill('TEST browser fixture, not human verification');page.locator('#apply').click();page.wait_for_function('document.getElementById("status").textContent.includes("revision 1")')
 page.locator('#zoom').fill('70');page.locator('#zoom').dispatch_event('input');page.evaluate('document.getElementById("viewport").scrollTop=400');assert page.evaluate('document.getElementById("viewport").scrollTop')==400
 page.locator('#showAI').check();assert 'AI_proposal' in page.locator('#context').inner_text();page.locator('#showAI').uncheck()
 page.locator('#save').click();page.wait_for_function('document.getElementById("status").textContent.includes("부분 저장")')
 with page.expect_download() as dl:page.locator('#export').click()
 export=OUT/'TEST-export.json';dl.value.save_as(str(export));saved=json.load(open(export));assert saved['test_mode'] and not any(r['human_verified'] for p in saved['pages'] for r in p['regions'])
 page.screenshot(path=str(OUT/'edited.png'),full_page=True);browser.close()
 browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox']);page=browser.new_page(viewport={'width':1600,'height':1100});page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto(args.url);page.wait_for_function('document.getElementById("text").value==="TEST 한글 English"')
 page.locator('#reload').click();page.wait_for_function('document.getElementById("status").textContent.includes("재개 완료")')
 page.locator('#import').set_input_files(str(export));page.wait_for_function('document.getElementById("status").textContent.includes("불러오기 완료")')
 imported=page.evaluate('fetch("/api/export").then(r=>r.json())');assert imported==saved
 # Drive actual add/split/merge controls; all saved in test-sessions only.
 page.locator('#reason').fill('TEST lineage browser fixture');page.locator('#add').click();page.locator('#opConfirm').click();page.wait_for_function('document.getElementById("status").textContent.includes("revision 2")')
 page.locator('#split').click();page.locator('#opConfirm').click();page.wait_for_function('document.getElementById("status").textContent.includes("revision 3")')
 boxes=page.locator('.mergeSelect');boxes.nth(1).check();boxes.nth(2).check();page.locator('#merge').click();page.locator('#opConfirm').click();page.wait_for_function('document.getElementById("status").textContent.includes("revision 4")')
 final=page.evaluate('fetch("/api/export").then(r=>r.json())');assert len(final['history'])==4;assert not any(r['human_verified'] for p in final['pages'] for r in p['regions'])
 page.screenshot(path=str(OUT/'resumed-lineage.png'),full_page=True)
 assert not errors,errors
 (OUT/'browser-receipt.json').write_text(json.dumps({'real_browser':'Google Chrome headless via existing Playwright','edited_saved_closed_reopened_exported_imported_equal':True,'add_split_merge_UI':True,'zoom_and_scroll':True,'AI_hidden_by_default':True,'human_verified_count':0,'page_errors':errors,'test_mode':True,'final_revision':final['revision']},indent=2))
 browser.close()
