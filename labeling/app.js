'use strict';
const $=id=>document.getElementById(id);
const savedVersion=localStorage.getItem('hybridocr-labeling-version');
const store={version:savedVersion==='v1'?'v1':'v2',items:[],item:null,bboxDraft:null,selectedBoxId:null,proposalManifest:null,drawMode:false,boxDrag:null,filter:'pending',author:localStorage.getItem('hybridocr-author') || '',client:localStorage.getItem('hybridocr-client') || crypto.randomUUID(),busy:false,dirty:false,zoom:1,rotation:0,fit:true,request:0,connected:false};
localStorage.setItem('hybridocr-client',store.client);
const statuses={unlabeled:'미작성',draft:'작성 중',verified:'검수 완료'};
function toast(message){$('toast').textContent=message;$('toast').classList.remove('hidden');clearTimeout(toast.timer);toast.timer=setTimeout(()=>$('toast').classList.add('hidden'),5500);}
async function api(action,payload=null,extra={}){
  const query=new URLSearchParams({action,client_id:store.client,version:store.version,...extra});
  const options={cache:'no-store'};
  if(payload){options.method='POST';options.headers={'Content-Type':'application/json'};options.body=JSON.stringify({...payload,author:store.author,client_id:store.client});}
  const response=await fetch('/api/labeling?'+query,options);
  const data=await response.json().catch(()=>({error:'서버 응답을 읽지 못했습니다. 연결을 확인해 주세요.'}));
  if(!response.ok)throw new Error(data.error || '요청을 처리하지 못했습니다.');
  return data;
}
function owns(){return Boolean(store.item?.claim?.mine && Date.parse(store.item.claim.until)>Date.now());}
function updateControls(){
  const own=owns() && !(store.version==='v1' && store.item?.seed_verified);
  const currentStatus=store.version==='v2'?store.item?.bbox_annotation?.status:store.item?.status;
  $('transcription').disabled=$('notes').disabled=store.version!=='v1' || !own || store.busy;
  $('bbox-text').disabled=$('bbox-notes').disabled=$('bbox-order').disabled=$('bbox-uncertain').disabled=$('no-text').disabled=store.version!=='v2' || !own || store.busy;
  $('claim').classList.toggle('hidden',own);$('claim').disabled=!store.item || store.busy || store.item?.seed_verified || Boolean(store.item?.claim && !store.item.claim.mine);
  $('claim').textContent=currentStatus==='verified'?(store.version==='v2'?'bbox 수정하기':'이 전사 수정하기'):'이 이미지 맡기';
  for(const id of ['save','verify','release']){$(id).classList.toggle('hidden',!own);$(id).disabled=store.busy;}
  $('next').disabled=store.busy || !store.connected;$('identity').disabled=store.busy;
  $('insert-tab')?.toggleAttribute('disabled',store.version!=='v1' || !own || store.busy);
  $('draw-box').disabled=store.version!=='v2' || !own || store.busy;
  $('import-proposals').disabled=store.version!=='v2' || !own || store.busy || !candidateLines().length;
  $('delete-box').disabled=$('copy-prediction').disabled=store.version!=='v2' || !own || store.busy || !selectedBox();
  $('bbox-overlay').classList.toggle('draw-active',store.version==='v2' && store.drawMode && own);
  $('bbox-overlay').classList.toggle('read-only',store.version!=='v2' || !own || store.busy);
}
function renderList(){
  const search=$('search').value.toLowerCase().trim();
  const items=store.items.filter(i=>store.filter==='all' || store.filter==='verified' && i.status==='verified' || store.filter==='pending' && i.status!=='verified' || store.filter==='mine' && (i.claim?.mine || store.author && i.annotator===store.author)).filter(i=>`${i.id} ${i.annotator} ${i.claim?.name || ''}`.toLowerCase().includes(search));
  $('image-list').replaceChildren();$('list-count').textContent=items.length;
  if(!items.length){const e=document.createElement('div');e.className='loading';e.textContent='조건에 맞는 이미지가 없습니다.';$('image-list').append(e);}
  for(const item of items){
    const count=store.version==='v2'?`${item.boxes || 0}개 줄`:'';
    const row=document.createElement('button');row.className='image-row'+(store.item?.id===item.id?' selected':'');row.dataset.id=item.id;row.setAttribute('aria-label',`${item.id} ${statuses[item.status]}`);
    const symbol=document.createElement('span');symbol.className='row-symbol';symbol.textContent=item.status==='verified'?'✓':'▤';
    const details=document.createElement('span');details.className='row-details';const title=document.createElement('div');title.className='row-title';title.textContent=item.id;
    const meta=document.createElement('div');meta.className='row-meta';meta.textContent=store.version==='v1' && item.seed_verified?'기존 정답 · 참고용':item.claim?`${item.claim.name} 님 작업 중`:store.version==='v2' && item.no_text?`${statuses[item.status]} · 텍스트 없음${item.annotator?' · '+item.annotator:''}`:item.annotator?`${statuses[item.status]} · ${item.annotator}${count?' · '+count:''}`:store.version==='v2'?`아직 bbox가 없습니다${count?' · '+count:''}`:'아직 작성되지 않았어요';details.append(title,meta);
    const dot=document.createElement('span');dot.className='row-dot '+(item.claim?'busy':item.status);row.append(symbol,details,dot);row.onclick=()=>select(item.id);row.disabled=store.busy;$('image-list').append(row);
  }
}
async function refresh(){
  const version=store.version;
  try{
    const data=await api('list');if(version!==store.version)return;store.items=data.items;store.connected=true;
    const stats=data.stats;const percent=Math.round(stats.verified/stats.total*100);$('progress-value').textContent=percent+'%';$('progress-bar').style.width=percent+'%';
    $('progress-note').textContent=store.version==='v2'?`${stats.verified}장 bbox 검수 완료`:`${stats.verified}장 전사 검수 완료`;$('remaining-note').textContent=`${stats.remaining}장 남음`;$('pending-count').textContent=stats.remaining;
    const current=store.items.find(i=>i.id===store.item?.id);
    if(current){store.item.claim=current.claim;updateClaimMessage();const revision=store.version==='v2'?store.item.bbox_annotation.revision:store.item.revision;if(current.revision!==revision){$('save-state').textContent='새 수정 있음 · 다시 맡기';}}
    $('sync-time').textContent='최근 동기화 '+new Date().toLocaleTimeString('ko-KR',{hour:'2-digit',minute:'2-digit'});$('connection-error').classList.add('hidden');renderList();updateControls();
  }catch(error){if(version!==store.version)return;store.connected=false;$('connection-error').textContent=error.message;$('connection-error').classList.remove('hidden');updateControls();}
}
function draftKey(){return `hybridocr-labeling-draft:${store.version}:${store.client}:${store.item.id}`;}
function legacyDraftKey(){return `hybridocr-labeling-draft:${store.client}:${store.item.id}`;}
function backup(){
  if(!store.item || !owns())return;
  try{
    const draft=store.version==='v2'
      ?{bboxDraft:store.bboxDraft,author:store.author,revision:store.item.bbox_annotation.revision}
      :{text:$('transcription').value,notes:$('notes').value,author:store.author,revision:store.item.revision};
    localStorage.setItem(draftKey(),JSON.stringify(draft));
  }catch{}
}
function markDirty(){store.dirty=true;$('save-state').textContent='저장 전';$('save-state').classList.add('dirty');$('char-count').textContent=$('transcription').value.length.toLocaleString()+'자';backup();}
function markBBoxDirty(){store.dirty=true;$('save-state').textContent='저장 전';$('save-state').classList.add('dirty');backup();renderBoxList();renderBBoxOverlay();updateControls();}
function updateClaimMessage(){
  const item=store.item;if(!item)return;
  $('claim-message').textContent=store.version==='v1' && item.seed_verified?'기존에 검수한 정답입니다. 원본·표기 기준을 참고하세요.':owns()?`${store.author} 님의 ${store.version.toUpperCase()} 작업입니다. 25분간 보호되며 작업 중 자동 연장됩니다.`:item.claim?`${item.claim.name} 님이 작성 중입니다. 지금은 읽기만 가능합니다.`:'작업을 맡으면 다른 작성자와 겹치지 않게 입력할 수 있습니다.';
}
function setModeUI(){
  const v2=store.version==='v2';
  document.querySelectorAll('[data-version]').forEach(button=>{const active=button.dataset.version===store.version;button.classList.toggle('active',active);button.setAttribute('aria-pressed',String(active));});
  $('v1-fields').hidden=v2;$('v2-fields').hidden=!v2;$('bbox-overlay').classList.toggle('hidden',!v2);
  $('workspace-heading').textContent=v2?'한 줄씩, 정확하게.':'한 장씩, 정확하게.';
  $('workspace-summary').textContent=v2?'bbox 후보를 검수하고 줄마다 정답 문구를 입력해 주세요.':'이미지를 직접 읽고 전체 전사를 작성해 주세요. V1 라벨은 별도로 보존됩니다.';
  $('progress-label').textContent=v2?'V2 bbox 검수 진행률':'V1 전체 전사 진행률';
  $('editor-eyebrow').textContent=v2?'줄별 정답 입력':'직접 읽고 작성';
  $('editor-title').textContent=v2?'V2 bbox 전사':'정답 전사';
  $('viewer-hint').textContent=v2?'bbox를 선택해 편집하거나 ‘박스 그리기’를 눌러 추가하세요.':'확대 후 드래그로 이미지를 이동하며 전체 전사를 확인하세요.';
  $('viewer-mode').textContent=v2?'초록 점선은 OCR 후보 · 주황 실선은 정답 bbox':'이미지 전체 정답 전사 · V1 상태';
  $('editor-footer-note').textContent=v2?'작성 중에는 임시 저장 · 모든 줄을 확인한 뒤 검수 완료':'작성 중에는 임시 저장 · 원본 대조 후 검수 완료';
  if(!v2)store.drawMode=false;
  $('draw-box').classList.toggle('active',store.drawMode);$('draw-box').textContent=store.drawMode?'박스 그리기 중 · 드래그하세요':'＋ 박스 그리기';
  $('bbox-overlay').classList.toggle('draw-active',v2 && store.drawMode);
}
async function addV1Reference(item){
  if(store.version!=='v2')return item;
  try{const reference=await api('item',null,{id:item.id,version:'v1'});item.v1_reference=reference.text || '';}catch{item.v1_reference='';}
  return item;
}
function showItem(item,restore=false){
  const changed=store.item?.id!==item.id;if(changed){store.drawMode=false;store.boxDrag=null;}store.item=item;store.dirty=false;
  store.selectedBoxId=null;setModeUI();
  const status=store.version==='v2'?item.bbox_annotation.status:item.status;
  $('image-title').textContent=item.id;$('image-status').className='badge '+(item.claim?'busy':status==='unlabeled'?'neutral':status);$('image-status').textContent=store.version==='v1' && item.seed_verified?'기존 검수 정답':item.claim?`${item.claim.name} 작업 중`:statuses[status];
  if(store.version==='v1'){
    $('transcription').value=item.text || '';$('notes').value=item.notes || '';$('char-count').textContent=(item.text || '').length.toLocaleString()+'자';
    store.bboxDraft=null;
  }else{
    store.bboxDraft=structuredClone(item.bbox_annotation);$('bbox-notes').value=store.bboxDraft.notes || '';$('no-text').checked=store.bboxDraft.no_text;
    $('v1-reference-text').textContent=item.v1_reference || 'V1에는 아직 저장된 전체 전사가 없습니다.';
  }
  const revision=store.version==='v2'?item.bbox_annotation.revision:item.revision;
  $('save-state').textContent=revision?'서버 저장됨':store.version==='v1' && item.seed_verified?'검수된 정답':'미작성';$('save-state').classList.remove('dirty');
  updateClaimMessage();
  $('history').replaceChildren();
  for(const h of (item.history || []).slice(0,3)){const line=document.createElement('div');line.textContent=`${new Date(h.at).toLocaleString('ko-KR')} · ${h.author} · ${statuses[h.status]}`;$('history').append(line);}
  if(changed){store.rotation=0;store.fit=true;$('source-image').hidden=false;$('source-image').src=item.image.file;$('open-image').href=item.image.file;$('empty-view').classList.add('hidden');}
  renderBoxList();renderBBoxOverlay();updateControls();renderList();
  if(restore && owns()){
    let draft,legacy=false;
    try{draft=JSON.parse(localStorage.getItem(draftKey()) || 'null');if(!draft && store.version==='v1'){draft=JSON.parse(localStorage.getItem(legacyDraftKey()) || 'null');legacy=Boolean(draft);}}catch{}
    const serverRevision=store.version==='v2'?item.bbox_annotation.revision:item.revision;
    if(store.version==='v1' && draft && draft.author===store.author && (draft.text!==$('transcription').value || draft.notes!==$('notes').value) && confirm(draft.revision!==serverRevision?'서버에 더 새로운 수정이 있습니다. 이 브라우저의 이전 미저장 입력을 편집기에 복원할까요?':'이 브라우저에 저장 전 입력이 남아 있습니다. 복원할까요?')){
      $('transcription').value=draft.text;$('notes').value=draft.notes;markDirty();if(legacy)localStorage.removeItem(legacyDraftKey());
    }else if(store.version==='v2' && draft?.bboxDraft && draft.author===store.author && confirm(draft.revision!==serverRevision?'서버에 더 새로운 bbox 수정이 있습니다. 이 브라우저의 미저장 입력을 복원할까요?':'이 브라우저에 저장 전 bbox 입력이 남아 있습니다. 복원할까요?')){
      store.bboxDraft={...store.bboxDraft,boxes:structuredClone(draft.bboxDraft.boxes || []),notes:draft.bboxDraft.notes || '',no_text:Boolean(draft.bboxDraft.no_text),proposal_version:draft.bboxDraft.proposal_version || '',revision:serverRevision};
      $('bbox-notes').value=store.bboxDraft.notes;$('no-text').checked=store.bboxDraft.no_text;markBBoxDirty();
    }
  }
}
function confirmLeave(){return !store.dirty || confirm('아직 서버에 저장하지 않은 내용이 있습니다. 입력은 이 브라우저에 보관됩니다. 이동할까요?');}
async function releaseCurrent(){
  if(owns())await api('release',{id:store.item.id});
}
async function select(id){
  if(store.busy || store.item?.id===id)return;
  if(!confirmLeave())return;
  store.busy=true;updateControls();const request=++store.request;
  try{await releaseCurrent();let item=await api('item',null,{id});item=await addV1Reference(item);if(request===store.request)showItem(item,true);}
  catch(error){toast(error.message);}finally{store.busy=false;updateControls();renderList();refresh();}
}
function requireName(){if(store.author)return true;$('name-dialog').showModal();$('author-name').focus();return false;}
async function claim(next=false){
  if(store.busy || !requireName())return;
  if(next && !confirmLeave())return;
  store.busy=true;updateControls();
  try{const old=store.item?.id;if(next)await releaseCurrent();let item=await api(next?'next':'claim',next?{exclude_id:old}:{id:store.item.id});item=await addV1Reference(item);showItem(item,true);(store.version==='v2'?$('draw-box'):$('transcription')).focus();}
  catch(error){toast(error.message);}finally{store.busy=false;updateControls();refresh();}
}
async function save(status){
  if(store.version==='v2')return saveBBox(status);
  if(store.busy || !owns())return;
  if(status==='verified' && !$('transcription').value.trim()){toast('전사 내용을 입력한 뒤 검수 완료해 주세요.');return;}
  const payload={id:store.item.id,text:$('transcription').value,notes:$('notes').value,status,revision:store.item.revision};
  const key=draftKey();store.busy=true;updateControls();$('save-state').textContent='저장 중…';
  try{const item=await api('save',payload);localStorage.removeItem(key);showItem(item);toast(status==='verified'?'검수 완료했습니다. 다음 이미지를 맡아 주세요.':'공유 저장소에 임시 저장했습니다.');}
  catch(error){$('save-state').textContent='저장 실패 · 입력 유지';$('save-state').classList.add('dirty');toast(error.message);}
  finally{store.busy=false;updateControls();refresh();}
}
async function saveBBox(status){
  if(store.busy || !owns() || !store.bboxDraft)return;
  const draft=store.bboxDraft;
  if(status==='verified' && (draft.no_text?draft.boxes.length!==0:draft.boxes.length===0 || draft.boxes.some(box=>!box.text.trim() || box.uncertain))){
    toast('모든 bbox에 정답을 입력하고 불확실 표시를 해제하거나, 텍스트 없음으로 표시한 뒤 검수 완료해 주세요.');return;
  }
  const payload={id:store.item.id,revision:store.item.bbox_annotation.revision,status,boxes:draft.boxes,notes:draft.notes,no_text:draft.no_text,proposal_version:draft.proposal_version};
  const key=draftKey(),reference=store.item.v1_reference;store.busy=true;updateControls();$('save-state').textContent='V2 저장 중…';
  try{const item=await api('save',payload);item.v1_reference=reference;localStorage.removeItem(key);showItem(item);toast(status==='verified'?'V2 bbox 검수를 완료했습니다.':'V2 bbox 초안을 저장했습니다.');}
  catch(error){$('save-state').textContent='저장 실패 · 입력 유지';$('save-state').classList.add('dirty');toast(error.message);}
  finally{store.busy=false;updateControls();refresh();}
}
async function switchVersion(version){
  if(!['v1','v2'].includes(version) || version===store.version || store.busy)return;
  if(!confirmLeave())return;
  store.busy=true;updateControls();
  try{
    await releaseCurrent();store.version=version;localStorage.setItem('hybridocr-labeling-version',version);
    store.item=null;store.items=[];store.bboxDraft=null;store.selectedBoxId=null;store.dirty=false;store.drawMode=false;store.boxDrag=null;store.connected=false;
    $('empty-view').classList.remove('hidden');$('source-image').hidden=true;$('source-image').removeAttribute('src');$('image-stage').style.width='';$('image-stage').style.height='';$('image-visual').style.width='';$('image-visual').style.height='';$('image-visual').style.transform='';
    $('image-title').textContent='이미지를 선택하세요';$('image-status').className='badge neutral';$('image-status').textContent='준비';$('save-state').textContent='선택 대기';$('claim-message').textContent='작업을 맡은 뒤 입력할 수 있습니다.';$('history').replaceChildren();
    $('transcription').value='';$('notes').value='';$('bbox-text').value='';$('bbox-notes').value='';$('no-text').checked=false;$('selected-box').hidden=true;
    setModeUI();renderBoxList();renderBBoxOverlay();await refresh();
  }catch(error){toast(error.message);}
  finally{store.busy=false;updateControls();}
}
function resizeImage(){
  const img=$('source-image');if(!img.naturalWidth)return;const rotated=store.rotation%180!==0;
  const w=rotated?img.naturalHeight:img.naturalWidth,h=rotated?img.naturalWidth:img.naturalHeight;
  const visual=$('image-visual'),stage=$('image-stage'),padding=getComputedStyle(stage);
  const horizontalPadding=parseFloat(padding.paddingLeft)+parseFloat(padding.paddingRight),verticalPadding=parseFloat(padding.paddingTop)+parseFloat(padding.paddingBottom);
  if(store.fit)store.zoom=Math.min(1,Math.max(.05,Math.min(($('viewer').clientWidth-horizontalPadding)/w,($('viewer').clientHeight-verticalPadding)/h)));
  img.style.width=img.naturalWidth*store.zoom+'px';img.style.height=img.naturalHeight*store.zoom+'px';
  visual.style.width=img.naturalWidth*store.zoom+'px';visual.style.height=img.naturalHeight*store.zoom+'px';visual.style.transform=`rotate(${store.rotation}deg)`;
  stage.style.width=w*store.zoom+horizontalPadding+'px';stage.style.height=h*store.zoom+verticalPadding+'px';$('zoom-value').textContent=Math.round(store.zoom*100)+'%';
  if(store.version==='v2')renderBBoxOverlay();
}
const svgNS='http://www.w3.org/2000/svg';
function selectedBox(){return store.bboxDraft?.boxes.find(box=>box.id===store.selectedBoxId) || null;}
function candidateRecord(){
  if(store.version!=='v2' || !store.item)return null;
  const entry=store.proposalManifest?.images?.[store.item.id];if(!entry)return null;
  const size=entry.source_size, image=store.item.image;
  return size?.width===image.source_width && size?.height===image.source_height && entry.file===image.file && entry.input_sha256===entry.asset_sha256 && entry.asset_sha256===image.asset_sha256 && entry.source_sha256===image.source_sha256?entry:null;
}
function candidateLines(){return candidateRecord()?.lines || [];}
function renderBoxList(){
  const list=$('box-list');if(!list)return;list.replaceChildren();
  const boxes=store.bboxDraft?.boxes || [];const candidates=candidateLines();
  $('bbox-count').textContent=`${boxes.length}개 줄`;
  if(!boxes.length){
    const empty=document.createElement('div');empty.className='bbox-empty';
    empty.textContent=store.bboxDraft?.no_text?'텍스트 없음으로 표시했습니다.':candidates.length?`${candidates.length}개 OCR 후보가 있습니다. 가져온 뒤 줄별 정답을 입력하세요.`:'박스를 그리거나 OCR 후보를 가져오세요.';
    list.append(empty);
  }
  for(const box of [...boxes].sort((a,b)=>a.order-b.order)){
    const row=document.createElement('button');row.type='button';row.className='box-row'+(box.id===store.selectedBoxId?' selected':'');
    const number=document.createElement('span');number.className='box-number';number.textContent=box.order;
    const body=document.createElement('span');body.className='box-row-content';
    const text=document.createElement('strong');text.textContent=box.text.trim() || box.prediction || '정답 입력 필요';
    const meta=document.createElement('small');meta.textContent=box.uncertain?'판독 불확실':box.text.trim()?'정답 입력됨':box.prediction?'OCR 후보만 있음':'수동 bbox';
    body.append(text,meta);row.append(number,body);row.onclick=()=>selectBox(box.id);list.append(row);
  }
}
function selectBox(id){
  store.selectedBoxId=id;const box=selectedBox(),panel=$('selected-box');panel.hidden=!box;
  if(box){
    $('selected-box-title').textContent=`${box.order}번 줄 · ${box.id}`;
    $('prediction-text').textContent=box.prediction || 'OCR 예측이 없습니다.';
    $('bbox-text').value=box.text;$('bbox-order').value=box.order;$('bbox-uncertain').checked=box.uncertain;
  }
  renderBoxList();renderBBoxOverlay();updateControls();
}
function renderBBoxOverlay(){
  const overlay=$('bbox-overlay');if(!overlay)return;overlay.replaceChildren();
  const item=store.version==='v2'?store.item:null;
  overlay.classList.toggle('hidden',!item);
  if(!item)return;
  const width=item.image.source_width,height=item.image.source_height;
  overlay.setAttribute('viewBox',`0 0 ${width} ${height}`);overlay.setAttribute('preserveAspectRatio','none');
  const record=candidateRecord();
  if(store.proposalManifest && store.proposalManifest.images?.[item.id] && !record)$('proposal-status').textContent='후보 이미지 크기·파일 fingerprint 불일치 · 수동 bbox를 사용하세요';
  else if(record)$('proposal-status').textContent=`${record.lines.length}개 OCR 후보 · ${record.proposal_version || store.proposalManifest.proposal_version}`;
  else if(store.proposalManifest)$('proposal-status').textContent='이 이미지의 OCR 후보 없음 · 수동 bbox 가능';
  for(const line of candidateLines()){
    if(!Array.isArray(line.bbox) || line.bbox.length!==4)continue;
    const rect=document.createElementNS(svgNS,'rect');rect.setAttribute('class','candidate-box');
    rect.setAttribute('x',line.bbox[0]);rect.setAttribute('y',line.bbox[1]);rect.setAttribute('width',line.bbox[2]-line.bbox[0]);rect.setAttribute('height',line.bbox[3]-line.bbox[1]);rect.setAttribute('pointer-events','none');
    const confidence=line.confidence==null?'':` · 신뢰도 ${Number(line.confidence).toFixed(3)}`;
    const title=document.createElementNS(svgNS,'title');title.textContent=`OCR 후보 ${line.order}: ${line.prediction || '(문구 없음)'}${confidence}`;rect.append(title);overlay.append(rect);
  }
  for(const box of [...(store.bboxDraft?.boxes || [])].sort((a,b)=>a.order-b.order)){
    const [x1,y1,x2,y2]=box.bbox,group=document.createElementNS(svgNS,'g');group.dataset.boxId=box.id;
    const rect=document.createElementNS(svgNS,'rect');rect.setAttribute('class','bbox-box'+(box.id===store.selectedBoxId?' active':''));
    rect.setAttribute('x',x1);rect.setAttribute('y',y1);rect.setAttribute('width',x2-x1);rect.setAttribute('height',y2-y1);rect.dataset.boxId=box.id;group.append(rect);
    const label=document.createElementNS(svgNS,'text');label.setAttribute('class','bbox-label');label.setAttribute('font-size',28/Math.max(.05,store.zoom));label.setAttribute('x',x1+5);label.setAttribute('y',Math.max(25,y1-7));label.textContent=`${box.order}. ${(box.text || box.prediction || '').slice(0,24)}`;label.dataset.boxId=box.id;group.append(label);
    if(box.id===store.selectedBoxId){
      for(const [handle,x,y] of [['nw',x1,y1],['ne',x2,y1],['sw',x1,y2],['se',x2,y2]]){
        const circle=document.createElementNS(svgNS,'circle');circle.setAttribute('class',`bbox-handle handle-${handle}`);circle.setAttribute('cx',x);circle.setAttribute('cy',y);circle.setAttribute('r',12/Math.max(.05,store.zoom));circle.dataset.boxId=box.id;circle.dataset.handle=handle;group.append(circle);
      }
    }
    overlay.append(group);
  }
}
$('source-image').onload=()=>{resizeImage();$('viewer').scrollTo(0,0);};$('source-image').onerror=()=>toast('이미지를 불러오지 못했습니다. 배포 파일을 확인해 주세요.');
$('zoom-in').onclick=()=>{store.fit=false;store.zoom=Math.min(3,store.zoom*1.3);resizeImage();};$('zoom-out').onclick=()=>{store.fit=false;store.zoom=Math.max(.05,store.zoom/1.3);resizeImage();};$('fit').onclick=()=>{store.fit=true;resizeImage();$('viewer').scrollTo(0,0);};$('rotate').onclick=()=>{store.rotation=(store.rotation+90)%360;store.fit=true;resizeImage();};
function clamp(value,min,max){return Math.max(min,Math.min(max,value));}
function svgPoint(event){
  const overlay=$('bbox-overlay'),matrix=overlay.getScreenCTM();if(!matrix)return null;
  const point=overlay.createSVGPoint();point.x=event.clientX;point.y=event.clientY;
  const converted=point.matrixTransform(matrix.inverse());
  return {x:Math.round(converted.x),y:Math.round(converted.y)};
}
function boundedImagePoint(event){
  const point=svgPoint(event),image=store.item?.image;if(!point || !image)return null;
  return {x:clamp(point.x,0,image.source_width),y:clamp(point.y,0,image.source_height)};
}
function setDrawPreview(start,end){
  const overlay=$('bbox-overlay');let rect=overlay.querySelector('.draw-preview');
  if(!rect){rect=document.createElementNS(svgNS,'rect');rect.setAttribute('class','draw-preview');overlay.append(rect);}
  rect.setAttribute('x',Math.min(start.x,end.x));rect.setAttribute('y',Math.min(start.y,end.y));
  rect.setAttribute('width',Math.abs(end.x-start.x));rect.setAttribute('height',Math.abs(end.y-start.y));
}
function applyBoxPointer(event){
  const drag=store.boxDrag;if(!drag)return;const point=svgPoint(event);if(!point)return;
  const box=store.bboxDraft?.boxes.find(item=>item.id===drag.boxId);if(!box)return;
  const [ox1,oy1,ox2,oy2]=drag.original,[width,height]=[store.item.image.source_width,store.item.image.source_height];
  if(drag.type==='move'){
    const dx=clamp(point.x-drag.start.x,-ox1,width-ox2),dy=clamp(point.y-drag.start.y,-oy1,height-oy2);
    box.bbox=[ox1+dx,oy1+dy,ox2+dx,oy2+dy];
  }else{
    let [x1,y1,x2,y2]=drag.original;const handle=drag.handle;
    if(handle.includes('w'))x1=clamp(point.x,0,x2-2);if(handle.includes('e'))x2=clamp(point.x,x1+2,width);
    if(handle.includes('n'))y1=clamp(point.y,0,y2-2);if(handle.includes('s'))y2=clamp(point.y,y1+2,height);
    box.bbox=[x1,y1,x2,y2];
  }
  drag.moved=drag.moved || Math.abs(point.x-drag.start.x)>1 || Math.abs(point.y-drag.start.y)>1;
  renderBBoxOverlay();
}
$('bbox-overlay').onpointerdown=event=>{
  if(event.button!==0 || store.version!=='v2' || !store.item || store.busy || !owns())return;
  const overlay=$('bbox-overlay'),target=event.target.closest?.('[data-box-id]'),point=target?svgPoint(event):boundedImagePoint(event);if(!point)return;
  if(target){
    event.preventDefault();event.stopPropagation();const boxId=target.dataset.boxId;selectBox(boxId);
    if(!owns())return;
    const handle=target.dataset.handle || '';
    store.boxDrag={type:handle?'resize':'move',boxId,handle,start:point,original:[...selectedBox().bbox],moved:false};overlay.setPointerCapture(event.pointerId);return;
  }
  if(!store.drawMode || !owns())return;
  event.preventDefault();event.stopPropagation();
  store.boxDrag={type:'draw',start:point,current:point};overlay.setPointerCapture(event.pointerId);setDrawPreview(point,point);
};
$('bbox-overlay').onpointermove=event=>{
  if(!store.boxDrag || store.busy)return;
  if(store.boxDrag.type==='draw'){
    const point=boundedImagePoint(event);if(point){store.boxDrag.current=point;setDrawPreview(store.boxDrag.start,point);}
  }else applyBoxPointer(event);
};
$('bbox-overlay').onpointerup=event=>{
  const drag=store.boxDrag;if(!drag)return;
  if(store.busy || !owns()){
    if(drag.type==='draw')$('bbox-overlay').querySelector('.draw-preview')?.remove();
    else {const box=store.bboxDraft?.boxes.find(item=>item.id===drag.boxId);if(box)box.bbox=drag.original;renderBBoxOverlay();}
    store.boxDrag=null;return;
  }
  if(drag.type==='draw'){
    const end=boundedImagePoint(event) || drag.current,start=drag.start,x1=Math.min(start.x,end.x),y1=Math.min(start.y,end.y),x2=Math.max(start.x,end.x),y2=Math.max(start.y,end.y);
    $('bbox-overlay').querySelector('.draw-preview')?.remove();
    if(x2-x1>=6 && y2-y1>=6){
      store.bboxDraft.no_text=false;$('no-text').checked=false;
      const box={id:`line-${crypto.randomUUID()}`,bbox:[x1,y1,x2,y2],text:'',prediction:'',order:store.bboxDraft.boxes.length+1,uncertain:false};
      store.bboxDraft.boxes.push(box);store.selectedBoxId=box.id;selectBox(box.id);markBBoxDirty();
    }
  }else{
    applyBoxPointer(event);if(drag.moved)markBBoxDirty();else renderBBoxOverlay();
  }
  store.boxDrag=null;
};
$('bbox-overlay').onpointercancel=()=>{
  const drag=store.boxDrag;if(!drag)return;
  if(drag.type==='draw')$('bbox-overlay').querySelector('.draw-preview')?.remove();
  else {const box=store.bboxDraft?.boxes.find(item=>item.id===drag.boxId);if(box)box.bbox=drag.original;renderBBoxOverlay();}
  store.boxDrag=null;
};
let pan=null;
$('viewer').onpointerdown=e=>{if(e.button!==0 || !store.item)return;pan={x:e.clientX,y:e.clientY,left:$('viewer').scrollLeft,top:$('viewer').scrollTop};$('viewer').setPointerCapture(e.pointerId);$('viewer').classList.add('dragging');};
$('viewer').onpointermove=e=>{if(pan){$('viewer').scrollLeft=pan.left-(e.clientX-pan.x);$('viewer').scrollTop=pan.top-(e.clientY-pan.y);}};
$('viewer').onpointerup=$('viewer').onpointercancel=()=>{pan=null;$('viewer').classList.remove('dragging');};window.addEventListener('resize',()=>{if(store.fit)resizeImage();});
function insertTranscriptionTab(){
  const t=$('transcription');
  if(t.disabled || t.readOnly || store.busy)return;
  if(t.value.length-(t.selectionEnd-t.selectionStart)+1>t.maxLength)return;
  t.setRangeText('\t',t.selectionStart,t.selectionEnd,'end');markDirty();t.focus();
}
async function loadProposals(){
  try{
    const response=await fetch('/data/text-box-proposals.json',{cache:'no-store'});
    if(!response.ok)throw new Error(response.status===404?'후보 sidecar 없음':'후보 파일을 읽지 못함');
    const manifest=await response.json();
    if(manifest.schema_version!==1 || !manifest.images || typeof manifest.images!=='object')throw new Error('후보 파일 형식 오류');
    store.proposalManifest=manifest;$('proposal-status').textContent=`${manifest.engine || 'OCR'} · ${manifest.proposal_version || '후보 준비됨'}`;
  }catch(error){store.proposalManifest=null;$('proposal-status').textContent=error.message.includes('없음')?'OCR 후보 파일 없음 · 수동 bbox 가능':'OCR 후보를 사용할 수 없음 · 수동 bbox 가능';}
  renderBoxList();renderBBoxOverlay();updateControls();
}
function setDrawMode(enabled){
  store.drawMode=Boolean(enabled);$('draw-box').classList.toggle('active',store.drawMode);
  $('draw-box').textContent=store.drawMode?'박스 그리기 중 · 드래그하세요':'＋ 박스 그리기';
  $('bbox-overlay').classList.toggle('draw-active',store.version==='v2' && store.drawMode && owns());
}
function importProposals(){
  if(!owns() || !store.bboxDraft)return;
  const record=candidateRecord(),lines=record?.lines || [];
  if(!lines.length){toast('가져올 OCR bbox 후보가 없습니다.');return;}
  if(store.bboxDraft.boxes.length && !confirm('현재 bbox와 정답 문구를 OCR 후보로 교체할까요? 아직 저장하지 않은 편집 내용도 바뀝니다.'))return;
  const width=store.item.image.source_width,height=store.item.image.source_height,ids=new Set();
  const boxes=[];
  for(const [index,line] of lines.entries()){
    const bbox=line.bbox;
    if(!Array.isArray(bbox) || bbox.length!==4 || !bbox.every(Number.isInteger) || bbox[0]<0 || bbox[1]<0 || bbox[2]>width || bbox[3]>height || bbox[0]>=bbox[2] || bbox[1]>=bbox[3])continue;
    const baseId=typeof line.id==='string' && /^[a-zA-Z0-9_-]{1,80}$/.test(line.id)?line.id:`proposal-${index+1}`;
    let id=baseId,suffix=1;while(ids.has(id))id=`${baseId}-${suffix++}`;ids.add(id);
    boxes.push({id,bbox:[...bbox],text:'',prediction:String(line.prediction || ''),order:boxes.length+1,uncertain:false});
  }
  if(!boxes.length){toast('유효한 bbox 후보가 없습니다. 이미지를 확인해 주세요.');return;}
  store.bboxDraft.boxes=boxes;store.bboxDraft.no_text=false;store.bboxDraft.proposal_version=record.proposal_version || store.proposalManifest.proposal_version || '';
  $('no-text').checked=false;setDrawMode(false);store.selectedBoxId=boxes[0].id;selectBox(store.selectedBoxId);markBBoxDirty();
  toast(`${boxes.length}개 후보를 가져왔습니다. OCR 문구는 정답에 복사되지 않았습니다.`);
}
function reorderSelected(value){
  const box=selectedBox();if(!box)return;
  const order=Number(value);if(!Number.isInteger(order))return;
  const boxes=[...store.bboxDraft.boxes].sort((a,b)=>a.order-b.order).filter(item=>item.id!==box.id);
  boxes.splice(clamp(order,1,store.bboxDraft.boxes.length)-1,0,box);boxes.forEach((item,index)=>item.order=index+1);
  $('bbox-order').value=box.order;markBBoxDirty();
}
function updateSelectedBox(change){
  const box=selectedBox();if(!box || !owns())return;
  Object.assign(box,change);markBBoxDirty();
}
$('insert-tab').onclick=insertTranscriptionTab;
$('transcription').addEventListener('keydown',e=>{
  if(e.key==='Tab' && !e.shiftKey && !e.ctrlKey && !e.altKey && !e.metaKey && !e.isComposing){
    e.preventDefault();insertTranscriptionTab();
  }
});
$('transcription').oninput=$('notes').oninput=markDirty;$('claim').onclick=()=>claim();$('next').onclick=()=>claim(true);$('save').onclick=()=>save('draft');$('verify').onclick=()=>save('verified');
$('release').onclick=async()=>{if(store.busy || !confirmLeave())return;store.busy=true;updateControls();try{await releaseCurrent();let item=await api('item',null,{id:store.item.id});item=await addV1Reference(item);showItem(item);toast('작업을 반납했습니다.');}catch(error){toast(error.message);}finally{store.busy=false;updateControls();refresh();}};
$('draw-box').onclick=()=>setDrawMode(!store.drawMode);$('import-proposals').onclick=importProposals;
$('bbox-text').oninput=()=>updateSelectedBox({text:$('bbox-text').value});
$('bbox-notes').oninput=()=>{if(!owns() || !store.bboxDraft)return;store.bboxDraft.notes=$('bbox-notes').value;markBBoxDirty();};
$('bbox-order').onchange=()=>reorderSelected($('bbox-order').value);
$('bbox-uncertain').onchange=()=>updateSelectedBox({uncertain:$('bbox-uncertain').checked});
$('copy-prediction').onclick=()=>{const box=selectedBox();if(box && owns()){$('bbox-text').value=box.prediction;updateSelectedBox({text:box.prediction});}};
$('delete-box').onclick=()=>{const box=selectedBox();if(!box || !owns() || !confirm('선택한 bbox를 삭제할까요?'))return;store.bboxDraft.boxes=store.bboxDraft.boxes.filter(item=>item.id!==box.id);store.bboxDraft.boxes.sort((a,b)=>a.order-b.order).forEach((item,index)=>item.order=index+1);store.selectedBoxId=null;$('selected-box').hidden=true;markBBoxDirty();};
$('no-text').onchange=()=>{
  if(!owns() || !store.bboxDraft)return;
  if($('no-text').checked && store.bboxDraft.boxes.length && !confirm('텍스트 없음으로 표시하면 현재 bbox를 모두 삭제합니다. 계속할까요?')){$('no-text').checked=false;return;}
  if($('no-text').checked){store.bboxDraft.boxes=[];store.selectedBoxId=null;$('selected-box').hidden=true;}
  store.bboxDraft.no_text=$('no-text').checked;markBBoxDirty();
};
document.querySelectorAll('[data-version]').forEach(button=>button.onclick=()=>switchVersion(button.dataset.version));
$('search').oninput=renderList;document.querySelectorAll('[data-filter]').forEach(button=>button.onclick=()=>{store.filter=button.dataset.filter;document.querySelectorAll('[data-filter]').forEach(b=>b.classList.toggle('active',b===button));renderList();});
$('refresh').onclick=refresh;$('help').onclick=()=>$('help-dialog').showModal();$('close-help').onclick=()=>$('help-dialog').close();
$('identity').onclick=()=>{if(store.busy)return;$('author-name').value=store.author;$('name-dialog').showModal();};$('name-cancel').onclick=()=>$('name-dialog').close();
$('name-form').onsubmit=async e=>{e.preventDefault();const name=$('author-name').value.trim();if(!name)return;
  if(store.author && name!==store.author){if(!confirmLeave())return;try{await releaseCurrent();}catch(error){toast(error.message);return;}}
  store.author=name;localStorage.setItem('hybridocr-author',name);$('identity').textContent=name+' 님';$('name-dialog').close();
  if(store.item){try{let item=await api('item',null,{id:store.item.id});item=await addV1Reference(item);showItem(item);}catch(error){toast(error.message);}}refresh();
};
$('export').onclick=()=>{$('export-description').textContent=store.version==='v2'?'V2 줄 bbox와 줄별 정답만 내려받습니다. V1 이미지 전사와는 별도 파일입니다.':'V1 이미지 전체 전사만 내려받습니다. V2 bbox 정답과는 별도 파일입니다.';$('export-dialog').showModal();};$('close-export').onclick=()=>$('export-dialog').close();
async function download(format){try{
  const scope=document.querySelector('[name=scope]:checked').value;const query=new URLSearchParams({action:'export',version:store.version,format,scope,client_id:store.client});const response=await fetch('/api/labeling?'+query,{cache:'no-store'});
  if(!response.ok){const error=await response.json();throw new Error(error.error);}
  const blob=await response.blob();const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=format==='zip'?`hybridocr-${store.version==='v2'?'v2-bbox-':'v1-'}labels-${scope}.zip`:`hybridocr-${store.version}-labels.json`;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);$('export-dialog').close();
}catch(error){toast(error.message);}}
$('download-zip').onclick=()=>download('zip');$('download-json').onclick=()=>download('json');
window.addEventListener('beforeunload',e=>{if(store.dirty){e.preventDefault();e.returnValue='';}});
window.addEventListener('keydown',e=>{if((e.ctrlKey || e.metaKey) && e.key==='s'){e.preventDefault();save('draft');}});
setInterval(refresh,30000);
setInterval(async()=>{if(owns() && !store.busy){try{const item=await api('renew',{id:store.item.id});store.item.claim=item.claim;updateControls();}catch(error){toast(error.message);}}},3*60*1000);
setModeUI();loadProposals();
if(store.author)$('identity').textContent=store.author+' 님';else{$('name-dialog').showModal();$('author-name').focus();}
refresh();
