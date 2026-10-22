'use strict';
const $=id=>document.getElementById(id);
const store={items:[],item:null,filter:'pending',author:localStorage.getItem('hybridocr-author') || '',client:localStorage.getItem('hybridocr-client') || crypto.randomUUID(),busy:false,dirty:false,zoom:1,rotation:0,fit:true,request:0,connected:false};
localStorage.setItem('hybridocr-client',store.client);
const statuses={unlabeled:'미작성',draft:'작성 중',verified:'검수 완료'};
function toast(message){$('toast').textContent=message;$('toast').classList.remove('hidden');clearTimeout(toast.timer);toast.timer=setTimeout(()=>$('toast').classList.add('hidden'),5500);}
async function api(action,payload=null,extra={}){
  const query=new URLSearchParams({action,client_id:store.client,...extra});
  const options={cache:'no-store'};
  if(payload){options.method='POST';options.headers={'Content-Type':'application/json'};options.body=JSON.stringify({...payload,author:store.author,client_id:store.client});}
  const response=await fetch('/api/labeling?'+query,options);
  const data=await response.json().catch(()=>({error:'서버 응답을 읽지 못했습니다. 연결을 확인해 주세요.'}));
  if(!response.ok)throw new Error(data.error || '요청을 처리하지 못했습니다.');
  return data;
}
function owns(){return Boolean(store.item?.claim?.mine && Date.parse(store.item.claim.until)>Date.now());}
function updateControls(){
  const own=owns() && !store.item?.seed_verified;
  $('transcription').disabled=$('notes').disabled=!own || store.busy;
  $('claim').classList.toggle('hidden',own);$('claim').disabled=!store.item || store.busy || store.item?.seed_verified || Boolean(store.item?.claim && !store.item.claim.mine);
  $('claim').textContent=store.item?.status==='verified'?'이 전사 수정하기':'이 이미지 맡기';
  for(const id of ['save','verify','release']){$(id).classList.toggle('hidden',!own);$(id).disabled=store.busy;}
  $('next').disabled=store.busy || !store.connected;$('identity').disabled=store.busy;
  $('insert-tab')?.toggleAttribute('disabled',!own || store.busy);
}
function renderList(){
  const search=$('search').value.toLowerCase().trim();
  const items=store.items.filter(i=>store.filter==='all' || store.filter==='verified' && i.status==='verified' || store.filter==='pending' && i.status!=='verified' || store.filter==='mine' && (i.claim?.mine || store.author && i.annotator===store.author)).filter(i=>`${i.id} ${i.annotator} ${i.claim?.name || ''}`.toLowerCase().includes(search));
  $('image-list').replaceChildren();$('list-count').textContent=items.length;
  if(!items.length){const e=document.createElement('div');e.className='loading';e.textContent='조건에 맞는 이미지가 없습니다.';$('image-list').append(e);}
  for(const item of items){
    const row=document.createElement('button');row.className='image-row'+(store.item?.id===item.id?' selected':'');row.dataset.id=item.id;row.setAttribute('aria-label',`${item.id} ${statuses[item.status]}`);
    const symbol=document.createElement('span');symbol.className='row-symbol';symbol.textContent=item.status==='verified'?'✓':'▤';
    const details=document.createElement('span');details.className='row-details';const title=document.createElement('div');title.className='row-title';title.textContent=item.id;
    const meta=document.createElement('div');meta.className='row-meta';meta.textContent=item.seed_verified?'기존 정답 · 참고용':item.claim?`${item.claim.name} 님 작업 중`:item.annotator?`${statuses[item.status]} · ${item.annotator}`:'아직 작성되지 않았어요';details.append(title,meta);
    const dot=document.createElement('span');dot.className='row-dot '+(item.claim?'busy':item.status);row.append(symbol,details,dot);row.onclick=()=>select(item.id);row.disabled=store.busy;$('image-list').append(row);
  }
}
async function refresh(){
  try{
    const data=await api('list');store.items=data.items;store.connected=true;
    const stats=data.stats;const percent=Math.round(stats.verified/stats.total*100);$('progress-value').textContent=percent+'%';$('progress-bar').style.width=percent+'%';
    $('progress-note').textContent=`${stats.verified}장 검수 완료`;$('remaining-note').textContent=`${stats.remaining}장 남음`;$('pending-count').textContent=stats.remaining;
    const current=store.items.find(i=>i.id===store.item?.id);
    if(current){store.item.claim=current.claim;updateClaimMessage();if(current.revision!==store.item.revision){$('save-state').textContent='새 수정 있음 · 다시 맡기';}}
    $('sync-time').textContent='최근 동기화 '+new Date().toLocaleTimeString('ko-KR',{hour:'2-digit',minute:'2-digit'});$('connection-error').classList.add('hidden');renderList();updateControls();
  }catch(error){store.connected=false;$('connection-error').textContent=error.message;$('connection-error').classList.remove('hidden');updateControls();}
}
function draftKey(){return `hybridocr-labeling-draft:${store.client}:${store.item.id}`;}
function backup(){
  if(!store.item || !owns())return;
  try{localStorage.setItem(draftKey(),JSON.stringify({text:$('transcription').value,notes:$('notes').value,author:store.author,revision:store.item.revision}));}catch{}
}
function markDirty(){store.dirty=true;$('save-state').textContent='저장 전';$('save-state').classList.add('dirty');$('char-count').textContent=$('transcription').value.length.toLocaleString()+'자';backup();}
function updateClaimMessage(){
  const item=store.item;if(!item)return;
  $('claim-message').textContent=item.seed_verified?'기존에 검수한 정답입니다. 원본·표기 기준을 참고하세요.':owns()?`${store.author} 님의 작업입니다. 25분간 보호되며 작업 중 자동 연장됩니다.`:item.claim?`${item.claim.name} 님이 작성 중입니다. 지금은 읽기만 가능합니다.`:'작업을 맡으면 다른 작성자와 겹치지 않게 입력할 수 있습니다.';
}
function showItem(item,restore=false){
  const changed=store.item?.id!==item.id;store.item=item;store.dirty=false;
  $('image-title').textContent=item.id;$('image-status').className='badge '+(item.claim?'busy':item.status==='unlabeled'?'neutral':item.status);$('image-status').textContent=item.seed_verified?'기존 검수 정답':item.claim?`${item.claim.name} 작업 중`:statuses[item.status];
  $('transcription').value=item.text || '';$('notes').value=item.notes || '';$('char-count').textContent=(item.text || '').length.toLocaleString()+'자';
  $('save-state').textContent=item.revision?'서버 저장됨':item.seed_verified?'검수된 정답':'미작성';$('save-state').classList.remove('dirty');
  updateClaimMessage();
  $('history').replaceChildren();
  for(const h of (item.history || []).slice(0,3)){const line=document.createElement('div');line.textContent=`${new Date(h.at).toLocaleString('ko-KR')} · ${h.author} · ${statuses[h.status]}`;$('history').append(line);}
  if(changed){store.rotation=0;store.fit=true;$('source-image').hidden=false;$('source-image').src=item.image.file;$('empty-view').classList.add('hidden');}
  updateControls();renderList();
  if(restore && owns()){
    let draft;try{draft=JSON.parse(localStorage.getItem(draftKey()) || 'null');}catch{}
    if(draft && draft.author===store.author && (draft.text!==$('transcription').value || draft.notes!==$('notes').value) && confirm(draft.revision!==item.revision?'서버에 더 새로운 수정이 있습니다. 이 브라우저의 이전 미저장 입력을 편집기에 복원할까요?':'이 브라우저에 저장 전 입력이 남아 있습니다. 복원할까요?')){
      $('transcription').value=draft.text;$('notes').value=draft.notes;markDirty();
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
  try{await releaseCurrent();const item=await api('item',null,{id});if(request===store.request)showItem(item,true);}
  catch(error){toast(error.message);}finally{store.busy=false;updateControls();renderList();refresh();}
}
function requireName(){if(store.author)return true;$('name-dialog').showModal();$('author-name').focus();return false;}
async function claim(next=false){
  if(store.busy || !requireName())return;
  if(next && !confirmLeave())return;
  store.busy=true;updateControls();
  try{const old=store.item?.id;if(next)await releaseCurrent();const item=await api(next?'next':'claim',next?{exclude_id:old}:{id:store.item.id});showItem(item,true);$('transcription').focus();}
  catch(error){toast(error.message);}finally{store.busy=false;updateControls();refresh();}
}
async function save(status){
  if(store.busy || !owns())return;
  if(status==='verified' && !$('transcription').value.trim()){toast('전사 내용을 입력한 뒤 검수 완료해 주세요.');return;}
  const payload={id:store.item.id,text:$('transcription').value,notes:$('notes').value,status,revision:store.item.revision};
  const key=draftKey();store.busy=true;updateControls();$('save-state').textContent='저장 중…';
  try{const item=await api('save',payload);localStorage.removeItem(key);showItem(item);toast(status==='verified'?'검수 완료했습니다. 다음 이미지를 맡아 주세요.':'공유 저장소에 임시 저장했습니다.');}
  catch(error){$('save-state').textContent='저장 실패 · 입력 유지';$('save-state').classList.add('dirty');toast(error.message);}
  finally{store.busy=false;updateControls();refresh();}
}
function resizeImage(){
  const img=$('source-image');if(!img.naturalWidth)return;const rotated=store.rotation%180!==0;
  const w=rotated?img.naturalHeight:img.naturalWidth,h=rotated?img.naturalWidth:img.naturalHeight;
  if(store.fit)store.zoom=Math.min(1,Math.max(.05,Math.min(($('viewer').clientWidth-52)/w,($('viewer').clientHeight-52)/h)));
  img.style.width=img.naturalWidth*store.zoom+'px';img.style.height=img.naturalHeight*store.zoom+'px';img.style.transform=`translate(-50%,-50%) rotate(${store.rotation}deg)`;
  $('image-stage').style.width=w*store.zoom+52+'px';$('image-stage').style.height=h*store.zoom+52+'px';$('zoom-value').textContent=Math.round(store.zoom*100)+'%';
}
$('source-image').onload=()=>{resizeImage();$('viewer').scrollTo(0,0);};$('source-image').onerror=()=>toast('이미지를 불러오지 못했습니다. 배포 파일을 확인해 주세요.');
function zoomImage(value){
  const viewer=$('viewer'),img=$('source-image');if(!img.naturalWidth)return;
  const old=store.zoom;
  const centerX=(viewer.scrollLeft+viewer.clientWidth/2)/Math.max(viewer.scrollWidth,1);
  const centerY=(viewer.scrollTop+viewer.clientHeight/2)/Math.max(viewer.scrollHeight,1);
  store.fit=false;store.zoom=Math.max(.05,Math.min(3,value));resizeImage();
  if(old!==store.zoom)viewer.scrollTo(centerX*viewer.scrollWidth-viewer.clientWidth/2,centerY*viewer.scrollHeight-viewer.clientHeight/2);
}
$('zoom-in').onclick=()=>zoomImage(store.zoom*1.3);
$('zoom-out').onclick=()=>zoomImage(store.zoom/1.3);
$('open-image').onclick=()=>zoomImage(1);
$('fit').onclick=()=>{store.fit=true;resizeImage();$('viewer').scrollTo(0,0);};
$('rotate').onclick=()=>{store.rotation=(store.rotation+90)%360;store.fit=true;resizeImage();};
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
$('insert-tab').onclick=insertTranscriptionTab;
$('transcription').addEventListener('keydown',e=>{
  if(e.key==='Tab' && !e.shiftKey && !e.ctrlKey && !e.altKey && !e.metaKey && !e.isComposing){
    e.preventDefault();insertTranscriptionTab();
  }
});
$('transcription').oninput=$('notes').oninput=markDirty;$('claim').onclick=()=>claim();$('next').onclick=()=>claim(true);$('save').onclick=()=>save('draft');$('verify').onclick=()=>save('verified');
$('release').onclick=async()=>{if(store.busy || !confirmLeave())return;store.busy=true;updateControls();try{await releaseCurrent();showItem(await api('item',null,{id:store.item.id}));toast('작업을 반납했습니다.');}catch(error){toast(error.message);}finally{store.busy=false;updateControls();refresh();}};
$('search').oninput=renderList;document.querySelectorAll('[data-filter]').forEach(button=>button.onclick=()=>{store.filter=button.dataset.filter;document.querySelectorAll('[data-filter]').forEach(b=>b.classList.toggle('active',b===button));renderList();});
$('refresh').onclick=refresh;$('help').onclick=()=>$('help-dialog').showModal();$('close-help').onclick=()=>$('help-dialog').close();
$('identity').onclick=()=>{if(store.busy)return;$('author-name').value=store.author;$('name-dialog').showModal();};$('name-cancel').onclick=()=>$('name-dialog').close();
$('name-form').onsubmit=async e=>{e.preventDefault();const name=$('author-name').value.trim();if(!name)return;
  if(store.author && name!==store.author){if(!confirmLeave())return;try{await releaseCurrent();}catch(error){toast(error.message);return;}}
  store.author=name;localStorage.setItem('hybridocr-author',name);$('identity').textContent=name+' 님';$('name-dialog').close();
  if(store.item){try{showItem(await api('item',null,{id:store.item.id}));}catch(error){toast(error.message);}}refresh();
};
$('export').onclick=()=>$('export-dialog').showModal();$('close-export').onclick=()=>$('export-dialog').close();
async function download(format){try{
  const scope=document.querySelector('[name=scope]:checked').value;const response=await fetch(`/api/labeling?action=export&format=${format}&scope=${scope}`,{cache:'no-store'});
  if(!response.ok){const error=await response.json();throw new Error(error.error);}
  const blob=await response.blob();const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=format==='zip'?`hybridocr-labels-${scope}.zip`:'hybridocr-references.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);$('export-dialog').close();
}catch(error){toast(error.message);}}
$('download-zip').onclick=()=>download('zip');$('download-json').onclick=()=>download('json');
window.addEventListener('beforeunload',e=>{if(store.dirty){e.preventDefault();e.returnValue='';}});
window.addEventListener('keydown',e=>{if((e.ctrlKey || e.metaKey) && e.key==='s'){e.preventDefault();save('draft');}});
setInterval(refresh,30000);
setInterval(async()=>{if(owns() && !store.busy){try{const item=await api('renew',{id:store.item.id});store.item.claim=item.claim;updateControls();}catch(error){toast(error.message);}}},3*60*1000);
if(store.author)$('identity').textContent=store.author+' 님';else{$('name-dialog').showModal();$('author-name').focus();}
refresh();
