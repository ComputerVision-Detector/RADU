import seed from '../data/seed.json' with { type: 'json' };
import assetManifest from '../data/assets.manifest.json' with { type: 'json' };
import { AppError,clone } from './state.mjs';

const statuses=new Set(['unlabeled','draft','verified']);
const assetsById=new Map(assetManifest.images.map(asset=>[asset.id,asset]));
const active=(item,now)=>item.claim && Date.parse(item.claim.until)>now;
const mine=(item,actor,now)=>active(item,now) && item.claim.client===actor.client && item.claim.name===actor.name;

function sourceImage(source) {
  const asset=assetsById.get(source.id);
  if(!asset || asset.file!==source.image.file.split('/').at(-1) || !/^[a-f0-9]{64}$/.test(asset.sha256) || !/^[a-f0-9]{64}$/.test(asset.source_sha256)) {
    throw new AppError(503,`이미지 fingerprint를 확인할 수 없습니다: ${source.id}`);
  }
  return {...clone(source.image),asset_sha256:asset.sha256,source_sha256:asset.source_sha256};
}

function emptyItem(source) {
  return {
    id:source.id,
    image:sourceImage(source),
    bbox_annotation:{status:'unlabeled',revision:0,annotator:'',notes:'',no_text:false,proposal_version:'',boxes:[]},
    claim:null,
    history:[]
  };
}

export function createBBoxState(catalog=seed) {
  const items={};
  for(const source of Object.values(catalog.items))items[source.id]=emptyItem(source);
  return {schema:1,dataset_version:'v2',dataset:catalog.dataset,revision:catalog.revision,updated_at:catalog.updated_at,items};
}

export function reconcileBBoxState(stored,catalog=seed) {
  if(stored?.schema!==1 || stored.dataset_version!=='v2' || !stored.items || typeof stored.items!=='object' || Array.isArray(stored.items)) {
    throw new AppError(503,'V2 bbox 라벨 데이터 형식을 확인해야 합니다.');
  }
  const result=clone(stored);
  for(const id of Object.keys(result.items)) {
    if(!Object.hasOwn(catalog.items,id))throw new AppError(503,'저장된 이미지 목록이 배포 버전보다 최신입니다. 최신 앱으로 재배포해 주세요.');
  }
  for(const [id,source] of Object.entries(catalog.items)) {
    if(!Object.hasOwn(result.items,id))result.items[id]=emptyItem(source);
    else {
      const previous=result.items[id].image,expected=sourceImage(source);
      if(previous?.source_sha256!==expected.source_sha256 || previous?.source_width!==expected.source_width || previous?.source_height!==expected.source_height) {
        throw new AppError(503,`원본 이미지가 변경되었습니다. 기존 V2 bbox를 자동으로 이전할 수 없습니다: ${id}`);
      }
      result.items[id].image=expected;
    }
  }
  result.dataset_version='v2';
  result.dataset=catalog.dataset;
  result.revision=catalog.revision;
  return result;
}

function validBox(box,image,ids,orders) {
  if(!box || typeof box!=='object' || typeof box.id!=='string' || !/^[a-zA-Z0-9_-]{1,80}$/.test(box.id) || ids.has(box.id))return false;
  if(!Array.isArray(box.bbox) || box.bbox.length!==4 || !box.bbox.every(Number.isInteger))return false;
  const [x1,y1,x2,y2]=box.bbox;
  if(x1<0 || y1<0 || x2>image.source_width || y2>image.source_height || x1>=x2 || y1>=y2)return false;
  if(typeof box.text!=='string' || box.text.length>50000 || typeof box.prediction!=='string' || box.prediction.length>50000)return false;
  if(!Number.isInteger(box.order) || box.order<1 || orders.has(box.order) || typeof box.uncertain!=='boolean')return false;
  ids.add(box.id);orders.add(box.order);return true;
}

function annotationComplete(annotation) {
  if(annotation.no_text)return annotation.boxes.length===0;
  return annotation.boxes.length>0 && annotation.boxes.every(box=>box.text.trim() && !box.uncertain);
}

export function validateBBoxState(state,catalog=seed) {
  if(state?.schema!==1 || state.dataset_version!=='v2' || !state.items || typeof state.items!=='object' || Array.isArray(state.items) || Object.keys(state.items).length!==Object.keys(catalog.items).length) {
    throw new Error('V2 bbox 백업 형식을 확인하세요.');
  }
  for(const [id,source] of Object.entries(catalog.items)) {
    const item=state.items[id],annotation=item?.bbox_annotation,expectedImage=sourceImage(source);
    if(item?.image?.source_width!==expectedImage.source_width || item?.image?.source_height!==expectedImage.source_height)throw new Error(`V2 원본 이미지 크기 오류: ${id}`);
    if(!item || item.id!==id || item.image?.file!==expectedImage.file || item.image?.asset_sha256!==expectedImage.asset_sha256 || item.image?.source_sha256!==expectedImage.source_sha256 || !annotation || !statuses.has(annotation.status) || !Number.isInteger(annotation.revision) || annotation.revision<0 || typeof annotation.annotator!=='string' || typeof annotation.notes!=='string' || annotation.notes.length>8000 || typeof annotation.no_text!=='boolean' || typeof annotation.proposal_version!=='string' || annotation.proposal_version.length>200 || !Array.isArray(annotation.boxes) || annotation.boxes.length>1000 || !Array.isArray(item.history)) {
      throw new Error(`V2 bbox 라벨 형식 오류: ${id}`);
    }
    const ids=new Set(),orders=new Set();
    for(const box of annotation.boxes)if(!validBox(box,item.image,ids,orders))throw new Error(`V2 bbox 박스 형식 오류: ${id}`);
    if(annotation.no_text && annotation.boxes.length)throw new Error(`텍스트 없음 표시와 bbox가 함께 저장됨: ${id}`);
    if(annotation.status==='verified' && !annotationComplete(annotation))throw new Error(`미완료 bbox 검수 표시: ${id}`);
    if(annotation.boxes.some(box=>box.uncertain) && !annotation.notes.trim())throw new Error(`불확실한 bbox의 메모가 없습니다: ${id}`);
    if(item.claim!==null && (!item.claim || typeof item.claim.name!=='string' || typeof item.claim.client!=='string' || !Number.isFinite(Date.parse(item.claim.until))))throw new Error(`V2 작업 임대 형식 오류: ${id}`);
  }
  return state;
}

function actorFrom(body) {
  if(typeof body.author!=='string' || !body.author.trim() || body.author.trim().length>40)throw new AppError(400,'작성자 이름을 1~40자로 입력하세요.');
  if(typeof body.client_id!=='string' || !/^[a-zA-Z0-9-]{16,80}$/.test(body.client_id))throw new AppError(400,'브라우저 작업 식별자를 확인하세요. 새로고침 후 다시 시도하세요.');
  return {name:body.author.trim(),client:body.client_id};
}

function present(item,client,now) {
  const result=clone(item);
  result.claim=active(item,now)?{name:item.claim.name,until:item.claim.until,mine:item.claim.client===client}:null;
  return result;
}

export function listBBoxState(state,client='',now=Date.now()) {
  const items=Object.values(state.items).map(item=>({
    id:item.id,status:item.bbox_annotation.status,annotator:item.bbox_annotation.annotator,
    revision:item.bbox_annotation.revision,boxes:item.bbox_annotation.boxes.length,
    no_text:item.bbox_annotation.no_text,
    claim:active(item,now)?{name:item.claim.name,until:item.claim.until,mine:item.claim.client===client}:null
  }));
  const verified=items.filter(item=>item.status==='verified').length;
  return {items,stats:{total:items.length,verified,remaining:items.length-verified,draft:items.filter(item=>item.status==='draft').length,active:items.filter(item=>item.claim).length},updated_at:state.updated_at};
}

export function getBBoxItem(state,id,client='',now=Date.now()) {
  const item=Object.hasOwn(state.items,id)?state.items[id]:null;
  if(!item)throw new AppError(404,'이미지를 찾지 못했습니다.');
  return present(item,client,now);
}

export function mutateBBox(state,action,body,now=Date.now()) {
  const actor=actorFrom(body);let item;
  if(action==='next') {
    item=Object.values(state.items).find(candidate=>candidate.id!==body.exclude_id && candidate.bbox_annotation.status!=='verified' && (!active(candidate,now) || mine(candidate,actor,now)));
    if(!item)throw new AppError(409,'지금 맡을 수 있는 이미지가 없습니다. 다른 작업이 끝난 후 다시 확인하세요.');
  } else {
    item=Object.hasOwn(state.items,body.id)?state.items[body.id]:null;
    if(!item)throw new AppError(404,'이미지를 찾지 못했습니다.');
  }
  const annotation=item.bbox_annotation;
  if(action==='claim' || action==='next') {
    if(active(item,now) && !mine(item,actor,now))throw new AppError(409,`${item.claim.name} 님이 작업 중입니다.`);
    item.claim={...actor,until:new Date(now+25*60*1000).toISOString()};
  } else {
    if(!mine(item,actor,now))throw new AppError(409,'작업 권한이 만료되었거나 다른 작성자가 작업 중입니다. 다시 작업을 맡아 주세요.');
    if(action==='release')item.claim=null;
    else if(action==='renew')item.claim.until=new Date(now+25*60*1000).toISOString();
    else if(action==='save') {
      if(!Number.isInteger(body.revision) || body.revision!==annotation.revision)throw new AppError(409,'다른 창에서 수정한 내용이 있습니다. 현재 입력을 복사한 뒤 최신 결과를 다시 여세요.');
      if(!['draft','verified'].includes(body.status) || !Array.isArray(body.boxes) || body.boxes.length>1000 || typeof body.notes!=='string' || body.notes.length>8000 || typeof body.no_text!=='boolean' || typeof body.proposal_version!=='string' || body.proposal_version.length>200)throw new AppError(400,'bbox 저장 내용을 확인하세요.');
      const next={...annotation,status:body.status,annotator:actor.name,notes:body.notes,no_text:body.no_text,proposal_version:body.proposal_version,boxes:clone(body.boxes)};
      const ids=new Set(),orders=new Set();
      if(next.boxes.some(box=>!validBox(box,item.image,ids,orders)))throw new AppError(400,'bbox 좌표·문구·순서 값을 확인하세요.');
      if(next.no_text && next.boxes.length)throw new AppError(400,'텍스트 없음으로 표시한 이미지에는 bbox를 둘 수 없습니다.');
      if(next.boxes.some(box=>box.uncertain) && !next.notes.trim())throw new AppError(400,'불확실한 줄에 대한 메모를 입력하세요.');
      if(next.status==='verified' && !annotationComplete(next))throw new AppError(400,'모든 bbox의 정답을 입력하거나, bbox가 없는 이미지는 텍스트 없음으로 표시하세요. 불확실한 줄은 검수 완료할 수 없습니다.');
      next.revision+=1;item.bbox_annotation=next;
      const at=new Date(now).toISOString();
      item.history=[{at,author:actor.name,status:next.status,revision:next.revision,box_count:next.boxes.length},...(item.history || [])].slice(0,20);
      if(next.status==='verified')item.claim=null;
      else item.claim.until=new Date(now+25*60*1000).toISOString();
    } else throw new AppError(400,'지원하지 않는 작업입니다.');
  }
  state.updated_at=new Date(now).toISOString();return present(item,actor.client,now);
}

export async function applyBBoxMutation(store,action,body) {
  for(let attempt=0;attempt<5;attempt++) {
    const {state,version}=await store.read();const result=mutateBBox(state,action,body);
    if(await store.write(state,version,body.author))return result;
  }
  throw new AppError(409,'여러 작업이 동시에 저장되고 있습니다. 잠시 뒤 다시 저장해 주세요. 입력 내용은 유지됩니다.');
}
