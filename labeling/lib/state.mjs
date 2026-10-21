import seed from '../data/seed.json' with { type: 'json' };
export class AppError extends Error { constructor(status, message, code) { super(message); this.status = status; this.code = code; } }
export const clone = value => structuredClone(value);
export function initialState() { return clone(seed); }
export function labelOnly(item) {
  const result = Object.fromEntries(['id','text','status','annotator','notes'].map(key => [key, item[key]]));
  if (item.submitted_id) result.submitted_id = item.submitted_id;
  return result;
}
function identity(body) {
  if (typeof body.author !== 'string' || !body.author.trim() || body.author.trim().length > 40) throw new AppError(400,'작성자 이름을 1~40자로 입력하세요.');
  if (typeof body.client_id !== 'string' || !/^[a-zA-Z0-9-]{16,80}$/.test(body.client_id)) throw new AppError(400,'브라우저 작업 식별자를 확인하세요. 새로고침 후 다시 시도하세요.');
  return { name: body.author.trim(), client: body.client_id };
}
const active = (item, now) => item.claim && Date.parse(item.claim.until) > now;
const mine = (item, actor, now) => active(item, now) && item.claim.client === actor.client && item.claim.name === actor.name;
function present(item, client, now) {
  const result = clone(item);
  result.claim = active(item, now) ? { name: item.claim.name, until: item.claim.until, mine: item.claim.client === client } : null;
  return result;
}
export function listState(state, client='', now=Date.now()) {
  const items = Object.values(state.items).map(item => ({ id:item.id, status:item.status, annotator:item.annotator,
    revision:item.revision, seed_verified:item.seed_verified, characters:(item.text || '').length,
    claim:active(item,now) ? {name:item.claim.name,until:item.claim.until,mine:item.claim.client===client} : null }));
  const verified = items.filter(i=>i.status==='verified').length;
  return { items, stats:{total:items.length,verified,remaining:items.length-verified,draft:items.filter(i=>i.status==='draft').length,
    active:items.filter(i=>i.claim).length,initial_verified:items.filter(i=>i.seed_verified).length}, updated_at:state.updated_at };
}
export function getItem(state, id, client='', now=Date.now()) {
  const item=Object.hasOwn(state.items,id) ? state.items[id] : null; if(!item) throw new AppError(404,'이미지를 찾지 못했습니다.');
  return present(item, client, now);
}
export function mutate(state, action, body, now=Date.now()) {
  const actor=identity(body); let item;
  if(action==='next') {
    item=Object.values(state.items).find(i=>i.id!==body.exclude_id && !i.seed_verified && i.status!=='verified' && (!active(i,now) || mine(i,actor,now)));
    if(!item) throw new AppError(409,'지금 맡을 수 있는 이미지가 없습니다. 다른 작업이 끝난 후 다시 확인하세요.');
  } else { item=Object.hasOwn(state.items,body.id) ? state.items[body.id] : null; if(!item) throw new AppError(404,'이미지를 찾지 못했습니다.'); }
  if(item.seed_verified) throw new AppError(409,'기존에 검수한 5장 정답은 참고용으로 보호되어 있습니다.');
  if(action==='claim' || action==='next') {
    if(active(item,now) && !mine(item,actor,now)) throw new AppError(409,`${item.claim.name} 님이 작업 중입니다.`);
    item.claim={...actor,until:new Date(now+25*60*1000).toISOString()};
  } else {
    if(!mine(item,actor,now)) throw new AppError(409,'작업 권한이 만료되었거나 다른 작성자가 작업 중입니다. 다시 작업을 맡아 주세요.');
    if(action==='release') item.claim=null;
    else if(action==='renew') item.claim.until=new Date(now+25*60*1000).toISOString();
    else if(action==='save') {
      if(!Number.isInteger(body.revision) || body.revision!==item.revision) throw new AppError(409,'다른 창에서 수정한 내용이 있습니다. 현재 입력을 복사한 뒤 최신 결과를 다시 여세요.');
      if(typeof body.text!=='string' || body.text.length>50000 || typeof body.notes!=='string' || body.notes.length>8000) throw new AppError(400,'전사·메모 길이를 확인하세요.');
      if(!['draft','verified'].includes(body.status)) throw new AppError(400,'저장 상태가 올바르지 않습니다.');
      if(body.status==='verified' && !body.text.trim()) throw new AppError(400,'전사 내용을 작성한 뒤 검수 완료하세요.');
      item.text=body.text;item.notes=body.notes;item.status=body.status;item.annotator=actor.name;item.revision+=1;
      item.updated_at=new Date(now).toISOString();
      item.history=[{at:item.updated_at,author:actor.name,status:item.status,revision:item.revision},...(item.history || [])].slice(0,20);
      if(body.status==='verified') item.claim=null;
      else item.claim.until=new Date(now+25*60*1000).toISOString();
    } else throw new AppError(400,'지원하지 않는 작업입니다.');
  }
  state.updated_at=new Date(now).toISOString(); return present(item,actor.client,now);
}
export async function applyMutation(store,action,body) {
  for(let attempt=0;attempt<5;attempt++) {
    const {state,version}=await store.read(); const result=mutate(state,action,body);
    if(await store.write(state,version,body.author)) return result;
  }
  throw new AppError(409,'여러 작업이 동시에 저장되고 있습니다. 잠시 뒤 다시 저장해 주세요. 입력 내용은 유지됩니다.');
}
