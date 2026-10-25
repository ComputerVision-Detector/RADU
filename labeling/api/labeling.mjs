import { GitHubStore } from '../lib/github-store.mjs';
import { AppError,applyMutation,getItem,labelOnly,listState } from '../lib/state.mjs';
import { applyBBoxMutation,createBBoxState,getBBoxItem,listBBoxState,reconcileBBoxState,validateBBoxState } from '../lib/bbox-state.mjs';
import { zipFiles } from '../lib/zip.mjs';

function bboxLabelOnly(item) {
  const annotation=item.bbox_annotation;
  return {
    schema_version:1,dataset_version:'v2',image_id:item.id,image_file:item.image.file,
    image_size:{width:item.image.source_width,height:item.image.source_height},
    status:annotation.status,revision:annotation.revision,annotator:annotation.annotator,
    notes:annotation.notes,no_text:annotation.no_text,proposal_version:annotation.proposal_version,
    boxes:annotation.boxes
  };
}

function manifest(version,state,scope,count) {
  return {schema_version:1,dataset_version:version,scope,dataset:state.dataset,dataset_revision:state.revision,updated_at:state.updated_at,samples:count};
}

export function handlerFor(v1Store,v2Store) {
  return async function handler(req,res) {
    res.setHeader('Cache-Control','no-store');res.setHeader('X-Content-Type-Options','nosniff');
    try {
      const url=new URL(req.url,'http://local');const action=url.searchParams.get('action') || 'list';
      const version=url.searchParams.get('version') ?? 'v1';
      if(!['v1','v2'].includes(version))throw new AppError(400,'데이터 버전은 v1 또는 v2여야 합니다.');
      if(version==='v2' && (!v2Store || v2Store===v1Store))throw new AppError(503,'V2 bbox 저장소가 설정되지 않았습니다. V1 데이터는 변경하지 않았습니다.');
      const store=version==='v2'?v2Store:v1Store;
      if(req.method==='GET') {
        const {state}=await store.read();const client=url.searchParams.get('client_id') || '';
        if(action==='health')return res.status(200).json({status:'ok',dataset_version:version,storage:store.kind || 'shared',samples:Object.keys(state.items).length});
        if(action==='list')return res.status(200).json(version==='v2'?listBBoxState(state,client):listState(state,client));
        if(action==='item')return res.status(200).json(version==='v2'?getBBoxItem(state,url.searchParams.get('id'),client):getItem(state,url.searchParams.get('id'),client));
        if(action==='export') {
          const verified=url.searchParams.get('scope')==='verified';
          const samples=version==='v2'
            ?Object.values(state.items).filter(i=>!verified || i.bbox_annotation.status==='verified').map(bboxLabelOnly)
            :Object.values(state.items).filter(i=>!verified || i.status==='verified').map(labelOnly);
          if(url.searchParams.get('format')==='zip') {
            const prefix=version==='v2'?'bbox-labels':'labels';
            const files=[...samples.map(i=>[`${prefix}/${version==='v2'?i.image_id:i.id}.json`,JSON.stringify(i,null,2)+'\n']),[`${prefix}/manifest.json`,JSON.stringify(manifest(version,state,verified?'verified':'all',samples.length),null,2)+'\n']];
            const zip=zipFiles(files);
            res.setHeader('Content-Type','application/zip');res.setHeader('Content-Disposition',`attachment; filename="hybridocr-${version==='v2'?'v2-bbox-':'v1-'}labels-${verified?'verified':'all'}.zip"`);
            return res.status(200).send(zip);
          }
          const payload=version==='v2'
            ?{schema_version:1,dataset_version:'v2',source:'HybridOCR line bbox labeling',dataset:state.dataset,dataset_revision:state.revision,updated_at:state.updated_at,samples}
            :{version:1,dataset_version:'v1',source:'HybridOCR shared labeling',dataset:state.dataset,dataset_revision:state.revision,samples};
          res.setHeader('Content-Disposition',`attachment; filename="hybridocr-${version}-labels.json"`);
          return res.status(200).json(payload);
        }
        throw new AppError(404,'지원하지 않는 요청입니다.');
      }
      if(req.method!=='POST') {res.setHeader('Allow','GET, POST');throw new AppError(405,'지원하지 않는 요청 방식입니다.');}
      const origin=req.headers.origin;
      const host=req.headers['x-forwarded-host'] || req.headers.host;
      if(origin && new URL(origin).host!==host)throw new AppError(403,'다른 사이트에서의 저장 요청은 허용하지 않습니다.');
      if(!String(req.headers['content-type'] || '').startsWith('application/json'))throw new AppError(415,'JSON 형식으로 요청하세요.');
      if(!['claim','next','renew','release','save'].includes(action))throw new AppError(404,'지원하지 않는 작업입니다.');
      const body=typeof req.body==='string' ? JSON.parse(req.body) : req.body;
      if(!body || Array.isArray(body) || typeof body!=='object')throw new AppError(400,'입력 형식을 확인하세요.');
      const result=version==='v2'?await applyBBoxMutation(store,action,body):await applyMutation(store,action,body);return res.status(200).json(result);
    } catch(error) {
      const status=error instanceof AppError ? error.status : error instanceof SyntaxError ? 400 : 500;
      return res.status(status).json({error:status===500?'저장 처리 중 오류가 발생했습니다. 입력 내용은 유지됩니다.':error.message,...(error.code ? {code:error.code} : {})});
    }
  };
}
export default async function handler(req,res) {
  try {
    const v1Store=new GitHubStore();
    const v2Store=new GitHubStore(process.env,fetch,{
      statePath:'labeling/data/live-v2.json',initialState:createBBoxState,
      reconcileState:reconcileBBoxState,validateState:validateBBoxState
    });
    return await handlerFor(v1Store,v2Store)(req,res);
  }
  catch { return res.status(503).json({error:'공유 저장소 배포 설정을 확인해 주세요.'}); }
}
