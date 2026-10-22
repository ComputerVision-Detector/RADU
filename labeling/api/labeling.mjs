import { GitHubStore } from '../lib/github-store.mjs';
import { AppError,applyMutation,getItem,labelOnly,listState } from '../lib/state.mjs';
import { zipFiles } from '../lib/zip.mjs';

export function handlerFor(store) {
  return async function handler(req,res) {
    res.setHeader('Cache-Control','no-store');res.setHeader('X-Content-Type-Options','nosniff');
    try {
      const url=new URL(req.url,'http://local');const action=url.searchParams.get('action') || 'list';
      if(req.method==='GET') {
        const {state}=await store.read();const client=url.searchParams.get('client_id') || '';
        if(action==='health')return res.status(200).json({status:'ok',storage:store.kind || 'shared',samples:Object.keys(state.items).length});
        if(action==='list')return res.status(200).json(listState(state,client));
        if(action==='item')return res.status(200).json(getItem(state,url.searchParams.get('id'),client));
        if(action==='export') {
          const verified=url.searchParams.get('scope')==='verified';
          const samples=Object.values(state.items).filter(i=>!verified || i.status==='verified').map(labelOnly);
          if(url.searchParams.get('format')==='zip') {
            const zip=zipFiles(samples.map(i=>[`labels/${i.id}.json`,JSON.stringify(i,null,2)+'\n']));
            res.setHeader('Content-Type','application/zip');res.setHeader('Content-Disposition',`attachment; filename="hybridocr-labels-${verified?'verified':'all'}.zip"`);
            return res.status(200).send(zip);
          }
          res.setHeader('Content-Disposition','attachment; filename="hybridocr-references.json"');
          return res.status(200).json({version:1,source:'HybridOCR shared labeling',dataset:state.dataset,dataset_revision:state.revision,samples});
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
      const result=await applyMutation(store,action,body);return res.status(200).json(result);
    } catch(error) {
      const status=error instanceof AppError ? error.status : error instanceof SyntaxError ? 400 : 500;
      return res.status(status).json({error:status===500?'저장 처리 중 오류가 발생했습니다. 입력 내용은 유지됩니다.':error.message,...(error.code ? {code:error.code} : {})});
    }
  };
}
export default async function handler(req,res) {
  try { return await handlerFor(new GitHubStore())(req,res); }
  catch { return res.status(503).json({error:'공유 저장소 배포 설정을 확인해 주세요.'}); }
}
