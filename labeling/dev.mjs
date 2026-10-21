// Local shared-file server for QA / self-hosting. Vercel uses GitHubStore instead.
import http from 'node:http';
import { readFileSync,writeFileSync,mkdirSync,existsSync,renameSync,statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { handlerFor } from './api/labeling.mjs';
import { initialState } from './lib/state.mjs';
const root=fileURLToPath(new URL('.',import.meta.url));
const runtime=process.env.LABEL_LOCAL_STORE || path.join(root,'.runtime/state.json');
mkdirSync(path.dirname(runtime),{recursive:true});
function snapshot() {
  if(!existsSync(runtime))return {state:initialState(),version:null};
  const raw=readFileSync(runtime);return {state:JSON.parse(raw),version:createHash('sha256').update(raw).digest('hex')};
}
const store={read:async()=>snapshot(),write:async(state,version)=>{
  // No await between compare and atomic rename: a single local process serializes writes.
  if(snapshot().version!==version)return false;
  writeFileSync(runtime+'.tmp',JSON.stringify(state,null,2)+'\n');renameSync(runtime+'.tmp',runtime);return true;
}};
const api=handlerFor(store);
const types={'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.css':'text/css; charset=utf-8','.webp':'image/webp'};
const server=http.createServer(async(req,res)=>{
  res.status=code=>{res.statusCode=code;return res;};res.json=data=>{res.setHeader('Content-Type','application/json; charset=utf-8');res.end(JSON.stringify(data));};res.send=data=>res.end(data);
  const url=new URL(req.url,'http://localhost');
  if(url.pathname==='/api/labeling') {
    let size=0,parts=[];for await(const chunk of req){size+=chunk.length;if(size>200000){res.status(413).json({error:'입력이 너무 큽니다.'});return;}parts.push(chunk);}
    req.body=Buffer.concat(parts).toString('utf8');return api(req,res);
  }
  let filename=url.pathname==='/'?'index.html':url.pathname.slice(1);
  // Serve only public files, never live labels, environment files, or server source.
  if(!['index.html','app.js','app.css'].includes(filename) && !/^assets\/hf-\d{4}\.webp$/.test(filename)){res.statusCode=404;res.end('Not found');return;}
  const target=path.join(root,filename);
  if(!existsSync(target) || !statSync(target).isFile()){res.statusCode=404;res.end('Not found');return;}
  res.setHeader('Content-Type',types[path.extname(target)] || 'application/octet-stream');res.end(readFileSync(target));
});
const port=Number(process.env.PORT || 8014);server.listen(port,'127.0.0.1',()=>console.log(`HybridOCR labeling: http://127.0.0.1:${port}`));
