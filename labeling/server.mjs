// Standalone production server. Shared state lives on the host's persistent disk.
import http from 'node:http';
import { existsSync,statSync,createReadStream } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { handlerFor } from './api/labeling.mjs';
import { FileStore } from './lib/file-store.mjs';
import { createBBoxState,reconcileBBoxState,validateBBoxState } from './lib/bbox-state.mjs';
const root=fileURLToPath(new URL('.',import.meta.url));
const store=new FileStore(process.env.LABEL_LOCAL_STORE || path.join(root,'.runtime/state.json'),{
  backups:process.env.LABEL_BACKUP_DIR,limit:Number(process.env.LABEL_BACKUP_LIMIT ?? 50)
});
const bboxStore=new FileStore(process.env.LABEL_LOCAL_STORE_V2 || path.join(root,'.runtime/state-v2.json'),{
  backups:process.env.LABEL_BACKUP_DIR_V2 || path.join(root,'.runtime/backups-v2'),limit:Number(process.env.LABEL_BACKUP_LIMIT ?? 50),
  initialState:createBBoxState,reconcileState:reconcileBBoxState,validateState:validateBBoxState
});
// Fail at startup on corrupt state; never replace existing work with seed data.
await Promise.all([store.read(),bboxStore.read()]);
const api=handlerFor(store,bboxStore);
const types={'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.css':'text/css; charset=utf-8','.webp':'image/webp','.json':'application/json; charset=utf-8'};
const server=http.createServer(async(req,res)=>{
  res.status=code=>{res.statusCode=code;return res;};res.json=data=>{res.setHeader('Content-Type','application/json; charset=utf-8');res.end(JSON.stringify(data));};res.send=data=>res.end(data);
  res.setHeader('X-Content-Type-Options','nosniff');res.setHeader('Referrer-Policy','same-origin');
  try {
    const url=new URL(req.url,'http://localhost');
    if(url.pathname==='/healthz' && req.method==='GET'){await store.read();return res.status(200).json({status:'ok',storage:'persistent-file'});}
    if(url.pathname==='/api/labeling') {
      let size=0,parts=[];for await(const chunk of req){size+=chunk.length;if(size>200000){res.status(413).json({error:'입력이 너무 큽니다.'});return;}parts.push(chunk);}
      req.body=Buffer.concat(parts).toString('utf8');return await api(req,res);
    }
    const filename=url.pathname==='/'?'index.html':url.pathname.slice(1);
    if(!['GET','HEAD'].includes(req.method) || (!['index.html','app.js','app.css','data/text-box-proposals.json'].includes(filename) && !/^assets\/hf-\d{4}\.webp$/.test(filename))){res.statusCode=404;res.end('Not found');return;}
    const target=filename==='data/text-box-proposals.json'
      ?path.resolve(root,process.env.LABEL_PROPOSALS_FILE || filename)
      :path.join(root,filename);
    if(!existsSync(target) || !statSync(target).isFile()){res.statusCode=404;res.end('Not found');return;}
    res.setHeader('Content-Type',types[path.extname(target)] || 'application/octet-stream');
    res.setHeader('Cache-Control',filename.endsWith('.webp')?'public, max-age=86400':'no-cache');
    res.setHeader('Content-Length',statSync(target).size);
    if(req.method==='HEAD')return res.end();
    const stream=createReadStream(target);stream.on('error',()=>res.destroy());stream.pipe(res);
  } catch {if(!res.headersSent)res.status(503).json({error:'서버 저장 상태를 확인해야 합니다.'});else res.destroy();}
});
server.requestTimeout=30000;server.headersTimeout=15000;
const port=Number(process.env.PORT || 8014),host=process.env.HOST || '127.0.0.1';
server.listen(port,host,()=>console.log(`HybridOCR labeling: http://${host}:${port} (persistent file storage)`));
for(const name of ['SIGTERM','SIGINT'])process.on(name,()=>server.close(()=>process.exit(0)));
