import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync,readdirSync,readFileSync,writeFileSync,rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { FileStore } from '../lib/file-store.mjs';
import { initialState,mutate } from '../lib/state.mjs';
const actor={id:'hf-0000',author:'파일 저장 검증',client_id:'filetest-0000-0000-0000-000000000000'};
test('file store persists across restarts, detects stale writes and keeps bounded backups',async()=>{
 const root=mkdtempSync(path.join(tmpdir(),'label-file-'));
 try {
  const file=path.join(root,'state.json'),store=new FileStore(file,{limit:2});
  let current=await store.read();mutate(current.state,'claim',actor);assert.equal(await store.write(current.state,current.version),true);
  current=await store.read();mutate(current.state,'save',{...actor,text:'재시작 후에도 남는 정답',notes:'확인',status:'draft',revision:0});
  assert.equal(await store.write(current.state,current.version),true);
  assert.equal(await store.write(current.state,current.version),false);
  const restarted=new FileStore(file,{limit:2});assert.equal((await restarted.read()).state.items['hf-0000'].text,'재시작 후에도 남는 정답');
  for(let i=0;i<3;i++){current=await store.read();mutate(current.state,'renew',actor);await store.write(current.state,current.version);}
  assert.equal(readdirSync(path.join(root,'backups')).length,2);
  for(const name of readdirSync(path.join(root,'backups')))assert.equal(JSON.parse(readFileSync(path.join(root,'backups',name))).schema,1);
 } finally {rmSync(root,{recursive:true,force:true});}
});
test('two writers sharing a volume do not overwrite the same base version',async()=>{
 const root=mkdtempSync(path.join(tmpdir(),'label-race-'));
 try {
  const a=new FileStore(path.join(root,'state.json')),b=new FileStore(path.join(root,'state.json'));
  const first=await a.read(),second=await b.read();
  mutate(first.state,'claim',actor);mutate(second.state,'claim',{...actor,id:'hf-0001'});
  const results=await Promise.all([a.write(first.state,first.version),b.write(second.state,second.version)]);
  assert.equal(results.filter(Boolean).length,1);
 } finally {rmSync(root,{recursive:true,force:true});}
});
test('corrupt state never becomes a fresh seed and protected references cannot change',async()=>{
 const root=mkdtempSync(path.join(tmpdir(),'label-corrupt-'));
 try {
  const file=path.join(root,'state.json');writeFileSync(file,'corrupt');const store=new FileStore(file);
  await assert.rejects(()=>store.read(),/저장 파일/);
  assert.equal(readFileSync(file,'utf8'),'corrupt');
  rmSync(file);const current=await store.read();current.state.items['hf-0003'].text='wrong';
  await assert.rejects(()=>store.write(current.state,null),/기존 정답 변경/);
 } finally {rmSync(root,{recursive:true,force:true});}
});
test('explicit validated restore recovers corrupted state and preserves the old bytes',async()=>{
 const root=mkdtempSync(path.join(tmpdir(),'label-restore-'));
 try {
  const file=path.join(root,'state.json'),store=new FileStore(file);const backup=(await store.read()).state;
  writeFileSync(file,'damaged bytes');await store.replace(backup);
  assert.equal(Object.keys((await store.read()).state.items).length,Object.keys(initialState().items).length);
  const name=readdirSync(path.join(root,'backups')).find(n=>n.includes('-before-restore-'));
  assert.equal(readFileSync(path.join(root,'backups',name),'utf8'),'damaged bytes');
 } finally {rmSync(root,{recursive:true,force:true});}
});
test('older disk catalog gains new images without resetting annotations',async()=>{
 const root=mkdtempSync(path.join(tmpdir(),'label-migrate-'));
 try {
  const file=path.join(root,'state.json');const old=initialState();const newest=Object.keys(old.items).at(-1);delete old.items[newest];
  old.items['hf-0000'].text='기존 전사';old.items['hf-0000'].status='draft';old.items['hf-0000'].revision=4;
  writeFileSync(file,JSON.stringify(old));const store=new FileStore(file);
  const {state,version}=await store.read();assert.ok(state.items[newest]);assert.equal(state.items['hf-0000'].revision,4);
  assert.equal(await store.write(state,version),true);assert.equal((await store.read()).state.items['hf-0000'].text,'기존 전사');
 }finally{rmSync(root,{recursive:true,force:true});}
});
