import test from 'node:test';
import assert from 'node:assert/strict';
import { initialState,reconcileState,mutate,applyMutation,listState,getItem,labelOnly } from '../lib/state.mjs';
import { zipFiles } from '../lib/zip.mjs';
import { GitHubStore } from '../lib/github-store.mjs';
const alice={author:'이나현',client_id:'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa',id:'hf-0000'};
const bob={author:'김혜린',client_id:'bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb',id:'hf-0000'};
const now=Date.now();

test('catalog images load with five protected references',()=>{
 const state=initialState();const result=listState(state);assert.equal(result.stats.total,Object.keys(initialState().items).length);assert.equal(result.stats.verified,5);assert.equal(result.stats.remaining,Object.keys(initialState().items).length-5);
 assert.throws(()=>mutate(state,'claim',{...alice,id:'hf-0003'},now),/보호/);
 assert.throws(()=>getItem(state,'__proto__'),/찾지/);
});
test('one client owns a claim; stale revision is rejected; draft and verification persist',()=>{
 const state=initialState();const first=mutate(state,'claim',alice,now);assert.equal(first.claim.mine,true);
 assert.throws(()=>mutate(state,'claim',bob,now),/작업 중/);
 assert.equal(getItem(state,'hf-0000',bob.client_id,now).claim.mine,false);
 assert.equal(getItem(state,'hf-0000',bob.client_id,now).claim.client,undefined);
 mutate(state,'save',{...alice,text:'한국어 English\n표\t셀',notes:'원문 유지',status:'draft',revision:0},now+10);
 assert.throws(()=>mutate(state,'save',{...alice,text:'overwrite',notes:'',status:'draft',revision:0},now+20),/다른 창/);
 const result=mutate(state,'save',{...alice,text:'한국어 English\n표\t셀',notes:'원문 유지',status:'verified',revision:1},now+30);
 assert.equal(result.revision,2);assert.equal(result.claim,null);assert.equal(result.history.length,2);assert.equal(listState(state).stats.verified,6);
 assert.equal(labelOnly(result).text,'한국어 English\n표\t셀');
});
test('expired claim can be reassigned and original owner cannot save',()=>{
 const state=initialState();mutate(state,'claim',alice,now);mutate(state,'claim',bob,now+26*60*1000);
 assert.throws(()=>mutate(state,'save',{...alice,text:'a',notes:'',status:'draft',revision:0},now+26*60*1000),/권한/);
});
test('empty verified labels and invalid text types are rejected',()=>{
 const state=initialState();mutate(state,'claim',alice,now);
 assert.throws(()=>mutate(state,'save',{...alice,text:'   ',notes:'',status:'verified',revision:0},now),/내용/);
 assert.throws(()=>mutate(state,'save',{...alice,text:{},notes:'',status:'draft',revision:0},now),/길이/);
});
test('next excludes current image and releases leave labels intact',()=>{
 const state=initialState();const next=mutate(state,'next',{...alice,exclude_id:'hf-0000'},now);assert.equal(next.id,'hf-0001');
 mutate(state,'release',{...alice,id:next.id},now+1);assert.equal(state.items[next.id].claim,null);
});
test('concurrent writers retry without losing labels; conflict remains explicit',async()=>{
 let state=initialState(),version=0;let collide=true;
 const store={read:async()=>({state:structuredClone(state),version}),write:async(next,old)=>{if(collide){collide=false;state.items['hf-0001'].notes='other writer';version++;return false;}if(old!==version)return false;state=structuredClone(next);version++;return true;}};
 const result=await applyMutation(store,'claim',alice);assert.equal(result.id,'hf-0000');assert.equal(state.items['hf-0001'].notes,'other writer');
});
test('ZIP contains correctly named UTF-8 JSON data',()=>{
 const zip=zipFiles([['labels/hf-0000.json',JSON.stringify({text:'한글\nEnglish'})]]);
 assert.equal(zip.readUInt32LE(0),0x04034b50);assert.ok(zip.includes(Buffer.from('labels/hf-0000.json')));assert.ok(zip.includes(Buffer.from('한글')));
 assert.equal(zip.readUInt32LE(zip.length-22),0x06054b50);
});
test('GitHub writes use SHA concurrency and data-only branch',async()=>{
 const calls=[];
 const fetcher=async(url,options)=>{calls.push({url,options});if(url.includes('/git/ref/'))return new Response(JSON.stringify({object:{sha:'base'}}),{status:200});return new Response('{}',{status:409});};
 const github=new GitHubStore({LABEL_GITHUB_TOKEN:'test-only-placeholder'},fetcher);
 assert.equal(await github.write(initialState(),'old-sha','test'),false);
 const body=JSON.parse(calls.at(-1).options.body);assert.equal(body.sha,'old-sha');assert.equal(body.branch,'RADU/labeling-data');
 assert.throws(()=>new GitHubStore({LABEL_DATA_BRANCH:'main'}),/브랜치/);
});
test('missing deployment token never falls back to an ephemeral production file',async()=>{
 const github=new GitHubStore({});await assert.rejects(()=>github.read(),/LABEL_GITHUB_TOKEN/);
});
test('GitHub adapter creates a data branch and survives a fresh-instance read',async()=>{
 let branch=false,file=null,version=0;
 const fetcher=async(url,options)=>{
  if(url.includes('/git/ref/heads/RADU/labeling-data'))return new Response(JSON.stringify({object:{sha:'base'}}),{status:branch?200:404});
  if(url.includes('/git/ref/heads/RADU/labeling'))return new Response(JSON.stringify({object:{sha:'base'}}),{status:200});
  if(url.endsWith('/git/refs')){branch=true;return new Response('{}',{status:201});}
  if(options.method==='PUT'){
   const body=JSON.parse(options.body);if(body.sha!==(file?`sha-${version}`:undefined))return new Response('{}',{status:409});
   file=body.content;version++;return new Response('{}',{status:201});
  }
  return file?new Response(JSON.stringify({encoding:'base64',sha:`sha-${version}`,content:file}),{status:200}):new Response('{}',{status:404});
 };
 const env={LABEL_GITHUB_TOKEN:'test-placeholder'};
 await applyMutation(new GitHubStore(env,fetcher),'claim',alice);
 await applyMutation(new GitHubStore(env,fetcher),'save',{...alice,text:'서버 영구 저장',notes:'확인',status:'draft',revision:0});
 const {state}=await new GitHubStore(env,fetcher).read();assert.equal(state.items['hf-0000'].text,'서버 영구 저장');assert.equal(state.items['hf-0000'].revision,1);assert.equal(branch,true);
});
test('large GitHub state uses raw download without losing its concurrency SHA',async()=>{
 let reads=0;const fetcher=async(url,options)=>{
  reads++;return new Response(JSON.stringify(reads===1?{encoding:'none',sha:'large-sha'}:initialState()),{status:200});
 };
 const {state,version}=await new GitHubStore({LABEL_GITHUB_TOKEN:'test-placeholder'},fetcher).read();
 assert.equal(version,'large-sha');assert.equal(Object.keys(state.items).length,Object.keys(initialState().items).length);assert.equal(reads,2);
});
test('GitHub configuration errors distinguish missing, invalid, forbidden and rate limited tokens',async()=>{
 const missing=new GitHubStore({LABEL_GITHUB_TOKEN:'   '});
 await assert.rejects(()=>missing.read(),error=>error.code==='GITHUB_TOKEN_MISSING');
 for(const [status,headers,code] of [[401,{},'GITHUB_TOKEN_INVALID'],[403,{},'GITHUB_PERMISSION_DENIED'],[403,{'x-ratelimit-remaining':'0'},'GITHUB_RATE_LIMITED']]){
  const store=new GitHubStore({LABEL_GITHUB_TOKEN:'test-placeholder'},async()=>new Response('{}',{status,headers}));
  await assert.rejects(()=>store.read(),error=>error.code===code);
 }
});
test('non-concurrency GitHub validation failures are not hidden as editing conflicts',async()=>{
 const store=new GitHubStore({LABEL_GITHUB_TOKEN:'test-placeholder'},async(url)=>url.includes('/git/ref/')?new Response('{}',{status:200}):new Response(JSON.stringify({message:'Validation Failed'}),{status:422}));
 await assert.rejects(()=>store.write(initialState(),'old-sha','test'),error=>error.code==='GITHUB_WRITE_REJECTED');
});

test('catalog additions preserve saved annotations, revisions, history and claims',()=>{
 const old=initialState();mutate(old,'claim',alice,now);
 mutate(old,'save',{...alice,text:'보존할 전사',notes:'메모',status:'draft',revision:0},now);
 const original=structuredClone(old.items['hf-0000']);
 const catalog=initialState();catalog.items['hf-9999']={...structuredClone(catalog.items['hf-0001']),id:'hf-9999'};
 const merged=reconcileState(old,catalog);
 assert.deepEqual(merged.items['hf-0000'],original);
 assert.deepEqual(merged.items['hf-9999'],catalog.items['hf-9999']);
 assert.equal(old.items['hf-9999'],undefined);
 delete catalog.items['hf-0000'];assert.throws(()=>reconcileState(old,catalog),/재배포/);
});
test('GitHub old catalog reads append new images and retain the file concurrency SHA',async()=>{
 const old=initialState();const newest=Object.keys(old.items).at(-1);delete old.items[newest];
 old.items['hf-0000'].text='기존 공동 전사';old.items['hf-0000'].status='draft';old.items['hf-0000'].revision=7;
 const store=new GitHubStore({LABEL_GITHUB_TOKEN:'test'},async()=>new Response(JSON.stringify({sha:'old-sha',encoding:'base64',content:Buffer.from(JSON.stringify(old)).toString('base64')})));
 const {state,version}=await store.read();assert.equal(version,'old-sha');assert.ok(state.items[newest]);
 assert.equal(state.items['hf-0000'].text,'기존 공동 전사');assert.equal(state.items['hf-0000'].revision,7);
});
