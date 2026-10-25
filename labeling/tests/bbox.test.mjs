import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync,readFileSync,existsSync,rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { createBBoxState,reconcileBBoxState,validateBBoxState,mutateBBox } from '../lib/bbox-state.mjs';
import { initialState } from '../lib/state.mjs';
import { FileStore } from '../lib/file-store.mjs';
import { handlerFor } from '../api/labeling.mjs';

const actor={id:'hf-0000',author:'V2 검증',client_id:'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa'};
const box={id:'line-1',bbox:[10,10,100,50],text:'한국어 English 123',prediction:'OCR 후보',order:1,uncertain:false};
const save={...actor,status:'verified',revision:0,boxes:[box],notes:'',no_text:false,proposal_version:''};
const options={initialState:createBBoxState,reconcileState:reconcileBBoxState,validateState:validateBBoxState};
async function request(handler,url,body){
  const response={setHeader(){},status(code){this.code=code;return this;},json(data){this.data=data;},send(data){this.data=data;}};
  await handler({url,method:body?'POST':'GET',body,headers:{host:'localhost','content-type':'application/json'}},response);
  return response;
}

test('V2 starts with 163 empty annotations and never imports V1 text',()=>{
  const state=validateBBoxState(createBBoxState());
  assert.equal(Object.keys(state.items).length,163);
  for(const item of Object.values(state.items)){
    assert.equal(item.bbox_annotation.status,'unlabeled');
    assert.deepEqual(item.bbox_annotation.boxes,[]);
    assert.equal(item.claim,null);
  }
});

test('V2 rejects invalid, duplicate, uncertain and incomplete verified boxes',()=>{
  for(const changes of [
    {boxes:[]},{boxes:[{...box,bbox:[-1,10,100,50]}]},
    {boxes:[{...box,bbox:[10,10,10,50]}]},
    {boxes:[{...box,text:''}]},{boxes:[{...box,uncertain:true}],notes:'판독 보류'},
    {boxes:[box,{...box,id:'line-2'}]},
    {boxes:[box,{...box,order:2}]},{no_text:true},
    {status:'draft',boxes:[{...box,uncertain:true}]}
  ]){
    const state=createBBoxState();mutateBBox(state,'claim',actor);
    assert.throws(()=>mutateBBox(state,'save',{...save,...changes}),error=>error.status===400);
    assert.equal(state.items[actor.id].bbox_annotation.revision,0);
  }
});

test('V2 enforces ownership, revisions, verified release and explicit no-text',()=>{
  const state=createBBoxState(),now=Date.now();mutateBBox(state,'claim',actor,now);
  assert.throws(()=>mutateBBox(state,'claim',{...actor,client_id:'bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb'},now),error=>error.status===409);
  mutateBBox(state,'save',{...save,status:'draft',boxes:[{...box,uncertain:true}],notes:'판読 불확실'},now);
  assert.throws(()=>mutateBBox(state,'save',save,now),error=>error.status===409);
  mutateBBox(state,'save',{...save,revision:1},now);
  assert.equal(state.items[actor.id].claim,null);
  mutateBBox(state,'claim',{...actor,id:'hf-0001'},now);
  mutateBBox(state,'save',{...save,id:'hf-0001',boxes:[],no_text:true},now);
  validateBBoxState(state);
  assert.equal(state.items['hf-0001'].bbox_annotation.status,'verified');
});

test('V2 reconciliation refuses changed source hashes/dimensions without erasing labels',()=>{
  const state=createBBoxState();mutateBBox(state,'claim',actor);mutateBBox(state,'save',save);
  for(const [key,value] of [['source_sha256','0'.repeat(64)],['source_width',1],['source_height',1]]){
    const changed=structuredClone(state);changed.items[actor.id].image[key]=value;
    assert.throws(()=>reconcileBBoxState(changed),/원본 이미지/);
    assert.throws(()=>validateBBoxState(changed));
    assert.equal(changed.items[actor.id].bbox_annotation.boxes[0].text,box.text);
  }
  const recompressed=structuredClone(state);recompressed.items[actor.id].image.asset_sha256='0'.repeat(64);
  assert.deepEqual(validateBBoxState(reconcileBBoxState(recompressed)),state);
});

test('V2 API persists separately, exports provenance and leaves V1 bytes untouched',async()=>{
  const directory=mkdtempSync(path.join(tmpdir(),'radu-v2-test-'));
  try{
    const v1=new FileStore(path.join(directory,'v1.json'),{limit:0});
    const v2=new FileStore(path.join(directory,'v2.json'),{...options,limit:0});
    await v1.write(initialState(),null);
    const before=readFileSync(v1.file),handler=handlerFor(v1,v2);
    assert.equal((await request(handler,'/?action=health&version=v2')).data.dataset_version,'v2');
    assert.equal(existsSync(v2.file),false,'health must not initialize state on disk');
    assert.equal((await request(handler,'/?action=list&version=v2')).data.stats.verified,0);
    assert.equal((await request(handler,'/?action=claim&version=v2',actor)).code,200);
    assert.equal((await request(handler,'/?action=save&version=v2',save)).code,200);
    const exported=await request(handler,'/?action=export&scope=verified&version=v2');
    assert.equal(exported.data.samples.length,1);
    assert.equal(exported.data.samples[0].source_sha256,createBBoxState().items[actor.id].image.source_sha256);
    assert.deepEqual(exported.data.samples[0].boxes,[box]);
    assert.equal((await new FileStore(v2.file,{...options,limit:0}).read()).state.items[actor.id].bbox_annotation.status,'verified');
    assert.deepEqual(readFileSync(v1.file),before);
    assert.equal((await request(handlerFor(v1),'/?action=claim&version=v2',actor)).code,503);
    assert.equal((await request(handlerFor(v1,v1),'/?action=claim&version=v2',actor)).code,503);
    assert.equal((await request(handler,'/?action=claim&version=v3',actor)).code,400);
    assert.deepEqual(readFileSync(v1.file),before);
  }finally{rmSync(directory,{recursive:true,force:true});}
});
