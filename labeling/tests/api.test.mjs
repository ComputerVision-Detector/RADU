import test from 'node:test';
import assert from 'node:assert/strict';
import { handlerFor } from '../api/labeling.mjs';
import { initialState } from '../lib/state.mjs';
function setup(){let data=initialState(),version=0;const store={read:async()=>({state:structuredClone(data),version}),write:async(state,v)=>{if(v!==version)return false;data=structuredClone(state);version++;return true;}};return handlerFor(store);}
async function request(handler,url,method='GET',body,headers={}){const response={headers:{},setHeader(k,v){this.headers[k]=v;},status(code){this.code=code;return this;},json(data){this.data=data;},send(data){this.data=data;}};await handler({url,method,body,headers:{host:'label.example','content-type':'application/json',...headers}},response);return response;}
const who={author:'작성자',client_id:'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa',id:'hf-0000'};
test('API lists, claims, saves and exports actual persisted labels',async()=>{
 const handler=setup();assert.equal((await request(handler,'/?action=list')).data.stats.remaining,117);
 assert.equal((await request(handler,'/?action=claim','POST',who)).code,200);
 assert.equal((await request(handler,'/?action=save','POST',{...who,text:'전사',notes:'',status:'verified',revision:0})).code,200);
 const exported=await request(handler,'/?action=export&scope=verified');assert.equal(exported.data.samples.length,6);assert.equal(exported.data.samples.find(x=>x.id==='hf-0000').text,'전사');
 const zip=await request(handler,'/?action=export&format=zip');assert.equal(zip.headers['Content-Type'],'application/zip');assert.ok(Buffer.isBuffer(zip.data));
});
test('API rejects cross-origin mutation and returns conflict errors without credentials',async()=>{
 const handler=setup();assert.equal((await request(handler,'/?action=claim','POST',who,{origin:'https://other.example'})).code,403);
 assert.equal((await request(handler,'/?action=save','POST',{...who,text:'x',notes:'',status:'verified',revision:0})).code,409);
 assert.equal((await request(handler,'/?action=item&id=__proto__')).code,404);
});
