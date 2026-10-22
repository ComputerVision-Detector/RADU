// Persistent single-host storage for self-hosting; no GitHub credentials required.
import { mkdirSync,readFileSync,existsSync,writeFileSync,openSync,closeSync,fsyncSync,renameSync,unlinkSync,readdirSync,statSync } from 'node:fs';
import path from 'node:path';
import { createHash,randomUUID } from 'node:crypto';
import { initialState,AppError,reconcileState } from './state.mjs';
const wait=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
export function validateState(state) {
  const seed=initialState();
  if(state?.schema!==1 || !state.items || Object.keys(state.items).length!==Object.keys(seed.items).length) throw new Error('라벨 백업 형식을 확인하세요.');
  for(const [id,original] of Object.entries(seed.items)) {
    const item=state.items[id];
    if(!item || item.id!==id || !['unlabeled','draft','verified'].includes(item.status) || (item.text!==null && typeof item.text!=='string') || typeof item.notes!=='string' || typeof item.annotator!=='string' || !Number.isInteger(item.revision) || item.revision<0) throw new Error(`라벨 형식 오류: ${id}`);
    if(!Array.isArray(item.history) || item.image?.file!==original.image.file || (item.claim!==null && (!item.claim || typeof item.claim.name!=='string' || typeof item.claim.client!=='string' || !Number.isFinite(Date.parse(item.claim.until))))) throw new Error(`저장 메타데이터 오류: ${id}`);
    if(item.status==='verified' && !item.text?.trim()) throw new Error(`빈 검수 완료 전사: ${id}`);
    if(item.seed_verified!==original.seed_verified) throw new Error(`기존 정답 보호 상태 오류: ${id}`);
    if(original.seed_verified && ['text','notes','annotator','status'].some(key=>item[key]!==original[key])) throw new Error(`기존 정답 변경: ${id}`);
  }
  return state;
}
function atomicWrite(file,bytes) {
  const temporary=`${file}.${randomUUID()}.tmp`;
  let fd;
  try {
    fd=openSync(temporary,'wx',0o600);writeFileSync(fd,bytes);fsyncSync(fd);closeSync(fd);fd=undefined;
    renameSync(temporary,file);
    // Flush the rename on Linux; some filesystems do not support directory fsync.
    let directory;try{directory=openSync(path.dirname(file),'r');fsyncSync(directory);}catch{}finally{if(directory!==undefined)closeSync(directory);}
  } finally {if(fd!==undefined)closeSync(fd);if(existsSync(temporary))unlinkSync(temporary);}
}
export class FileStore {
  constructor(file,options={}) {
    this.kind='persistent-file';this.file=path.resolve(file);this.lock=this.file+'.lock';
    this.backups=path.resolve(options.backups || path.join(path.dirname(this.file),'backups'));
    this.limit=options.limit ?? 50;
    if(!Number.isInteger(this.limit) || this.limit<0 || this.limit>500)throw new Error('Backup limit must be between 0 and 500');
    mkdirSync(path.dirname(this.file),{recursive:true});if(this.limit)mkdirSync(this.backups,{recursive:true});
  }
  async read() {
    if(!existsSync(this.file))return {state:initialState(),version:null};
    const raw=readFileSync(this.file);
    try{return {state:validateState(reconcileState(JSON.parse(raw))),version:hash(raw)};}
    catch{throw new AppError(503,'서버 저장 파일을 읽지 못했습니다. 관리자에게 알려 주세요. 백업으로 복구할 수 있습니다.');}
  }
  async acquire() {
    for(let attempt=0;attempt<100;attempt++) {
      try{const fd=openSync(this.lock,'wx',0o600);writeFileSync(fd,JSON.stringify({pid:process.pid,at:new Date().toISOString()}));closeSync(fd);return;}
      catch(error){
        if(error.code!=='EEXIST')throw error;
        try{if(Date.now()-statSync(this.lock).mtimeMs>60000){unlinkSync(this.lock);continue;}}catch(error){if(error.code==='ENOENT')continue;throw error;}
        await wait(20);
      }
    }
    throw new AppError(503,'다른 저장을 처리 중입니다. 잠시 뒤 다시 저장하세요.');
  }
  async replace(state) {
    validateState(state);await this.acquire();
    try {
      if(existsSync(this.file)) {
        mkdirSync(this.backups,{recursive:true});
        const name=new Date().toISOString().replace(/[:.]/g,'-')+'-before-restore-'+randomUUID()+'.json';
        atomicWrite(path.join(this.backups,name),readFileSync(this.file));
      }
      atomicWrite(this.file,Buffer.from(JSON.stringify(state,null,2)+'\n'));
    } finally {unlinkSync(this.lock);}
  }
  async write(state,version) {
    validateState(state);await this.acquire();
    try {
      const current=await this.read();if(current.version!==version)return false;
      if(current.version && this.limit) {
        const name=new Date().toISOString().replace(/[:.]/g,'-')+'-'+randomUUID()+'.json';
        atomicWrite(path.join(this.backups,name),Buffer.from(JSON.stringify(current.state,null,2)+'\n'));
      }
      atomicWrite(this.file,Buffer.from(JSON.stringify(state,null,2)+'\n'));
      if(this.limit){const files=readdirSync(this.backups).filter(n=>n.endsWith('.json')).sort().reverse();for(const name of files.slice(this.limit))unlinkSync(path.join(this.backups,name));}
      return true;
    } finally {unlinkSync(this.lock);}
  }
}
