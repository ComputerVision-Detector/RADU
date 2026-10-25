// Admin commands: full-state backup to stdout; validated restore from stdin.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { reconcileState } from '../lib/state.mjs';
import { FileStore,validateState } from '../lib/file-store.mjs';
import { createBBoxState,reconcileBBoxState,validateBBoxState } from '../lib/bbox-state.mjs';
const command=process.argv[2];
const version=process.env.LABEL_STATE_VERSION || (process.argv.includes('--v2')?'v2':'v1');
if(!['v1','v2'].includes(version))throw new Error('LABEL_STATE_VERSION must be v1 or v2');
const root=fileURLToPath(new URL('../',import.meta.url));
const v2=version==='v2';
const file=v2
  ?process.env.LABEL_LOCAL_STORE_V2 || path.join(root,'.runtime/state-v2.json')
  :process.env.LABEL_LOCAL_STORE || path.join(root,'.runtime/state.json');
const store=new FileStore(file,v2?{
  backups:process.env.LABEL_BACKUP_DIR_V2 || path.join(root,'.runtime/backups-v2'),
  initialState:createBBoxState,reconcileState:reconcileBBoxState,validateState:validateBBoxState
}:{backups:process.env.LABEL_BACKUP_DIR});
if(command==='backup') {
  const {state}=await store.read();process.stdout.write(JSON.stringify(state,null,2)+'\n');
} else if(command==='restore' && process.argv.includes('--replace')) {
  const state=v2
    ?validateBBoxState(reconcileBBoxState(JSON.parse(readFileSync(0,'utf8'))))
    :validateState(reconcileState(JSON.parse(readFileSync(0,'utf8'))));
  // Restored claims must not keep an old browser's lease active.
  for(const item of Object.values(state.items))item.claim=null;
  await store.replace(state);
  console.log(`Restored ${version.toUpperCase()} state; annotations/revisions preserved and browser claims cleared.`);
} else {
  console.error('Usage: node tools/state.mjs backup > backup.json\n       node tools/state.mjs restore --replace < backup.json\n       Set LABEL_STATE_VERSION=v2 (or pass --v2) for the independent bbox dataset.');process.exitCode=1;
}
