// Admin commands: full-state backup to stdout; validated restore from stdin.
import { readFileSync } from 'node:fs';
import { FileStore,validateState } from '../lib/file-store.mjs';
const command=process.argv[2];
const file=process.env.LABEL_LOCAL_STORE || new URL('../.runtime/state.json',import.meta.url).pathname;
const store=new FileStore(file,{backups:process.env.LABEL_BACKUP_DIR});
if(command==='backup') {
  const {state}=await store.read();process.stdout.write(JSON.stringify(state,null,2)+'\n');
} else if(command==='restore' && process.argv.includes('--replace')) {
  const state=validateState(JSON.parse(readFileSync(0,'utf8')));
  // Restored claims must not keep an old browser's lease active.
  for(const item of Object.values(state.items))item.claim=null;
  await store.replace(state);
  console.log('Restored state; all text/revisions preserved and expired browser claims cleared.');
} else {
  console.error('Usage: node tools/state.mjs backup > backup.json\n       node tools/state.mjs restore --replace < backup.json');process.exitCode=1;
}
