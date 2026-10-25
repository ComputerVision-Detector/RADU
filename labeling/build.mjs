import { cpSync, mkdirSync, rmSync, existsSync } from 'node:fs';
rmSync('dist', { recursive: true, force: true }); mkdirSync('dist');
for (const file of ['index.html','app.js','app.css','assets']) {
  if (!existsSync(file)) throw new Error(`Missing required site asset: ${file}`);
  cpSync(file, `dist/${file}`, { recursive: true });
}
const proposals=process.env.LABEL_PROPOSALS_FILE || 'data/text-box-proposals.json';
if(process.env.LABEL_PROPOSALS_FILE && !existsSync(proposals))throw new Error(`Configured proposal sidecar does not exist: ${proposals}`);
if(existsSync(proposals)) {
  mkdirSync('dist/data',{recursive:true});
  cpSync(proposals,'dist/data/text-box-proposals.json');
}
console.log('Static annotation workspace built in dist/');
