import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import seed from '../data/seed.json' with {type:'json'};
import manifest from '../data/assets.manifest.json' with {type:'json'};
test('every labeled image has a matching published reading asset',()=>{
 assert.equal(manifest.images.length,Object.keys(seed.items).length);
 for(const entry of manifest.images){
  assert.equal(seed.items[entry.id].image.file,`assets/${entry.file}`);
  const bytes=readFileSync(new URL(`../assets/${entry.file}`,import.meta.url));
  assert.equal(createHash('sha256').update(bytes).digest('hex'),entry.sha256,entry.id);
 }
});
