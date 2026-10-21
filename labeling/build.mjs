import { cpSync, mkdirSync, rmSync, existsSync } from 'node:fs';
rmSync('dist', { recursive: true, force: true }); mkdirSync('dist');
for (const file of ['index.html','app.js','app.css','assets']) {
  if (!existsSync(file)) throw new Error(`Missing required site asset: ${file}`);
  cpSync(file, `dist/${file}`, { recursive: true });
}
console.log('Static annotation workspace built in dist/');
