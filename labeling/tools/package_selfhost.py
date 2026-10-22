"""Build an allowlisted, credential-free handoff archive (standard Python only)."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
APP=Path(__file__).resolve().parents[1]
FILES=['START_HERE.md','README.md','VERCEL_ADMIN_SETUP.md','Dockerfile','.dockerignore','compose.yaml','compose.domain.yaml','Caddyfile',
       '.env.selfhost.example','.env.example','package.json','server.mjs','dev.mjs','index.html','app.js','app.css',
       'build.mjs','ignore-build.mjs','vercel.json','api/labeling.mjs','data/seed.json','data/assets.manifest.json','data/dataset.catalog.json',
       'tools/state.mjs']

def package(output):
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    paths=[APP/name for name in FILES]+sorted((APP/'lib').glob('*.mjs'))+sorted((APP/'assets').glob('hf-*.webp'))
    seed=json.loads((APP/'data/seed.json').read_text())
    assert all((APP/item['image']['file']).is_file() for item in seed['items'].values()),'Catalog reading images are required'
    revision=subprocess.run(['git','rev-parse','HEAD'],cwd=APP,text=True,capture_output=True)
    provenance='Source revision: '+(revision.stdout.strip() if revision.returncode==0 else 'standalone package')+'\n'
    hashes=[]
    with tarfile.open(output,'w:gz',compresslevel=1) as archive:
        for path in paths:
            relative=path.relative_to(APP).as_posix();assert path.is_file(),relative
            archive.add(path,arcname='hybridocr-labeling/'+relative,recursive=False)
            hashes.append(hashlib.sha256(path.read_bytes()).hexdigest()+'  '+relative)
        for name,text in [('SOURCE_REVISION.txt',provenance),('MANIFEST.sha256','\n'.join(hashes)+'\n')]:
            raw=text.encode();info=tarfile.TarInfo('hybridocr-labeling/'+name);info.size=len(raw);info.mode=0o644;archive.addfile(info,io.BytesIO(raw))
    digest=hashlib.sha256(output.read_bytes()).hexdigest()
    Path(str(output)+'.sha256').write_text(digest+'  '+output.name+'\n')
    print(f'Created {output} ({output.stat().st_size/1024**2:.1f} MiB), {len(paths)} allowlisted files')
    return output

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=APP.parent/'artifacts/hybridocr-labeling-selfhost.tar.gz')
    package(parser.parse_args().output)
