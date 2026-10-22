"""Append HF images to the labeling catalog; frozen benchmarks and saved labels stay intact."""
import argparse
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / 'labeling'
BASE = ROOT / 'data/benchmark/qwen_ablation_pilot'
CATALOG = APP / 'data/dataset.catalog.json'

def read(path):
    return json.loads(path.read_text())

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='Download and append; otherwise only report changes')
    parser.add_argument('--revision', default='main')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    seed = read(APP / 'data/seed.json')
    repo = seed['dataset']
    api = HfApi()
    revision = api.dataset_info(repo, revision=args.revision).sha
    files = api.list_repo_files(repo, repo_type='dataset', revision=revision)
    images = sorted(f for f in files if Path(f).suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'})
    catalog = read(CATALOG) if CATALOG.exists() else read(BASE / 'additional_images.manifest.json')
    samples = catalog['samples']
    by_file = {s['hf_file']: s for s in samples if s.get('hf_file')}
    # The dataset migrated from parquet to loose images. Its provenance CSV preserves original IDs.
    if 'image_manifest.csv' in files:
        with open(hf_hub_download(repo, 'image_manifest.csv', repo_type='dataset', revision=revision), newline='') as stream:
            rows = list(csv.DictReader(stream))
        by_origin = {(s.get('shard'), str(s.get('row'))): s for s in samples if s.get('shard')}
        for row in rows:
            sample = by_origin.get((row.get('source_parquet'), row.get('source_row')))
            if sample and row['image_file'] not in by_file:
                sample['hf_file'] = row['image_file']
                by_file[row['image_file']] = sample
    if any(not s.get('hf_file') for s in samples):
        raise RuntimeError('Existing image provenance cannot be resolved safely; refusing to reassign IDs')
    new_files = [f for f in images if f not in by_file]
    print(f'HF revision: {revision}\nHF images: {len(images)}; existing catalog: {len(samples)}; additions: {len(new_files)}', flush=True)
    if not args.apply:
        return
    next_id = max(int(s['id'].split('-')[1]) for s in samples) + 1
    stage = APP / '.runtime/sync-assets'
    stage.mkdir(parents=True, exist_ok=True)
    def prepare(pair):
        offset, filename = pair
        id_ = f'hf-{next_id + offset:04d}'
        source = hf_hub_download(repo, filename, repo_type='dataset', revision=revision)
        original_sha = hashlib.sha256(Path(source).read_bytes()).hexdigest()
        with Image.open(source) as original:
            image = ImageOps.exif_transpose(original).convert('RGB')
            width, height = image.size
            png = BASE / 'images' / f'{id_}.png'
            if png.exists():
                with Image.open(png) as previous:
                    if previous.size != image.size or previous.convert('RGB').tobytes() != image.tobytes():
                        raise RuntimeError(f'Existing image differs: {id_}')
            else:
                image.save(png)
            source_sha = hashlib.sha256(png.read_bytes()).hexdigest()
            image.thumbnail((3200, 3200), Image.Resampling.LANCZOS)
            target = stage / f'{id_}.webp'
            image.save(target, quality=85, method=3)
            asset = dict(id=id_, file=target.name, width=image.width, height=image.height,
                         source_sha256=source_sha, sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                         bytes=target.stat().st_size, encoding='WebP quality 85, max side 3200; web reading copy, not benchmark input')
        sample = dict(id=id_, image=f'images/{id_}.png', sha256=source_sha, width=width, height=height,
                      hf_file=filename, hf_revision=revision, hf_source_sha256=original_sha)
        print(f'Prepared {id_}: {filename}', flush=True)
        return sample, asset
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        prepared = list(pool.map(prepare, enumerate(new_files)))
    manifest = read(APP / 'data/assets.manifest.json')
    for sample, asset in prepared:
        id_ = sample['id']
        (stage / asset['file']).replace(APP / 'assets' / asset['file'])
        samples.append(sample)
        manifest['images'].append(asset)
        seed['items'][id_] = dict(id=id_, text=None, status='unlabeled', annotator='', notes='', revision=0,
                                 seed_verified=False, claim=None, history=[], image=dict(file=f'assets/{id_}.webp', source_width=sample['width'], source_height=sample['height']))
        label = BASE / 'labels' / f'{id_}.json'
        if not label.exists():
            write(label, dict(id=id_, text=None, status='unlabeled', annotator='', notes=''))
    seed['revision'] = revision
    write(CATALOG, dict(dataset=repo, revision=revision, purpose='Append-only labeling catalog; frozen evaluation manifests unchanged', samples=samples))
    write(APP / 'data/assets.manifest.json', manifest)
    write(APP / 'data/seed.json', seed)
    print(f'Catalog ready: {len(samples)} images. Redeploy the labeling app to publish.')

if __name__ == '__main__':
    main()
