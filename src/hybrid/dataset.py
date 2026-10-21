"""Sample image-only HF repositories without scanning or buffering every image."""
from pathlib import Path
import random
from PIL import Image, ImageOps


def load_sample(dataset_id, split, seed):
    from huggingface_hub import HfApi, hf_hub_download
    from src.evaluation.benchmark_ocr_visual import load_random_records, extract_image
    info = HfApi().dataset_info(dataset_id, timeout=30)
    files = [item.rfilename for item in info.siblings]
    extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif'}
    images = sorted(name for name in files if Path(name).suffix.lower() in extensions)
    structured = any(Path(name).suffix.lower() in {'.parquet', '.csv', '.jsonl', '.arrow'} for name in files)
    if images and not structured and all('/' not in name for name in images):
        if split not in (None, 'train'):
            raise ValueError('이미지 전용 데이터셋은 train split을 사용하세요.')
        name = random.Random(seed).choice(images)
        path = hf_hub_download(dataset_id, name, repo_type='dataset', revision=info.sha)
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
        return image, {'dataset': dataset_id, 'revision': info.sha, 'file': name,
                       'split': 'train', 'seed': seed, 'loader': 'image repository sampling'}
    # Standard HF upload layout: bypass slow format discovery over loose image files.
    actual_split = split or 'train'
    shards = sorted(name for name in files if name.startswith(f'data/{actual_split}-') and name.endswith('.parquet'))
    if shards:
        import pyarrow.parquet as parquet
        rng = random.Random(seed)
        name = rng.choice(shards)
        path = hf_hub_download(dataset_id, name, repo_type='dataset', revision=info.sha)
        reader = parquet.ParquetFile(path)
        if reader.metadata.num_rows == 0:
            raise ValueError('선택한 parquet 파일에 표본이 없습니다.')
        row = rng.randrange(reader.metadata.num_rows)
        offset = row
        for group in range(reader.num_row_groups):
            count = reader.metadata.row_group(group).num_rows
            if offset < count:
                record = reader.read_row_group(group).slice(offset, 1).to_pylist()[0]
                _, image = extract_image(record)
                return image, {'dataset': dataset_id, 'revision': info.sha, 'file': name,
                               'row': row, 'split': actual_split, 'seed': seed,
                               'loader': 'seeded parquet shard and row sampling'}
            offset -= count
    records, actual_split = load_random_records(dataset_id, split, 1, seed)
    _, image = extract_image(records[0])
    return image, {'dataset': dataset_id, 'split': actual_split, 'seed': seed,
                   'loader': 'existing streaming loader'}
