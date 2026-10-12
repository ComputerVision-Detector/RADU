"""Validate prepared data and run a forward/loss check WITHOUT training or saving weights."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/cv/ocr'))
from recognition_runtime import RecognitionRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'src/cv/ocr/custom_rec_korean.yml')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    dictionary = Path(cfg['Global']['character_dict_path'])
    characters = set(dictionary.read_text(encoding='utf-8').splitlines()) | {' '}
    report = json.loads((ROOT / 'data/ai_hub_OCR/word_conversion_report.json').read_text())
    if report['dictionary_sha256'] != hashlib.sha256(dictionary.read_bytes()).hexdigest():
        raise ValueError('Dictionary changed after dataset preparation')
    if cfg['Global'].get('checkpoints'):
        raise ValueError('Initial word training must not resume the old character run')
    sources, samples, counts = {}, {}, {}
    for stage, split in (('Train', 'train'), ('Eval', 'val')):
        path = Path(cfg[stage]['dataset']['label_file_list'][0])
        if hashlib.sha256(path.read_bytes()).hexdigest() != report[split]['sha256']:
            raise ValueError(f'{split} label file changed since conversion')
        lines = path.read_text(encoding='utf-8').splitlines()
        sources[split] = set()
        for line in lines:
            image, label = line.split('\t', 1)
            if not label or len(label) > cfg['Global']['max_text_length'] or not set(label) <= characters:
                raise ValueError(f'Invalid label: {line}')
            sources[split].add(Path(image).stem)
        counts[split] = len(lines)
        samples[stage] = random.Random(2026).sample(lines, min(len(lines), max(32, args.batch_size)))
        if report[split]['counts']['ok'] != len(lines):
            raise ValueError('Conversion count differs from label file')
    overlap = sources['train'] & sources['val']
    if overlap:
        raise ValueError(f'Same source image in train and val: {sorted(overlap)[:10]}')
    real_labels = ROOT / 'data/ocr_real_eval/rec_gt.txt'
    real_lines = real_labels.read_text(encoding='utf-8').splitlines()
    for line in real_lines:
        image, label = line.split('\t', 1)
        if not Path(image).is_file() or not set(label) <= characters or len(label) > cfg['Global']['max_text_length']:
            raise ValueError(f'Invalid real-photo sample: {line}')
    runtime = RecognitionRuntime(args.config, cfg['Global']['pretrained_model'], args.device)
    import paddle
    from ppocr.data import build_dataloader
    from ppocr.losses import build_loss
    from ppocr.utils.logging import get_logger
    random.seed(2026)
    np.random.seed(2026)
    paddle.seed(2026)
    smoke = copy.deepcopy(cfg)
    smoke['Loss']['loss_config_list'][1]['SARLoss'] = {'ignore_index': len(runtime.decoder.character) + 1}
    forward = {}
    with tempfile.TemporaryDirectory(prefix='ocr_preflight_') as temporary:
        for stage in ('Train', 'Eval'):
            subset = Path(temporary) / f'{stage}.txt'
            subset.write_text('\n'.join(samples[stage])+'\n', encoding='utf-8')
            smoke[stage]['dataset']['label_file_list'] = [str(subset)]
            smoke[stage]['loader'].update(batch_size_per_card=args.batch_size, num_workers=0, use_shared_memory=False)
            loader = build_dataloader(smoke, stage, paddle.get_device(), get_logger())
            batch = next(iter(loader))
            if not bool(paddle.isfinite(batch[0]).all()):
                raise ValueError('Non-finite image tensor')
            with paddle.no_grad():
                if stage == 'Train':
                    runtime.model.train()
                    predictions = runtime.model(batch[0], data=batch[1:])
                    loss = build_loss(smoke['Loss'])(predictions, batch)
                    forward['loss'] = {key: float(value.mean()) for key, value in loss.items()}
                    if not all(np.isfinite(v) for v in forward['loss'].values()):
                        raise ValueError('Non-finite loss')
                else:
                    runtime.model.eval()
                    predictions = runtime.model(batch[0])
                    forward['prediction_shape'] = list(predictions.shape)
                    if predictions.shape[-1] != len(runtime.decoder.character):
                        raise ValueError('Decoder/output dimensions differ')
            forward[stage] = {'image_shape': list(batch[0].shape), 'label_lengths': batch[3].numpy().tolist()}
    result = {'device': args.device, 'counts': counts, 'real_photo_samples': len(real_lines),
              'train_val_source_overlap': len(overlap), 'forward_only': True,
              'optimizer_steps': 0, 'checks': forward}
    destination = ROOT / 'data/ocr_diagnostics' / ('preflight_gpu.json' if args.device.startswith('gpu') else 'preflight_cpu.json')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    print('READY: data, official weights, decoder, batch loading, CTC/SAR loss validated; no training performed.')


if __name__ == '__main__':
    main()
