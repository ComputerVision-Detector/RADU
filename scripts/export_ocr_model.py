"""Export one checkpoint with its training dictionary, then validate metadata."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'src/cv/ocr/custom_rec_korean.yml')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'data/models/ocr_word_finetuned/best_accuracy')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/models/ocr_word_inference')
    args = parser.parse_args()
    config = args.config.resolve()
    checkpoint = args.checkpoint.resolve()
    if checkpoint.suffix == '.pdparams':
        checkpoint = checkpoint.with_suffix('')
    weights = Path(str(checkpoint) + '.pdparams')
    if not weights.is_file():
        raise FileNotFoundError(weights)
    output = args.output.resolve()
    env = dict(os.environ, OMP_NUM_THREADS='1')
    subprocess.run([sys.executable, 'tools/export_model.py', '-c', str(config), '-o',
                    f'Global.pretrained_model={checkpoint}', 'Global.checkpoints=null',
                    'Global.use_gpu=False', f'Global.save_inference_dir={output}'],
                   cwd=ROOT / 'src/cv/ocr/PaddleOCR', env=env, check=True)
    sys.path.insert(0, str(ROOT / 'src/cv/ocr'))
    from infer_custom import validate_export
    validate_export(output, config)
    training = yaml.safe_load(config.read_text(encoding='utf-8'))
    dictionary = Path(training['Global']['character_dict_path'])
    (output / 'provenance.json').write_text(json.dumps({
        'checkpoint': str(weights), 'checkpoint_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
        'config': str(config), 'config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
        'dictionary_sha256': hashlib.sha256(dictionary.read_bytes()).hexdigest(),
    }, indent=2), encoding='utf-8')
    print(f'Validated custom export: {output}')


if __name__ == '__main__':
    main()
