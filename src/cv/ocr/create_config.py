"""Generate whole-word training and independent real-photo evaluation configs."""
import copy
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[3]
OCR = ROOT / 'src/cv/ocr'


def create_custom_yaml():
    base = OCR / 'PaddleOCR/configs/rec/PP-OCRv3/multi_language/korean_PP-OCRv3_mobile_rec.yml'
    config = yaml.safe_load(base.read_text(encoding='utf-8'))
    config['Global'].update({
        'pretrained_model': str(OCR / 'PaddleOCR/pretrain_models/korean_PP-OCRv3_rec_train/best_accuracy'),
        'checkpoints': None,
        'character_dict_path': str(OCR / 'PaddleOCR/ppocr/utils/dict/korean_dict.txt'),
        'save_model_dir': str(ROOT / 'data/models/ocr_word_finetuned'),
        'save_inference_dir': str(ROOT / 'data/models/ocr_word_inference'),
        'save_res_path': str(ROOT / 'data/models/ocr_word_finetuned/predictions.txt'),
        'epoch_num': 50, 'eval_batch_step': [0, 2000], 'save_epoch_step': 10,
        'cal_metric_during_train': True, 'distributed': False, 'use_gpu': True,
        'use_space_char': True, 'max_text_length': 25, 'seed': 2026,
    })
    config['Optimizer']['lr'].update(learning_rate=0.0001, warmup_epoch=2)
    config['Metric'].update(ignore_space=False, is_filter=False)
    for stage, split in (('Train', 'train'), ('Eval', 'val')):
        config[stage]['dataset'].update(
            name='SimpleDataSet', data_dir=str(ROOT),
            label_file_list=[str(ROOT / f'data/ai_hub_OCR/rec_gt_word_{split}.txt')])
        config[stage]['loader'].update(batch_size_per_card=64, num_workers=8 if stage == 'Train' else 4)
    for op in config['Train']['dataset']['transforms']:
        if 'RecConAug' in op:
            op['RecConAug'].update(prob=0.3, ext_data_num=1)
    output = OCR / 'custom_rec_korean.yml'
    output.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
    real = copy.deepcopy(config)
    real['Eval']['dataset']['label_file_list'] = [str(ROOT / 'data/ocr_real_eval/rec_gt.txt')]
    real['Eval']['loader'].update(batch_size_per_card=16, num_workers=0)
    (OCR / 'custom_rec_korean_real_eval.yml').write_text(
        yaml.safe_dump(real, allow_unicode=True, sort_keys=False), encoding='utf-8')
    print(f'Generated {output} and custom_rec_korean_real_eval.yml')


if __name__ == '__main__':
    create_custom_yaml()
