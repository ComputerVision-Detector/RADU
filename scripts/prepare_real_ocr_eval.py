"""Build a small, manually transcribed real-photo recognition holdout.

Coordinates refer to EXIF-oriented images at 1368x1824 display size.
No predictions are used as ground truth. Full pages stay outside training.
"""
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
# Only clearly legible horizontal words/short lines; formulae/vertical text omitted.
ANNOTATIONS = {
    'IMG_2756.JPG': [
        ('찾는', [235, 250, 307, 300]),
        ('전형적인', [433, 242, 592, 296]),
        ('문제', [715, 237, 790, 291]),
        ('모멘텀', [63, 602, 170, 657]),
        ('같이', [98, 1014, 184, 1069]),
        ('학습률', [896, 593, 994, 647]),
    ],
    'IMG_2728.JPG': [
        ('Token IDs', [421, 92, 560, 139]),
        ('Embedding', [405, 189, 535, 235]),
        ('Transformer Block', [379, 278, 578, 322]),
        ('Attention Block', [501, 326, 670, 367]),
        ('RMSNorm', [572, 374, 695, 412]),
        ('QKV Projection', [578, 417, 746, 459]),
        ('RoPE', [578, 565, 653, 605]),
        ('SDPA', [578, 612, 650, 650]),
        ('Causal Mask', [680, 758, 823, 797]),
        ('Sliding Window', [679, 806, 838, 851]),
        ('Scaling', [681, 901, 779, 944]),
        ('Mask 적용', [678, 951, 809, 995]),
        ('Softmax', [672, 1055, 775, 1096]),
        ('Router', [523, 1372, 606, 1411]),
        ('Expert MLP', [524, 1478, 650, 1520]),
    ],
    'KakaoTalk_20261011_022928686.jpg': [
        ('컴퓨터 비전', [158, 399, 742, 555]),
        ('Computer Vision', [229, 720, 964, 905]),
        ('마우스', [245, 1080, 546, 1186]),
        ('컴퓨터', [231, 1328, 473, 1437]),
    ],
}


def main():
    out = ROOT / 'data/ocr_real_eval'
    (out / 'crops').mkdir(parents=True, exist_ok=True)
    rows = []
    previews = []
    font = ImageFont.truetype('/usr/share/fonts/truetype/nanum/NanumGothic.ttf', 20)
    for source_name, entries in ANNOTATIONS.items():
        source = ROOT / source_name
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        image = ImageOps.exif_transpose(Image.open(source)).convert('RGB')
        for index, (label, box) in enumerate(entries):
            coords = [round(box[0]*image.width/1368), round(box[1]*image.height/1824),
                      round(box[2]*image.width/1368), round(box[3]*image.height/1824)]
            crop = image.crop(coords)
            name = f'{source.stem}_{index:02d}.png'
            path = out / 'crops' / name
            crop.save(path)
            rows.append({'image': str(path.relative_to(ROOT)), 'label': label,
                         'source': source_name, 'source_sha256': digest,
                         'box_exif_oriented_xyxy': coords, 'annotation': 'manual_visual_transcription'})
            tile = Image.new('RGB', (620, 140), 'white')
            crop.thumbnail((600, 100))
            tile.paste(crop, (5, 0))
            ImageDraw.Draw(tile).text((5, 107), f'{len(rows):02d}. {label}', font=font, fill='black')
            previews.append(tile)
    (out / 'annotations.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'rec_gt.txt').write_text(''.join(f"{ROOT / r['image']}\t{r['label']}\n" for r in rows), encoding='utf-8')
    sheet = Image.new('RGB', (1240, ((len(previews)+1)//2)*140), 'white')
    for i, tile in enumerate(previews):
        sheet.paste(tile, ((i % 2)*620, (i//2)*140))
    sheet.save(out / 'contact_sheet.jpg')
    print(f'Prepared {len(rows)} manually transcribed samples from {len(ANNOTATIONS)} pages: {out}')


if __name__ == '__main__':
    main()
