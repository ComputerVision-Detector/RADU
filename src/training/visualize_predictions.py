"""Build a local HTML page comparing ground truth and model predictions per image.

Works on unfinished evaluations, so it can be rerun while predictions are still being written.
Output truncated at the token limit is not valid JSON; its complete elements are still drawn,
because that is exactly the failure worth seeing.

  python -m src.training.visualize_predictions --data data/processed/radu/validation.jsonl \
      --run pilot=models/radu/pilot/eval_validation/adapter --run new=models/radu/new/eval_validation/adapter \
      --output models/radu/new/visual_validation
"""
from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path

from .evaluate import valid_box, valid_point
from .prepare import read_jsonl

THUMBNAIL_EDGE = 1400


def salvage_elements(text):
    """Return (nodes, arrows, relations, truncated), keeping every complete element of each list."""
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return [data.get(key) if isinstance(data.get(key), list) else [] for key in ("nodes", "arrows", "relations")] + [False]
        return [], [], [], True
    except ValueError:
        pass
    decoder = json.JSONDecoder()
    found = []
    for key in ("nodes", "arrows", "relations"):
        items = []
        start = text.find(f'"{key}"')
        position = text.find("[", start) if start >= 0 else -1
        if position >= 0:
            position += 1
            while True:
                while position < len(text) and text[position] in " \t\r\n,":
                    position += 1
                try:
                    item, position = decoder.raw_decode(text, position)
                except ValueError:
                    break
                items.append(item)
        found.append(items)
    return found[0], found[1], found[2], True


def elements_svg(nodes, arrows, width, height):
    """Inline SVG in original image coordinates. Model text is escaped; bad geometry is skipped."""
    stroke = max(width, height) / 500
    font = max(width, height) / 70
    parts = [f'<svg class="overlay" viewBox="0 0 {width} {height}" preserveAspectRatio="none">']
    for node in nodes:
        box = node.get("bbox") if isinstance(node, dict) else None
        if valid_box(box):
            x0, y0, x1, y1 = box
            parts.append(f'<rect class="node" x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" stroke-width="{stroke:.1f}"/>')
            parts.append(f'<text class="kind" x="{x0}" y="{max(font, y0 - stroke):.1f}" font-size="{font:.0f}">{escape(str(node.get("kind", "")))}</text>')
    for arrow in arrows:
        start, end = (arrow.get("start"), arrow.get("end")) if isinstance(arrow, dict) else (None, None)
        if valid_point(start) and valid_point(end):
            parts.append(f'<line class="arrow" x1="{start[0]}" y1="{start[1]}" x2="{end[0]}" y2="{end[1]}" stroke-width="{stroke:.1f}" marker-end="url(#head)"/>')
    parts.append("</svg>")
    return "".join(parts)


def panel(title, image, svg, facts):
    return (f'<figure><figcaption><b>{escape(title)}</b> {facts}</figcaption>'
            f'<div class="canvas"><img loading="lazy" src="{escape(image)}" alt="">{svg}</div></figure>')


def score_text(score):
    def part(label, key):
        tp, predicted, gold = score[key]
        return f'{label} {tp}/{gold} (예측 {predicted})'
    return " · ".join((part("노드", "nodes"), part("화살표", "arrows"), part("관계", "relations")))


def sample_f1(score):
    tp = sum(score[key][0] for key in ("nodes", "arrows"))
    total = sum(score[key][1] + score[key][2] for key in ("nodes", "arrows"))
    return 2 * tp / total if total else 0.0


def build_card(row, thumbnail, width, height, runs):
    gold = json.loads(row["completion"])
    gold_facts = f'노드 {len(gold.get("nodes", []))} · 화살표 {len(gold.get("arrows", []))} · 관계 {len(gold.get("relations", []))}'
    panels = [panel("정답", thumbnail, elements_svg(gold.get("nodes", []), gold.get("arrows", []), width, height), gold_facts)]
    last_f1, last_failed = None, False
    for name, predictions in runs:
        record = predictions.get(row["id"])
        if record is None:
            panels.append(panel(name, thumbnail, "", '<span class="wait">평가 대기</span>'))
            continue
        nodes, arrows, _, truncated = salvage_elements(record["prediction"])
        state = '<span class="bad">파싱 실패, 복원 표시</span> ' if truncated or not record["score"]["parsed"] else ""
        panels.append(panel(name, thumbnail, elements_svg(nodes, arrows, width, height), state + score_text(record["score"])))
        last_f1, last_failed = sample_f1(record["score"]), not record["score"]["parsed"]
    attributes = f'data-dataset="{escape(row["dataset"])}" data-failed="{int(last_failed)}" data-f1="{-1 if last_f1 is None else round(last_f1, 4)}"'
    return f'<section class="card" {attributes}><h2>{escape(row["id"])}</h2><div class="panels">{"".join(panels)}</div></section>'


PAGE = """<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>RADU 추론 비교</title><style>
:root{color-scheme:light dark;--bg:#f6f6f4;--fg:#1c1c1a;--muted:#62625c;--card:#fff;--line:#d8d8d2;--bad:#b3261e}
@media (prefers-color-scheme:dark){:root{--bg:#191917;--fg:#ecece8;--muted:#a5a59e;--card:#242421;--line:#3c3c38;--bad:#f2968f}}
body{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
h1{font-size:20px;margin:0 0 4px}h2{font-size:14px;margin:0 0 8px;overflow-wrap:anywhere}
.note{color:var(--muted);margin:0 0 12px}
.bar{position:sticky;top:0;z-index:2;display:flex;flex-wrap:wrap;gap:12px;align-items:center;padding:10px 0;background:var(--bg)}
.legend i{display:inline-block;width:14px;height:0;border-top:3px solid;margin:0 4px 3px 10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;margin-bottom:14px}
.panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}
figure{margin:0;min-width:0}figcaption{min-height:3em;margin-bottom:6px;color:var(--muted);font-size:12px}figcaption b{color:var(--fg);font-size:13px}
.canvas{position:relative;line-height:0;background:#fff}.canvas img{width:100%;height:auto;display:block}
.overlay{position:absolute;inset:0;width:100%;height:100%;overflow:hidden}
.node{fill:none;stroke:#1565c0}.arrow{stroke:#d81b60}.kind{fill:#0d47a1;paint-order:stroke;stroke:#fff;stroke-width:3px}
.bad{color:var(--bad);font-weight:600}.wait{font-style:italic}.hidden{display:none}
</style></head><body><svg width="0" height="0" style="position:absolute"><defs><marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#d81b60"/></marker></defs></svg>
<h1>RADU 추론 비교</h1><p class="note">__SUMMARY__</p>
<div class="bar"><label>데이터셋 <select id="dataset"><option value="">전체</option>__OPTIONS__</select></label>
<label><input type="checkbox" id="failed"> 파싱 실패만</label>
<label>정렬 <select id="sort"><option value="">원래 순서</option><option value="low">마지막 모델 점수 낮은 순</option><option value="high">높은 순</option></select></label>
<span id="count"></span><span class="legend"><i style="color:#1565c0"></i>노드<i style="color:#d81b60"></i>화살표</span></div>
<main id="cards">__CARDS__</main><script>
const cards=[...document.querySelectorAll('.card')],list=document.getElementById('cards');
const dataset=document.getElementById('dataset'),failed=document.getElementById('failed'),sort=document.getElementById('sort');
function update(){let shown=0;for(const c of cards){const on=(!dataset.value||c.dataset.dataset===dataset.value)&&(!failed.checked||c.dataset.failed==='1');c.classList.toggle('hidden',!on);shown+=on}
document.getElementById('count').textContent=shown+'장';
const order=sort.value?[...cards].sort((a,b)=>(sort.value==='low'?1:-1)*(a.dataset.f1-b.dataset.f1)):cards;for(const c of order)list.append(c)}
for(const e of [dataset,failed,sort])e.addEventListener('change',update);update();
</script></body></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="Prepared JSONL the predictions were made from")
    parser.add_argument("--run", action="append", required=True, help="name=evaluation directory, shown in the given order")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from PIL import Image

    runs = []
    for spec in args.run:
        name, _, directory = spec.partition("=")
        path = Path(directory) / "predictions.jsonl"
        runs.append((name, {record["id"]: record for record in read_jsonl(path)} if path.exists() else {}))
    output = Path(args.output)
    images = output / "images"
    images.mkdir(parents=True, exist_ok=True)
    cards, datasets, rows = [], [], list(read_jsonl(args.data))
    for index, row in enumerate(rows):
        thumbnail = images / f"{index:04d}.jpg"
        with Image.open(row["image"]) as image:
            width, height = image.size
            if not thumbnail.exists():
                copy = image.convert("RGB")
                copy.thumbnail((THUMBNAIL_EDGE, THUMBNAIL_EDGE))
                copy.save(thumbnail, quality=80)
        cards.append(build_card(row, f"images/{thumbnail.name}", width, height, runs))
        if row["dataset"] not in datasets:
            datasets.append(row["dataset"])
    done = ", ".join(f"{name} {sum(row['id'] in predictions for row in rows)}/{len(rows)}" for name, predictions in runs)
    options = "".join(f'<option value="{escape(name)}">{escape(name)}</option>' for name in datasets)
    page = PAGE.replace("__SUMMARY__", escape(f"이미지 {len(rows)}장. 완료된 예측: {done}. 좌표는 원본 이미지 기준입니다.")).replace("__OPTIONS__", options).replace("__CARDS__", "".join(cards))
    (output / "index.html").write_text(page, encoding="utf-8")
    print(output / "index.html", done)


if __name__ == "__main__":
    main()
