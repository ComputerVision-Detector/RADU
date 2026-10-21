"""Turn coordinate-grounded OCR/Qwen evidence into an editable vector document."""
from __future__ import annotations
from difflib import SequenceMatcher
import math
import re
from PIL import Image


def bounds(raw, width, height):
    if not isinstance(raw, (list, tuple)) or not raw:
        return None
    if isinstance(raw[0], (list, tuple)):
        if any(not isinstance(point, (list, tuple)) or len(point) < 2 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in point[:2]) for point in raw):
            return None
        raw = [min(p[0] for p in raw), min(p[1] for p in raw), max(p[0] for p in raw), max(p[1] for p in raw)]
    if len(raw) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in raw):
        return None
    x, y, r, b = raw
    if not (0 <= x < r <= width and 0 <= y < b <= height):
        return None
    return list(raw)


def normalized(text):
    return re.sub(r'\s+', '', re.sub(r'^\s*(?:[-*#]+|\d+\.)\s*', '', text)).casefold()


def layout_candidates(image, ocr_lines=()):
    import cv2
    from src.cv.detector import detect_components
    height, width = image.shape[:2]
    scale = min(1.0, 1600 / max(height, width))
    small = cv2.resize(image, (round(width*scale), round(height*scale))) if scale < 1 else image
    from src.cv.structure_rules import detect_structure
    # Rules own table/arrow geometry; retain existing generic shape proposals.
    found = [item for item in detect_components(small, include_text=False, include_arrows=False) if item['type'] == 'Box']
    output = []
    for item in found:
        box = [float(value)/scale for value in item['bbox']]
        box = [min(width if i%2==0 else height, max(0,value)) for i,value in enumerate(box)]
        if bounds(box,width,height) is None:
            continue
        if any(item['type']==old['type'] and _iou(box, old['bbox']) > .85 for old in output):
            continue
        candidate = {'id':f'layout-{len(output)+1:04d}', 'type':item['type'], 'bbox':box,
                     'shape':item.get('shape','process'), 'text':'', 'source':item.get('source','opencv-proposal')}
        if item['type']=='Arrow':
            candidate['start'] = [float(v)/scale for v in item['start']]
            candidate['end'] = [float(v)/scale for v in item['end']]
        output.append(candidate)
        if len(output) >= 80:
            break
    structures = detect_structure(image, ocr_lines, output)
    tables = structures['tables']
    # Remove generic boxes covered by a detected table (outer border / cell duplicates).
    def in_table(box):
        area = max(1, (box[2]-box[0])*(box[3]-box[1]))
        return any(max(0,min(box[2],t['bbox'][2])-max(box[0],t['bbox'][0])) * max(0,min(box[3],t['bbox'][3])-max(box[1],t['bbox'][1])) / area > .8 for t in tables)
    output = [item for item in output if not in_table(item['bbox'])]
    from src.cv.structure_rules import connect_arrows
    connect_arrows(structures['arrows'], [*output,*tables,*ocr_lines], (width,height))
    return [*output,*tables,*structures['arrows']]


def _iou(a,b):
    overlap=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return overlap/max(1,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-overlap)


def build_document(image_path, ocr, review, job_id):
    with Image.open(image_path) as image:
        width, height = image.size
    warnings = [{'code': 'visual_review', 'message': '원본과 대조하여 글자·표 셀·도형·화살표의 내용과 위치를 직접 확인하세요.'}]
    edits = {item['id']: item for item in review.get('vector_edits', [])
             if isinstance(item, dict) and isinstance(item.get('id'), str) and isinstance(item.get('text'), str)}
    # Older reviews have prose only. Keep every OCR anchor; attach conservative line suggestions.
    suggestions = [re.sub(r'^\s*(?:[-*#]+|\d+\.)\s*', '', text).strip()
                   for text in review.get('markdown', '').splitlines() if text.strip()]
    used = set()
    elements = []
    for index, line in enumerate(ocr.get('lines', [])):
        box = bounds(line.get('bbox'), width, height)
        if box is None:
            warnings.append({'code': 'invalid_ocr_bounds', 'message': f'OCR {index + 1}번 줄의 위치가 유효하지 않아 자동 배치하지 않았습니다.'})
            continue
        element_id = line.get('id') or f'ocr-text-{index + 1:04d}'
        original = str(line.get('text', ''))
        text = original
        item = edits.get(element_id)
        uncertain = False
        if item:
            text = item['text']
            uncertain = bool(item.get('uncertain', False))
        elif 'vector_edits' not in review and original:
            matches = [(SequenceMatcher(None, normalized(original), normalized(candidate)).ratio(), j, candidate)
                       for j, candidate in enumerate(suggestions) if j not in used]
            score, j, candidate = max(matches, default=(0, -1, ''))
            if score >= .78:
                text = candidate
                used.add(j)
                uncertain = text != original
        x, y, r, b = box
        glyph_width = sum(1 if ord(c) > 127 else .55 for c in text) or 1
        font = max(8, min(96, (b-y)*.75, max(8, (r-x-12)/glyph_width)))
        elements.append({'id': element_id, 'type': 'Text', 'bbox': box, 'text': text,
                         'original_text': original, 'source': 'qwen-review' if text != original else 'paddleocrvl16',
                         'style': {'font_size': round(font, 2), 'text_align': 'left', 'font_family': 'NanumGothic', 'text_color': '#18283d'},
                         'review_status': 'review' if uncertain else 'pending', 'layer_id': 'main'})
    known = {element['id'] for element in elements}
    unknown = set(edits) - known
    if unknown:
        warnings.append({'code': 'unknown_ocr_ids', 'message': '일부 Qwen 교정 항목의 OCR 위치를 찾지 못해 배치하지 않았습니다.'})
    layout = ocr.get('layout', [])
    if layout:
        warnings.append({'code':'layout_proposals','message':'표·도형·화살표는 검출 후보입니다. 잘못된 후보는 삭제하고 누락된 표·연결은 도구로 추가하세요.'})
    for index, region in enumerate([*layout, *review.get('regions', [])]):
        if not isinstance(region, dict):
            continue
        if region.get('type') == 'Table':
            cells = region.get('cells', [])
            if not isinstance(cells, list):
                continue
            candidates = [(cell.get('id') or f'qwen-table-{index+1}-cell-{j+1}', {**cell, 'type': 'Box', 'shape': 'process', 'semantic_type': 'table-cell', 'table_id': region.get('id', f'table-{index+1}'), 'source': region.get('source', 'qwen-region')})
                          for j, cell in enumerate(cells) if isinstance(cell, dict)]
        else:
            candidates = [(region.get('id') or f'qwen-region-{index+1}', region)]
        for element_id, candidate in candidates:
            box = bounds(candidate.get('bbox'), width, height)
            if box is None or candidate.get('type') not in {'Box', 'Arrow'}:
                warnings.append({'code': 'invalid_region', 'message': f'Qwen 구조 {index+1}의 좌표·유형을 확인하지 못해 자동 배치하지 않았습니다.'})
                continue
            element = {'id': element_id, 'type': candidate['type'], 'bbox': box,
                       'text': str(candidate.get('text', '')), 'source': candidate.get('source', 'qwen-region'),
                       'layer_id': 'main', 'review_status': 'pending',
                       'style': {'stroke': '#5268e8', 'fill': '#ffffff', 'font_size': 20, 'font_family': 'NanumGothic'}}
            if candidate['type'] == 'Arrow':
                points = [candidate.get('start'), candidate.get('end')]
                if any(not isinstance(p, list) or len(p) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in p) or not (0 <= p[0] <= width and 0 <= p[1] <= height) for p in points):
                    warnings.append({'code': 'invalid_arrow', 'message': f'Qwen 화살표 {index+1}의 끝점을 확인해 주세요.'})
                    continue
                element.update(start=points[0], end=points[1], route='straight')
                if candidate.get('source') == 'structure-rules':
                    element.update(direction=candidate.get('direction'), source_id=candidate.get('source_id'), target_id=candidate.get('target_id'), connection_status=candidate.get('connection_status'))
                    if candidate.get('direction') == 'bidirectional':
                        element['style'].update(marker_start='arrow', marker_end='arrow')
            else:
                shape = candidate.get('shape', 'process')
                element['shape'] = shape if shape in {'process', 'decision', 'terminator', 'input_output', 'connector'} else 'process'
                if candidate.get('semantic_type') == 'table-cell':
                    element.update(semantic_type='table-cell', table_id=candidate['table_id'])
                    for key in ('row', 'col', 'rowspan', 'colspan', 'text_ids'):
                        if key in candidate: element[key] = candidate[key]
                    # Cell owns its text; remove OCR copies whose centers fall inside the cell.
                    x, y, r, b = box
                    contained = [e for e in elements if e['type'] == 'Text' and (e['id'] in candidate.get('text_ids', []) if candidate.get('source') == 'structure-rules' else x <= (e['bbox'][0]+e['bbox'][2])/2 <= r and y <= (e['bbox'][1]+e['bbox'][3])/2 <= b)]
                    if not element['text'] or candidate.get('source') == 'structure-rules':
                        element['text'] = '\n'.join(e['text'] for e in contained)
                    elements = [e for e in elements if e not in contained]
            if candidate in layout:
                elements.insert(0, element)
            else:
                elements.append(element)
    from src.cv.structure_rules import connect_arrows
    rule_arrows = [e for e in elements if e['type'] == 'Arrow' and e.get('source') == 'structure-rules']
    connect_arrows(rule_arrows, [e for e in elements if e['type'] != 'Arrow'], (width,height))
    relations = []
    box_ids = {e['id'] for e in elements if e['type'] == 'Box'}
    for arrow in rule_arrows:
        start, end = arrow['start'], arrow['end']
        dx, dy = end[0]-start[0], end[1]-start[1]
        direction = ('right' if dx >= 0 else 'left') if abs(dx) >= abs(dy) else ('down' if dy >= 0 else 'up')
        relations.append({'id': 'relation-'+arrow['id'], 'type': 'flows_to', 'arrow_id': arrow['id'], 'source': arrow['source_id'] if arrow['source_id'] in box_ids else None, 'target': arrow['target_id'] if arrow['target_id'] in box_ids else None, 'direction': direction, 'review_status': 'review'})
    warnings.extend({'code': 'qwen_uncertain', 'message': text} for text in review.get('uncertainties', []) if isinstance(text, str))
    if 'vector_edits' not in review:
        warnings.append({'code': 'legacy_review', 'message': '이전 분석 결과입니다. OCR 좌표를 유지하며 유사한 교정 문장만 대응했습니다. 표·화살표는 도구로 추가하거나 새 분석으로 추출하세요.'})
    return {'version': '1.0', 'session_id': job_id, 'name': 'HybridOCR 시각 검수', 'revision': 0,
            'human_reviewed': False, 'image': {'width': width, 'height': height, 'source_url': f'/api/jobs/{job_id}/image'},
            'elements': elements, 'relations': relations, 'warnings': warnings,
            'layers': [{'id': 'main', 'name': '검수 요소', 'visible': True, 'locked': False}], 'active_layer': 'main',
            'canvas': {'grid': False, 'grid_size': 10, 'background': '#ffffff'}}
