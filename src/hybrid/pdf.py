"""Korean font embedded PDF with editable Markdown tables rendered as tables."""
import io
import os
import re
from pathlib import Path
from xml.sax.saxutils import escape


def render_pdf(markdown: str) -> bytes:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    font = os.getenv('HYBRIDOCR_FONT', '/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
    if not Path(font).is_file():
        raise RuntimeError('한글 TTF 폰트 경로를 HYBRIDOCR_FONT에 설정하세요.')
    if 'HybridKorean' not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont('HybridKorean', font))
    style = ParagraphStyle('body', fontName='HybridKorean', fontSize=10, leading=16, wordWrap='CJK')
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    story = []
    lines = markdown.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.strip().startswith('|') and line.strip().endswith('|'):
            rows = []
            while index < len(lines) and lines[index].strip().startswith('|') and lines[index].strip().endswith('|'):
                cells = lines[index].strip().strip('|').split('|')
                if not all(re.fullmatch(r'\s*:?-+:?\s*', cell) for cell in cells):
                    rows.append([Paragraph(escape(cell.strip()), style) for cell in cells])
                index += 1
            count = max(map(len, rows), default=0)
            if count:
                for row in rows:
                    row.extend([''] * (count-len(row)))
                table = Table(rows, colWidths=[doc.width/count]*count, repeatRows=1, splitInRow=1)
                table.setStyle(TableStyle([('GRID',(0,0),(-1,-1),.5,colors.grey),
                    ('VALIGN',(0,0),(-1,-1),'TOP'), ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8edf5'))]))
                story.extend([table, Spacer(1, 10)])
            continue
        text = re.sub(r'^#{1,6}\s+', '', line)
        story.append(Paragraph(escape(text), style) if text else Spacer(1, 8))
        index += 1
    if not story:
        story.append(Spacer(1, 1))
    doc.build(story)
    return buffer.getvalue()
