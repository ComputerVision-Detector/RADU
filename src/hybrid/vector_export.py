"""Export the user's actual vector geometry, keeping the source page coordinate system."""
import re
from src.cv.exporter import to_svg


def render_vector(document, fmt):
    width, height = document['image']['width'], document['image']['height']
    svg = to_svg(document)
    # Legacy exporter crops to element bounds; HybridOCR retains the original full page.
    svg = re.sub(r'<svg\b[^>]*>', f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', svg, count=1)
    if fmt == 'svg':
        return svg.encode('utf-8')
    import cairosvg
    if fmt == 'pdf':
        return cairosvg.svg2pdf(bytestring=svg.encode('utf-8'), output_width=794, output_height=794*height/width)
    return cairosvg.svg2png(bytestring=svg.encode('utf-8'), output_width=min(2400, width), output_height=min(2400, width)*height/width)
