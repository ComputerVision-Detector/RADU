import asyncio
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from PIL import Image
from starlette.requests import Request

from src.web.backend.app import _normalize_upload_image, _validate_document, analyze


def encoded_image(image, image_format):
    stream = io.BytesIO()
    image.save(stream, format=image_format)
    return stream.getvalue()


class ImageUploadTests(unittest.TestCase):
    def test_svg_is_rasterized_to_png(self):
        svg = b'<svg xmlns="http://www.w3.org/2000/svg" width="12" height="8"></svg>'
        png = encoded_image(Image.new("RGB", (12, 8), "white"), "PNG")
        converter = types.SimpleNamespace(svg2png=lambda **kwargs: png)

        with patch.dict(sys.modules, {"cairosvg": converter}):
            payload, extension = _normalize_upload_image(svg)

        self.assertEqual(extension, "png")
        with Image.open(io.BytesIO(payload)) as decoded:
            self.assertEqual(decoded.format, "PNG")
            self.assertEqual(decoded.size, (12, 8))

    def test_pdf_input_is_rejected_with_explicit_message(self):
        with self.assertRaises(HTTPException) as raised:
            _normalize_upload_image(b"%PDF-1.7")

        self.assertEqual(raised.exception.status_code, 415)
        self.assertIn("PDF 입력은 아직 지원하지 않습니다", raised.exception.detail)

    def test_svg_with_entity_declaration_is_rejected(self):
        svg = b'<!DOCTYPE svg [<!ENTITY x "expanded">]><svg>&x;</svg>'

        with self.assertRaises(HTTPException) as raised:
            _normalize_upload_image(svg)

        self.assertEqual(raised.exception.status_code, 400)

    def test_large_canvas_dimensions_are_not_rejected_by_fixed_pixel_limit(self):
        document = {
            "image": {"width": 50_000, "height": 50_000},
            "elements": [],
            "relations": [],
        }

        self.assertIs(_validate_document(document), document)

    def test_tiff_is_normalized_to_browser_friendly_png(self):
        source = encoded_image(Image.new("RGB", (12, 8), "white"), "TIFF")

        payload, extension = _normalize_upload_image(source)

        self.assertEqual(extension, "png")
        with Image.open(io.BytesIO(payload)) as decoded:
            self.assertEqual(decoded.format, "PNG")
            self.assertEqual(decoded.size, (12, 8))

    def test_gif_is_normalized_using_its_first_frame(self):
        first = Image.new("RGB", (10, 6), "white")
        second = Image.new("RGB", (10, 6), "black")
        stream = io.BytesIO()
        first.save(stream, format="GIF", save_all=True, append_images=[second])

        payload, extension = _normalize_upload_image(stream.getvalue())

        self.assertEqual(extension, "png")
        with Image.open(io.BytesIO(payload)) as decoded:
            self.assertEqual(decoded.format, "PNG")
            self.assertEqual(decoded.getpixel((0, 0)), (255, 255, 255))

    def test_avif_and_heif_uploads_normalize_to_png(self):
        registered = set(Image.registered_extensions().values())
        for image_format in ("AVIF", "HEIF"):
            with self.subTest(image_format=image_format):
                if image_format not in registered:
                    self.fail(f"{image_format} decoder dependency is not installed")
                source = encoded_image(Image.new("RGB", (12, 8), "white"), image_format)

                payload, extension = _normalize_upload_image(source)

                self.assertEqual(extension, "png")
                with Image.open(io.BytesIO(payload)) as decoded:
                    self.assertEqual(decoded.format, "PNG")
                    self.assertEqual(decoded.size, (12, 8))

    def test_unrecognized_bytes_return_a_clear_client_error(self):
        with self.assertRaises(HTTPException) as raised:
            _normalize_upload_image(b"not an image")

        self.assertEqual(raised.exception.status_code, 400)
        self.assertIn("읽을 수 없는 이미지", raised.exception.detail)

    def test_analysis_endpoint_passes_normalized_tiff_to_worker(self):
        source = encoded_image(Image.new("RGB", (12, 8), "white"), "TIFF")

        async def run():
            async def receive():
                return {"type": "http.request", "body": source, "more_body": False}

            request = Request({
                "type": "http",
                "method": "POST",
                "path": "/api/analyze",
                "headers": [
                    (b"content-type", b"image/tiff"),
                    (b"content-length", str(len(source)).encode()),
                    (b"x-file-name", b"diagram.tiff"),
                ],
            }, receive)
            with tempfile.TemporaryDirectory() as directory:
                worker = AsyncMock(return_value={"elements": [], "relations": []})
                with patch("src.web.backend.app.OUT", Path(directory)), patch(
                    "src.web.backend.app._run_analysis_worker", worker
                ):
                    result = await analyze(request, ocr_engine="none", device="cpu")
                    source_path = worker.await_args.args[0]
                    self.assertTrue(source_path.name.endswith(".source.png"))
                    with Image.open(source_path) as decoded:
                        self.assertEqual(decoded.format, "PNG")
                    self.assertEqual(result["elements"], [])

        asyncio.run(run())

    def test_analysis_endpoint_accepts_body_larger_than_previous_upload_limit(self):
        source = encoded_image(Image.new("RGB", (2, 2), "white"), "PNG")
        source += b"\0" * (20 * 1024 * 1024 + 1)

        async def run():
            async def receive():
                return {"type": "http.request", "body": source, "more_body": False}

            request = Request({
                "type": "http",
                "method": "POST",
                "path": "/api/analyze",
                "headers": [
                    (b"content-type", b"image/png"),
                    (b"content-length", str(len(source)).encode()),
                    (b"x-file-name", b"large.png"),
                ],
            }, receive)
            with tempfile.TemporaryDirectory() as directory:
                worker = AsyncMock(return_value={"elements": [], "relations": []})
                with patch("src.web.backend.app.OUT", Path(directory)), patch(
                    "src.web.backend.app._run_analysis_worker", worker
                ):
                    result = await analyze(request, ocr_engine="none", device="cpu")
                    source_path = worker.await_args.args[0]
                    self.assertGreater(source_path.stat().st_size, 20 * 1024 * 1024)
                    self.assertEqual(result["elements"], [])

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
