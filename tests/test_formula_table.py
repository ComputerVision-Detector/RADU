"""Formula and table targets: grid layout, leakage rules and strict validation; no real data."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from PIL import Image

from src.training import normalize_formula_table as module
from src.training.evaluate import score_record
from src.training.prepare import build_prompt
from src.training.schema import PROMPT, validate_record


def table_record(cells, rows=2, cols=2):
    return {"id": "x", "dataset": "pubtabnet", "image": "a.png", "width": 100, "height": 100, "split": "train", "groups": ["g"], "source_revision": "r",
            "annotation_origin": "source_annotation", "tasks": ["tables"], "coverage": {"tables": "complete"},
            "target": {"tables": [{"id": "t1", "bbox": [0, 0, 100, 100], "rows": rows, "cols": cols, "cells": cells}]}}


class GridTests(unittest.TestCase):
    def test_spans_push_later_cells_along(self):
        tokens = ["<tr>", "<td", ' rowspan="2"', ">", "</td>", "<td>", "</td>", "</tr>", "<tr>", "<td>", "</td>", "</tr>"]
        cells = [{"tokens": ["a"], "bbox": [0, 0, 10, 20]}, {"tokens": []}, {"tokens": ["b"], "bbox": [12, 11, 20, 20]}]
        rows, cols, placed = module.html_table_cells(tokens, cells, 100, 100)
        self.assertEqual((rows, cols), (2, 2))
        self.assertEqual(placed, [[0, 0, 2, 1, 0, 0, 10, 20], [0, 1, 1, 1], [1, 1, 1, 1, 12, 11, 20, 20]])

    def test_voc_spanning_cell_replaces_grid_cells(self):
        xml = """<annotation><size><width>100</width><height>50</height></size>
        <object><name>table</name><bndbox><xmin>0</xmin><ymin>0</ymin><xmax>100</xmax><ymax>50</ymax></bndbox></object>
        <object><name>table row</name><bndbox><xmin>0</xmin><ymin>0</ymin><xmax>100</xmax><ymax>25</ymax></bndbox></object>
        <object><name>table row</name><bndbox><xmin>0</xmin><ymin>25</ymin><xmax>100</xmax><ymax>50</ymax></bndbox></object>
        <object><name>table column</name><bndbox><xmin>0</xmin><ymin>0</ymin><xmax>50</xmax><ymax>50</ymax></bndbox></object>
        <object><name>table column</name><bndbox><xmin>50</xmin><ymin>0</ymin><xmax>100</xmax><ymax>50</ymax></bndbox></object>
        <object><name>table spanning cell</name><bndbox><xmin>0</xmin><ymin>0</ymin><xmax>100</xmax><ymax>25</ymax></bndbox></object></annotation>"""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "a.xml"
            path.write_text(xml)
            width, height, box, rows, cols, cells = module.voc_table(path)
        self.assertEqual((width, height, box, rows, cols), (100, 50, [0, 0, 100, 50], 2, 2))
        self.assertEqual(cells, [[0, 0, 1, 2, 0, 0, 100, 25], [1, 0, 1, 1, 0, 25, 50, 50], [1, 1, 1, 1, 50, 25, 100, 50]])
        validate_record({**table_record(cells), "height": 50, "target": {"tables": [{"id": "t1", "bbox": box, "rows": rows, "cols": cols, "cells": cells}]}})


class MalformedSourceTests(unittest.TestCase):
    def test_table_with_span_past_grid_is_skipped(self):
        tokens = ["<tr>", "<td", ' rowspan="9"', ">", "</td>", "</tr>"]
        candidate = {"html": {"structure": {"tokens": tokens}, "cells": [{"tokens": []}]}, "groups": ["g"]}
        self.assertIsNone(module.build_target(candidate, "pubtabnet", None, 100, 100))


class ValidationTests(unittest.TestCase):
    def test_overlap_and_out_of_grid_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlapping"):
            validate_record(table_record([[0, 0, 1, 2], [0, 1, 1, 1]]))
        with self.assertRaisesRegex(ValueError, "outside table grid"):
            validate_record(table_record([[1, 1, 2, 1]]))

    def test_cell_text_is_not_a_target(self):
        record = table_record([[0, 0, 1, 1]])
        record["target"]["tables"][0]["text"] = "secret"
        with self.assertRaisesRegex(ValueError, "cell text"):
            validate_record(record)

    def test_diagram_prompt_is_unchanged(self):
        prompt = build_prompt({"tasks": ["arrows", "nodes", "relations"], "width": 10, "height": 20})
        self.assertEqual(prompt, PROMPT.replace("{tasks}", "arrows, nodes, relations").replace("{width}", "10").replace("{height}", "20"))
        self.assertIn("latex", build_prompt({"tasks": ["formulas"], "width": 10, "height": 20}))


class BuildTests(unittest.TestCase):
    def candidates(self, base):
        for split, name, group in (("train", "a", "w1"), ("test", "b", "w1"), ("train", "c", "w2"), ("validation", "d", "w3"), ("test", "e", "w4")):
            yield {"split": split, "id": name, "image": f"{name}.png", "latex": name, "group": group}

    def test_group_stays_in_strictest_split_and_duplicates_drop(self):
        target = lambda candidate, dataset, base, width, height: (["formulas"], {"formulas": [{"id": "f1", "bbox": [0, 0, width, height], "latex": candidate["latex"]}]}, [candidate["group"]])
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            for index, name in enumerate("abcd"):
                Image.new("L", (8, 8), index * 40).save(base / f"{name}.png")
            Image.new("L", (8, 8), 120).save(base / "e.png")  # Same pixels as d, which is less strict.
            with mock.patch.dict(module.SOURCES, {"unimer": self.candidates}), mock.patch.object(module, "build_target", target):
                rows, taken, skipped, _ = module.build("unimer", base, {"train": 5, "validation": 5, "test": 5})
        splits = {row["id"]: row["split"] for row in rows}
        self.assertEqual(splits, {"b": "test", "e": "test", "c": "train"})
        self.assertEqual(skipped, {"duplicate_image": 1, "group_in_stricter_split": 1})

    def test_ink_renders_with_strokes(self):
        ink = '<ink><trace id="0">0 0 0, 10 10 1</trace><trace id="1">5 5 2</trace></ink>'
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "sub/a.png"
            module.render_ink(ink, output)
            with Image.open(output) as image:
                self.assertLess(image.getextrema()[0], 255)


class ScoreTests(unittest.TestCase):
    def test_formula_match_ignores_spacing_only(self):
        gold = {"formulas": [{"id": "f1", "bbox": [0, 0, 9, 9], "latex": "x ^ { 2 }"}]}
        self.assertEqual(score_record(json.dumps({"formulas": [{"latex": "x^{2}"}]}), gold, 10, 10)["formulas"], (1, 1, 1))
        self.assertEqual(score_record(json.dumps({"formulas": [{"latex": "x^{3}"}]}), gold, 10, 10)["formulas"], (0, 1, 1))

    def test_table_scores_cell_grid_and_survives_bad_cells(self):
        gold = {"tables": [{"id": "t1", "bbox": [0, 0, 9, 9], "rows": 1, "cols": 2, "cells": [[0, 0, 1, 1, 0, 0, 4, 9], [0, 1, 1, 1]]}]}
        pred = {"tables": [{"cells": [[0, 0, 1, 1], [0, 5, 1, 1], "bad", [1]]}, 3]}
        self.assertEqual(score_record(json.dumps(pred), gold, 10, 10)["tables"], (1, 2, 2))


if __name__ == "__main__":
    unittest.main()
