"""Formula and table targets: grid layout, leakage rules and strict validation; no real data."""
from collections import Counter
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
from src.training.train import quota_indices


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


class NewSourceTargetTests(unittest.TestCase):
    def test_large_page_keeps_grid_without_boxes_and_oversize_page_is_dropped(self):
        cell = lambda row: (row, 0, 1, 1, [0, row, 10, row + 1])
        small = module.table_target([([0, 0, 10, 200], [cell(0)])], 100, 400, ["g"])
        self.assertEqual(small[1]["tables"][0]["cells"], [[0, 0, 1, 1, 0, 0, 10, 1]])
        large = module.table_target([([0, 0, 10, 200], [cell(r) for r in range(module.MAX_CELLS + 1)])], 100, 400, ["g"])
        self.assertEqual(large[1]["tables"][0]["cells"][0], [0, 0, 1, 1])
        self.assertIsNone(module.table_target([([0, 0, 10, 200], [cell(r) for r in range(module.MAX_GRID_CELLS + 1)])], 100, 400, ["g"]))

    def test_table_with_overlapping_source_cells_is_dropped(self):
        cells = [(0, 0, 1, 2, [0, 0, 5, 5]), (0, 1, 1, 1, [5, 0, 9, 5])]
        self.assertIsNone(module.table_target([([0, 0, 9, 9], cells)], 10, 10, ["g"]))

    def test_tabrecset_labels_count_from_one_and_drop_text(self):
        shape = lambda label, group, x: {"label": label, "group_id": group, "points": [[x, 0], [x + 5, 0], [x + 5, 5], [x, 5]]}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "a.json"
            path.write_text(json.dumps({"imageWidth": 20, "imageHeight": 10, "shapes": [shape("1-2-1-1-a-b", 0, 5), shape("1-1-1-1-c", 0, 0)]}))
            tasks, target, _ = module.tabrecset_target({"annotation": str(path), "groups": ["g"]}, 20, 10)
        self.assertEqual(tasks, ["tables"])
        self.assertEqual(target["tables"][0]["cells"], [[0, 0, 1, 1, 0, 0, 5, 5], [0, 1, 1, 1, 5, 0, 10, 5]])
        self.assertNotIn("a-b", json.dumps(target))

    def test_ibem_percent_boxes_become_pixels_and_sort_by_reading_order(self):
        page = {"embedded": {"0": {"x_min": 50, "y_min": 50, "x_max": 60, "y_max": 60, "latex_norm": "$ y $"}}, "isolated": {"0": {"x_min": 10, "y_min": 10, "x_max": 20, "y_max": 20, "latex_norm": "$ \\begin{equation*} x \\end{equation*} $"}}}
        _, target, _ = module.ibem_target({"page": page, "groups": ["g"]}, 200, 100)
        self.assertEqual(target["formulas"], [{"id": "f1", "bbox": [20, 10, 40, 20], "latex": "x"}, {"id": "f2", "bbox": [100, 50, 120, 60], "latex": "y"}])

    def test_ibem_page_with_broken_formula_is_dropped_and_align_stays_math(self):
        page = {"embedded": {"0": {"x_min": 1, "y_min": 1, "x_max": 2, "y_max": 2, "split": "double", "latex_norm": "$ x $"}}}
        self.assertIsNone(module.ibem_target({"page": page, "groups": ["g"]}, 100, 100))
        self.assertEqual(module.ibem_latex("\\begin{align*} a & b \\end{align*}"), "\\begin{aligned} a & b \\end{aligned}")

    def test_table_with_omitted_empty_cells_is_dropped(self):
        cells = [(0, 0, 1, 1, [0, 0, 5, 5]), (1, 1, 1, 1, [5, 5, 9, 9])]
        self.assertIsNone(module.table_target([([0, 0, 9, 9], cells)], 10, 10, ["g"]))

    def test_hme100k_skips_chinese_and_numbered_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            (base / "source").mkdir()
            (base / "source/train.txt").write_text("images/0.png\tx + 1\nimages/1.png\t5 、 x\nimages/2.png\t解 x\n", encoding="utf-8")
            self.assertEqual([row["id"] for row in module.hme100k_candidates(base)], ["0"])

    def test_same_image_in_other_source_is_skipped(self):
        target = lambda candidate, dataset, base, width, height: (["formulas"], {"formulas": [{"id": "f1", "bbox": [0, 0, width, height], "latex": "x"}]}, [candidate["id"]])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "hme100k").mkdir()
            (root / "unimer/source/UniMER-Test/hwe").mkdir(parents=True)
            Image.new("L", (8, 8), 1).save(root / "hme100k/a.png")
            Image.new("L", (8, 8), 2).save(root / "hme100k/b.png")
            Image.new("L", (8, 8), 1).save(root / "unimer/source/UniMER-Test/hwe/z.png")
            pool = lambda base: [{"split": "train", "id": name, "image": f"{name}.png"} for name in "ab"]
            with mock.patch.dict(module.SOURCES, {"hme100k": pool}), mock.patch.object(module, "build_target", target):
                rows, _, skipped, _ = module.build("hme100k", root / "hme100k", {"train": 5, "validation": 5, "test": 5})
        self.assertEqual([row["id"] for row in rows], ["b"])
        self.assertEqual(skipped, {"same_image_in_other_source": 1})

    def test_hdflowmind_keeps_boxes_and_never_reads_text_objects(self):
        obj = lambda name, box: f"<object><name>{name}</name><bndbox><xmin>{box[0]}</xmin><ymin>{box[1]}</ymin><xmax>{box[2]}</xmax><ymax>{box[3]}</ymax></bndbox></object>"
        xml = f"<annotation><size><width>100</width><height>50</height></size>{obj('rectangle', (1, 1, 9, 9))}{obj('text', (2, 2, 4, 4))}{obj('arrow', (9, 5, 20, 6))}<object><name>point_to</name><point><x>1</x><y>1</y></point></object></annotation>"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "a.xml"
            path.write_text(xml)
            tasks, target, _ = module.hdflowmind_target({"xml": str(path), "groups": ["g"]}, 100, 50)
        self.assertEqual(sorted(tasks), ["arrows", "nodes"])
        self.assertEqual(target["nodes"], [{"id": "n1", "bbox": [1, 1, 9, 9], "kind": "rectangle"}])
        self.assertEqual(target["arrows"], [{"id": "a1", "bbox": [9, 5, 20, 6], "start": None, "end": None}])

    def test_document_pages_share_one_group(self):
        self.assertEqual(module.tabrecset_group("ia_100000015112_7"), module.tabrecset_group("ia_100000015112_6"))
        self.assertEqual(module.tabrecset_group("2_110b10422f8ab0cb43ab4311f513aea5_segR__tR__yf"), "110b10422f8ab0cb43ab4311f513aea5")


class SamplerTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{"dataset": "big"} for _ in range(1000)] + [{"dataset": "small"} for _ in range(10)]

    def test_large_source_never_repeats_and_small_source_repeats_evenly(self):
        drawn = list(quota_indices(self.rows, {"big": 0.5, "small": 0.5}, 800))
        big = [i for i in drawn if i < 1000]
        small = Counter(i for i in drawn if i >= 1000)
        self.assertEqual(len(big), len(set(big)))
        self.assertAlmostEqual(len(big) / 800, 0.5, delta=0.07)
        self.assertLessEqual(max(small.values()) - min(small.values()), 1)

    def test_order_is_reproducible_for_resume(self):
        self.assertEqual(list(quota_indices(self.rows, {"big": 0.5, "small": 0.5}, 50, 7)), list(quota_indices(self.rows, {"big": 0.5, "small": 0.5}, 50, 7)))

    def test_missing_share_is_refused(self):
        with self.assertRaisesRegex(ValueError, "small"):
            list(quota_indices(self.rows, {"big": 1.0}, 1))


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
    def test_box_only_arrow_scores_by_box_and_arrow_without_box_or_ends_is_ignored(self):
        gold = {"arrows": [{"id": "a1", "bbox": [0, 0, 10, 10], "start": None, "end": None}]}
        good = {"arrows": [{"id": "a1", "bbox": [0, 0, 10, 10], "start": None, "end": None}, {"id": "a2", "start": None, "end": None}]}
        self.assertEqual(score_record(json.dumps(good), gold, 100, 100)["arrows"], (1, 1, 1))
        far = {"arrows": [{"id": "a1", "bbox": [50, 50, 60, 60], "start": [50, 50], "end": [60, 60]}]}
        self.assertEqual(score_record(json.dumps(far), gold, 100, 100)["arrows"], (0, 1, 1))


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
