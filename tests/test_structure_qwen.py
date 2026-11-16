import json
from pathlib import Path
import tempfile
import unittest

from src.cv import structure_qwen as sq


def raw(groups):
    return {"groups": {name: {"text": text, "tasks": sq_tasks, "adapter": "a", "generated_tokens": 1, "truncated": False, "seconds": 0.1}
                       for name, (sq_tasks, text) in groups.items()}}


DIAGRAM = json.dumps({
    "nodes": [{"id": "n1", "bbox": [10, 10, 60, 40], "kind": "terminator"},
              {"id": "n2", "bbox": [10, 100, 60, 140], "kind": "decision"},
              {"id": "n3", "bbox": [200, 100, 260, 140], "kind": "process"},
              {"id": "p", "bbox": [0, 0, 300, 200], "kind": "pool"},
              {"id": "bad", "bbox": [5, 5, 5], "kind": "process"}],
    "arrows": [{"id": "a1", "bbox": [35, 40, 35, 100], "start": [35, 40], "end": [35, 100]},
               {"id": "a2", "bbox": [60, 110, 200, 130], "start": None, "end": None}],
    "relations": [{"id": "r1", "source": "n1", "target": "n3", "directed": True, "arrow_id": "a1"},
                  {"id": "r2", "source": "n2", "target": None, "directed": True, "arrow_id": "a2"}],
})


class BuildDocumentTests(unittest.TestCase):
    def test_diagram_becomes_editor_elements_and_model_relations_win(self):
        document, hints = sq.build_document(raw({"diagram": (["nodes", "arrows", "relations"], DIAGRAM)}), 300, 200)
        boxes = [e for e in document["elements"] if e["type"] == "Box"]
        self.assertEqual([e["shape"] for e in boxes], ["terminator", "decision", "process", "process"])
        self.assertIs(boxes[3]["flow_node"], False)
        self.assertEqual(hints, {"qwen-arrow-0001": ("qwen-node-0001", "qwen-node-0003")})
        flat = next(e for e in document["elements"] if e["id"] == "qwen-arrow-0001")
        self.assertEqual(flat["bbox"], [35, 40, 36, 100])
        self.assertTrue(any(w["code"] == "structure_invalid_items" for w in document["warnings"]))
        result = sq.relate(document, hints)
        edges = {r["arrow_id"]: r for r in result["relations"] if r["type"] == "flows_to"}
        # Geometry alone would end a1 at n2; the model relation says n3.
        self.assertEqual((edges["qwen-arrow-0001"]["target"], edges["qwen-arrow-0001"]["endpoint_method"]), ("qwen-node-0003", "model_relation"))
        self.assertEqual(edges["qwen-arrow-0002"]["endpoint_method"], "bbox_approximation")
        self.assertNotIn("qwen-node-0004", {edges[a][k] for a in edges for k in ("source", "target")})
        self.assertFalse(any(w.get("element_id") == "qwen-node-0004" for w in result["warnings"]))

    def test_formula_and_table_and_cut_off_json(self):
        formula = json.dumps({"formulas": [{"id": "f1", "bbox": [1, 2, 50, 20], "latex": "x^2"}, {"id": "f2", "bbox": [1, 2, 50, 20], "latex": " "}]})
        table = '{"tables": [{"id": "t1", "bbox": [0, 30, 100, 90], "rows": 1, "cols": 2, "cells": [[0, 0, 1, 1, 0, 30, 50, 90], [0, 1, 1, 1]]}, {"id": "t2", "bbox": [0, 100, 90, 150], "rows": 1, "cols": 1, "cells": [[0, 0, 1, 1]]}, {"id": "t3", "bbox": [0,'
        document, hints = sq.build_document(raw({"formula": (["formulas"], formula), "table": (["tables"], table)}), 100, 200)
        kinds = [(e["type"], e["semantic_type"]) for e in document["elements"]]
        self.assertEqual(kinds, [("Text", "formula"), ("Box", "table-cell"), ("Box", "table")])
        self.assertEqual(document["elements"][0]["latex"], "x^2")
        self.assertEqual(document["structure"]["tables"][0]["cells"], [[0, 0, 1, 1], [0, 1, 1, 1]])
        self.assertFalse(document["structure"]["metadata"]["groups"]["table"]["valid_json"])
        self.assertTrue(any(w["code"] == "structure_truncated" for w in document["warnings"]))
        document["elements"].append({"id": "t", "type": "Text", "bbox": [10, 40, 30, 60], "text": "셀"})
        result = sq.relate(document, hints)
        self.assertIn(("contains", "qwen-table-01-cell-0001", "t"), {(r["type"], r["source"], r["target"]) for r in result["relations"]})
        self.assertFalse(any(w["code"] == "isolated_element" for w in result["warnings"]))


class BindingTests(unittest.TestCase):
    def test_changed_adapter_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "configs").mkdir()
            (root / "adapter").write_text("weights")
            (root / sq.BINDING).write_text(json.dumps({"files": {"adapter": sq._sha256(root / "adapter")}}))
            sq.verify_binding(root)
            (root / "adapter").write_text("other weights")
            with self.assertRaises(RuntimeError):
                sq.verify_binding(root)

    def test_cpu_is_refused_and_modes_cover_the_three_tasks(self):
        with self.assertRaises(ValueError):
            sq.cuda_device("cpu")
        self.assertEqual(sq.cuda_device("gpu:0"), "cuda:0")
        self.assertEqual(sq.MODES["qwen-all"], ["diagram", "formula", "table"])


class WebOptionTests(unittest.TestCase):
    def test_api_rejects_unknown_model_and_cpu(self):
        from fastapi.testclient import TestClient
        from src.web.backend.app import app
        client = TestClient(app)
        self.assertEqual(client.post("/api/analyze?structure=other", content=b"x").status_code, 400)
        self.assertEqual(client.post("/api/analyze?structure=qwen-diagram&device=cpu", content=b"x").status_code, 400)

    def test_page_offers_every_mode_and_auto_mode_starts_with_qwen(self):
        html = (Path(__file__).resolve().parents[1] / "src/web/frontend/index.html").read_text(encoding="utf-8")
        for mode in sq.MODES:
            self.assertIn(f'<option value="{mode}"', html)
        self.assertIn('<option value="qwen-all" selected>', html)
        self.assertNotIn('<option value="opencv" selected>', html)


if __name__ == "__main__":
    unittest.main()
