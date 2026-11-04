"""Structure metrics must reward correct graphs and never crash on malformed model output."""
import json
import unittest

from src.training.evaluate import aggregate, iou, score_record, structure_bucket


GOLD = {"nodes": [{"id": "n1", "bbox": [0, 0, 100, 100], "kind": "process"}, {"id": "n2", "bbox": [0, 200, 100, 300], "kind": "data"}],
        "arrows": [{"id": "a1", "bbox": [40, 100, 60, 200], "start": [50, 100], "end": [50, 200]}],
        "relations": [{"id": "r1", "source": "n1", "target": "n2", "directed": True, "arrow_id": "a1"}]}


def score(pred):
    return score_record(pred if isinstance(pred, str) else json.dumps(pred), GOLD, 1000, 1000)


class ScoreTests(unittest.TestCase):
    def test_perfect_prediction_with_different_ids(self):
        pred = {"nodes": [{"id": "x", "bbox": [0, 0, 100, 100], "kind": "process"}, {"id": "y", "bbox": [0, 200, 100, 300], "kind": "data"}],
                "arrows": [{"id": "z", "bbox": [40, 100, 60, 200], "start": [50, 100], "end": [50, 200]}],
                "relations": [{"source": "x", "target": "y"}]}
        result = score(pred)
        self.assertEqual(result["nodes"], (2, 2, 2))
        self.assertEqual(result["arrows"], (1, 1, 1))
        self.assertEqual(result["relations"], (1, 1, 1))

    def test_reversed_relation_is_wrong(self):
        pred = {"nodes": GOLD["nodes"], "arrows": [], "relations": [{"source": "n2", "target": "n1"}]}
        self.assertEqual(score(pred)["relations"], (0, 1, 1))

    def test_reversed_arrow_does_not_match(self):
        pred = {"nodes": [], "arrows": [{"id": "a", "start": [50, 200], "end": [50, 100]}], "relations": []}
        self.assertEqual(score(pred)["arrows"], (0, 1, 1))

    def test_wrong_kind_does_not_match(self):
        pred = {"nodes": [{"id": "n", "bbox": [0, 0, 100, 100], "kind": "decision"}], "arrows": [], "relations": []}
        self.assertEqual(score(pred)["nodes"], (0, 1, 2))

    def test_one_gold_cannot_match_two_predictions(self):
        node = {"bbox": [0, 0, 100, 100], "kind": "process"}
        self.assertEqual(score({"nodes": [{"id": "a", **node}, {"id": "b", **node}]})["nodes"], (1, 2, 2))

    def test_malformed_output_scores_zero_without_crashing(self):
        for text in ("not json", "[1, 2]", '{"nodes": "x", "arrows": 3, "relations": null}', '{"nodes": [{"id": "a", "bbox": [5, 5, 1, 1], "kind": "process"}, 7]}'):
            self.assertEqual(score(text)["nodes"][0], 0)
        self.assertFalse(score("not json")["parsed"])

    def test_aggregate_micro_f1(self):
        metrics = aggregate([score({"nodes": GOLD["nodes"]}), score("bad")])
        self.assertEqual(metrics["parsed"], 1)
        self.assertEqual(metrics["nodes"]["recall"], 0.5)
        self.assertEqual(metrics["nodes"]["precision"], 1.0)

    def test_iou_disjoint_and_identical(self):
        self.assertEqual(iou([0, 0, 1, 1], [2, 2, 3, 3]), 0.0)
        self.assertEqual(iou([0, 0, 2, 2], [0, 0, 2, 2]), 1.0)

    def test_structure_bucket_reads_manifest(self):
        manifests = {"fc_a": {"writer1_2b.png": {"structure_seen_in_train": False}}}
        self.assertEqual(structure_bucket("fc_a:writer1_2b", manifests), "structure_unseen")
        self.assertIsNone(structure_bucket("hdbpmn:ex00_writer0001", manifests))


if __name__ == "__main__":
    unittest.main()
