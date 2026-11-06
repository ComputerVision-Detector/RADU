"""The page must show what truncated output contains and never trust model text."""
import json
import unittest

from src.training.visualize_predictions import elements_svg, salvage_elements


class SalvageTests(unittest.TestCase):
    def test_valid_json_is_not_marked_truncated(self):
        nodes, arrows, relations, truncated = salvage_elements(json.dumps({"nodes": [{"id": "a"}], "arrows": [], "relations": []}))
        self.assertEqual((nodes, arrows, relations, truncated), ([{"id": "a"}], [], [], False))

    def test_truncated_output_keeps_complete_elements(self):
        text = '{"arrows": [{"id": "a0", "start": [1, 2], "end": [3, 4]}, {"id": "a1", "start": [5, '
        nodes, arrows, _, truncated = salvage_elements(text)
        self.assertEqual(nodes, [])
        self.assertEqual([arrow["id"] for arrow in arrows], ["a0"])
        self.assertTrue(truncated)

    def test_non_object_output_is_empty(self):
        self.assertEqual(salvage_elements("[1, 2]"), ([], [], [], True))
        self.assertEqual(salvage_elements("not json"), ([], [], [], True))

    def test_non_list_values_are_ignored(self):
        self.assertEqual(salvage_elements('{"nodes": "x", "arrows": 3}')[:2], [[], []])


class SvgTests(unittest.TestCase):
    def test_model_text_is_escaped(self):
        svg = elements_svg([{"bbox": [0, 0, 5, 5], "kind": "<script>alert(1)</script>"}], [], 10, 10)
        self.assertNotIn("<script>", svg)

    def test_bad_geometry_and_types_are_skipped(self):
        svg = elements_svg([{"bbox": [5, 5, 1, 1]}, 7], [{"start": "x", "end": [1, 2]}, None], 10, 10)
        self.assertNotIn("<rect", svg)
        self.assertNotIn("<line", svg)


if __name__ == "__main__":
    unittest.main()
