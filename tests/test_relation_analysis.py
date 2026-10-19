import asyncio
import unittest

from src.web.backend.app import reanalyze_relations


def manual_flowchart():
    return {
        "version": "1.0",
        "image": {"width": 400, "height": 200},
        "elements": [
            {"id": "start", "type": "Box", "shape": "terminator", "bbox": [20, 50, 100, 100]},
            {"id": "finish", "type": "Box", "shape": "process", "bbox": [260, 50, 340, 100]},
            {
                "id": "arrow-1",
                "type": "Arrow",
                "bbox": [99, 74, 261, 76],
                "start": {"x": 100, "y": 75},
                "end": {"x": 260, "y": 75},
            },
        ],
        "relations": [],
        "warnings": [
            {
                "code": "manual_arrow_unlinked",
                "element_id": "arrow-1",
                "message": "Unlinked until analyzed",
            }
        ],
    }


class RelationAnalysisTests(unittest.TestCase):
    def analyze(self, document):
        return asyncio.run(reanalyze_relations(document))

    def test_manual_arrow_endpoints_are_analyzed_and_reanalysis_is_idempotent(self):
        first = self.analyze(manual_flowchart())
        second = self.analyze(first)

        self.assertEqual(len(first["relations"]), 1)
        self.assertEqual(first["relations"][0]["source"], "start")
        self.assertEqual(first["relations"][0]["target"], "finish")
        self.assertEqual(first["relations"][0]["endpoint_method"], "explicit")
        self.assertNotIn(
            "manual_arrow_unlinked",
            {warning["code"] for warning in first["warnings"]},
        )
        self.assertEqual(first["warnings"], second["warnings"])

    def test_reanalysis_preserves_manual_relation_review_metadata(self):
        document = manual_flowchart()
        document["relations"] = [
            {
                "id": "user-edge",
                "type": "flows_to",
                "source": "start",
                "target": "finish",
                "arrow_id": "arrow-1",
                "review_status": "confirmed",
                "label": "완료",
            }
        ]

        result = self.analyze(document)
        relation = result["relations"][0]

        self.assertEqual(relation["id"], "user-edge")
        self.assertEqual(relation["review_status"], "confirmed")
        self.assertEqual(relation["label"], "완료")

    def test_image_detector_array_endpoints_remain_supported(self):
        document = manual_flowchart()
        arrow = document["elements"][2]
        arrow["start"] = [100, 75]
        arrow["end"] = [260, 75]

        result = self.analyze(document)

        self.assertEqual(result["relations"][0]["source"], "start")
        self.assertEqual(result["relations"][0]["target"], "finish")

    def test_boundary_attachment_beats_a_nearer_small_box_center(self):
        document = manual_flowchart()
        document["elements"][0]["bbox"] = [0, 0, 100, 100]
        document["elements"].insert(
            1,
            {"id": "nearby", "type": "Box", "shape": "process", "bbox": [101, 45, 105, 55]},
        )
        arrow = document["elements"][-1]
        arrow["start"] = {"x": 100, "y": 50}
        arrow["end"] = {"x": 260, "y": 75}

        result = self.analyze(document)

        self.assertEqual(result["relations"][0]["source"], "start")


if __name__ == "__main__":
    unittest.main()
