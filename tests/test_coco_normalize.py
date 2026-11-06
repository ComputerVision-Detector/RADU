"""Split-manifest leakage rules and COCO-to-normalized joining; no GPU or real data."""
import unittest

from src.training.build_coco_splits import FC_A_FAMILY_ALIASES, build_cmp, build_fc_a, build_hdbpmn, verify_disjoint
from src.training.normalize_coco import arrow_geometry, normalized_rows


def fc_a_images():
    images = {}
    for writer in range(1, 21):
        for family in range(1, 15):
            images[f"writer{writer}_{family}.png"] = f"hash-{writer}-{family}"
            images[f"writer{writer}_{family}b.png"] = f"hash-{writer}-{family}b"
    return images


class SplitTests(unittest.TestCase):
    def test_fc_a_writers_never_cross_splits_and_is_deterministic(self):
        manifest = verify_disjoint(build_fc_a(fc_a_images()))
        self.assertEqual(manifest, build_fc_a(fc_a_images()))
        self.assertEqual({s["split"] for s in manifest["samples"].values()}, {"train", "validation", "test"})

    def test_fc_a_variants_share_a_family_and_held_families_are_test_only(self):
        manifest = build_fc_a(fc_a_images())
        held = set(manifest["held_out_families"])
        self.assertEqual(len(held), 3)
        for sample in manifest["samples"].values():
            if sample["family"] in held:
                self.assertEqual(sample["split"], "test")
                self.assertFalse(sample["structure_seen_in_train"])
            template = sample["variant"].rstrip("b")
            self.assertEqual(FC_A_FAMILY_ALIASES.get(template, template), sample["family"])
        self.assertTrue(any(v == "held_out_family_outside_test_writers" for v in manifest["excluded"].values()))

    def test_same_structure_templates_cannot_cross_held_boundary(self):
        manifest = build_fc_a(fc_a_images())
        families = {s["family"] for s in manifest["samples"].values()}
        self.assertNotIn("1", families)
        self.assertEqual(manifest["excluded"]["writer8_1.png"], "near_duplicate_of:writer8_6.png")

    def test_exact_duplicates_are_excluded_with_reason(self):
        images = fc_a_images()
        images["writer1_6_.png"] = images["writer1_6.png"]
        manifest = build_fc_a(images)
        self.assertNotIn("writer1_6_.png", manifest["samples"])
        self.assertEqual(manifest["excluded"]["writer1_6_.png"], "exact_duplicate_of:writer1_6.png")

    def test_cross_split_group_is_rejected(self):
        manifest = {"dataset": "hdbpmn", "samples": {"a": {"split": "train", "groups": ["g"]}, "b": {"split": "test", "groups": ["g"]}}}
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            verify_disjoint(manifest)

    def test_hdbpmn_keeps_official_partition(self):
        manifest = build_hdbpmn({"ex00_writer0001.jpg": ("val", "h1"), "ex01_writer0001.jpg": ("val", "h1"), "ex00_writer0002.jpg": ("test", "h2")})
        self.assertEqual(manifest["samples"]["ex00_writer0001.jpg"]["split"], "validation")
        self.assertEqual(manifest["samples"]["ex00_writer0002.jpg"]["split"], "test")
        self.assertIn("ex01_writer0001.jpg", manifest["excluded"])


class CmpSplitTests(unittest.TestCase):
    PARTITIONS = {"writer000_fc_001.png": ("train", "h1"), "writer000_fc_001b.png": ("train", "h2"),
                  "writer001_fc_001.png": ("val", "h3"), "writer002_fc_002.png": ("test", "h4")}

    def test_scan_shares_writer_groups_with_its_source(self):
        original, scan = build_cmp("fc_b", self.PARTITIONS), build_cmp("fc_b_scan", self.PARTITIONS)
        name = "writer001_fc_001.png"
        self.assertEqual(original["samples"][name]["groups"], scan["samples"][name]["groups"])
        self.assertEqual(scan["samples"][name]["split"], "validation")
        self.assertEqual(scan["version"], "fc_b_scan_v1")
        verify_disjoint(scan)

    def test_fa_validation_omits_relations_only(self):
        manifest = build_cmp("fa", {"writer000_fa_001.png": ("train", "h1"), "writer001_fa_001.png": ("val", "h2")})
        self.assertNotIn("tasks", manifest["samples"]["writer000_fa_001.png"])
        self.assertEqual(manifest["samples"]["writer001_fa_001.png"]["tasks"], ["nodes", "arrows"])

    def test_unknown_name_is_rejected(self):
        with self.assertRaises(ValueError):
            build_cmp("fc_b", {"other.png": ("train", "h1")})


class FaConnectivityTests(unittest.TestCase):
    CATEGORY_MAP = {"state": {"type": "node", "kind": "state"}, "arrow": {"type": "arrow"}}

    def document(self, arrow):
        return {"categories": [], "images": [{"id": 0, "file_name": "writer000_fa_001.png", "width": 100, "height": 80}],
                "annotations": [{"id": 0, "image_id": 0, "category": "state", "bbox": [50, 50, 10, 10], "keypoints": [0, 0, 0, 0, 0, 0]},
                                {"id": 1, "image_id": 0, "category": "arrow", "bbox": [1, 1, 49, 49], "keypoints": [2, 2, 2, 50, 50, 2], **arrow}]}

    def rows(self, arrow, sample):
        manifest = {"samples": {"writer000_fa_001.png": {"split": "train", "groups": ["fa:writer:000"], **sample}}}
        return list(normalized_rows(self.document(arrow), "train", manifest, "source", self.CATEGORY_MAP, "fa", "rev"))

    def test_initial_arrow_has_open_source(self):
        relation = self.rows({"arrow_next": 0}, {})[0]["target"]["relations"][0]
        self.assertIsNone(relation["source"])
        self.assertEqual(relation["target"], "n1")

    def test_validation_arrows_keep_geometry_without_relations(self):
        row = self.rows({}, {"tasks": ["nodes", "arrows"]})[0]
        self.assertEqual(row["target"]["arrows"][0]["start"], [2, 2])
        self.assertNotIn("relations", row["target"])

    def test_missing_target_still_fails(self):
        with self.assertRaisesRegex(ValueError, "missing source connectivity"):
            self.rows({"arrow_prev": 0}, {})


class NormalizeTests(unittest.TestCase):
    CATEGORY_MAP = {"process": {"type": "node", "kind": "process"}, "text": {"type": "text"}, "arrow": {"type": "arrow"}}

    def document(self):
        return {"categories": [], "images": [{"id": 0, "file_name": "writer1_1.png", "width": 100, "height": 80},
                                             {"id": 1, "file_name": "writer1_2.png", "width": 100, "height": 80}],
                "annotations": [
                    {"id": 0, "image_id": 0, "category": "process", "bbox": [1.2, 2.5, 10.1, 10], "keypoints": [0, 0, 0, 0, 0, 0]},
                    {"id": 1, "image_id": 0, "category": "process", "bbox": [50, 50, 10, 10], "keypoints": [0, 0, 0, 0, 0, 0]},
                    {"id": 2, "image_id": 0, "category": "text", "bbox": [2, 3, 5, 5], "keypoints": [0, 0, 0, 0, 0, 0], "text": "secret"},
                    {"id": 3, "image_id": 0, "category": "arrow", "bbox": [11, 12, 39, 38], "keypoints": [11.4, 12.6, 2, 50, 50, 2], "arrow_prev": 0, "arrow_next": 1},
                    {"id": 4, "image_id": 1, "category": "process", "bbox": [5, 5, 5, 5], "keypoints": [0, 0, 0, 0, 0, 0]},
                ]}

    def manifest(self):
        return {"samples": {"writer1_1.png": {"split": "train", "groups": ["fc_a:writer:1"]}}}

    def rows(self):
        return list(normalized_rows(self.document(), "train", self.manifest(), "source", self.CATEGORY_MAP, "fc_a", "rev"))

    def test_first_keypoint_is_arrow_start(self):
        geometry = arrow_geometry(self.document()["annotations"])
        self.assertEqual(geometry["3"], {"start": [11, 13], "end": [50, 50]})

    def test_excluded_images_are_dropped_and_coordinates_are_integers(self):
        rows = self.rows()
        self.assertEqual([row["id"] for row in rows], ["writer1_1"])
        self.assertEqual(rows[0]["target"]["nodes"][0]["bbox"], [1, 2, 12, 13])

    def test_relation_uses_source_connectivity_and_text_stays_out_of_target(self):
        target = self.rows()[0]["target"]
        self.assertEqual(target["relations"], [{"id": "r1", "source": "n1", "target": "n2", "directed": True, "arrow_id": "a1"}])
        self.assertNotIn("secret", str(target))

    def test_record_has_no_ocr_input(self):
        self.assertNotIn("ocr", self.rows()[0])


if __name__ == "__main__":
    unittest.main()
