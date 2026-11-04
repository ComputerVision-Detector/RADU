"""Safety-critical preparation/masking checks; no model/network/GPU required."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from src.training.schema import validate_record
from src.training.prepare import build_prompt, prepare
from src.training.import_coco import convert
from src.training.train import language_lora_targets, mask_labels, source_balanced_weights, training_schedule


def record(dataset="fc_a", split="train"):
    return {"id":"1", "dataset":dataset, "image":"sample.png", "width":100, "height":80,
            "source_revision":"fixture-v1", "annotation_origin":"source_annotation",
            "split":split, "groups":[f"{dataset}:writer:{split}"],
            "tasks":["nodes"], "coverage":{"nodes":"complete"},
            "target":{"nodes":[{"id":"n1", "bbox":[10,10,30,30], "kind":"process"}]}}


class ContractTests(unittest.TestCase):
    def test_all_six_sources_use_same_contract(self):
        for source in ("hdbpmn","fc_a","ai2d_rst","didi","casia_ohfc","flowlearn"):
            self.assertEqual(validate_record(record(source))["dataset"], source)

    def test_unknown_annotations_are_not_negative_labels(self):
        row = record()
        row["target"]["relations"] = []
        with self.assertRaisesRegex(ValueError, "supervised"):
            validate_record(row)

    def test_incomplete_coverage_rejected(self):
        row = record()
        row["coverage"]["nodes"] = "partial"
        with self.assertRaisesRegex(ValueError, "complete"):
            validate_record(row)

    def test_dangling_relation_and_arrow_reference_rejected(self):
        row = record()
        row["tasks"].append("relations")
        row["coverage"]["relations"] = "complete"
        row["target"]["relations"] = [{"id":"e1", "source":"n1", "target":"missing", "directed":True, "arrow_id":None}]
        with self.assertRaisesRegex(ValueError, "dangling"):
            validate_record(row)
        row["target"]["relations"][0].update(target=None, arrow_id="missing")
        with self.assertRaisesRegex(ValueError, "arrow reference"):
            validate_record(row)

    def test_no_transcription_in_output(self):
        row = record()
        row["target"]["nodes"][0]["text"] = "do not learn OCR"
        with self.assertRaisesRegex(ValueError, "text is not a target"):
            validate_record(row)

    def test_bad_geometry_and_duplicate_ids(self):
        row = record()
        row["target"]["nodes"][0]["bbox"][2] = 101
        with self.assertRaises(ValueError):
            validate_record(row)
        row = record()
        row["target"]["nodes"] *= 2
        with self.assertRaises(ValueError):
            validate_record(row)

    def test_prompt_literal_braces_preserved_and_image_only(self):
        prompt = build_prompt(record())
        self.assertIn("{id,bbox,kind}", prompt)
        self.assertIn("Requested keys: nodes", prompt)
        self.assertIn("100 x 80", prompt)
        self.assertNotIn("OCR", prompt)

    def test_text_links_task_rejected(self):
        row = record()
        row["tasks"] = ["text_links"]
        row["coverage"] = {"text_links":"complete"}
        row["target"] = {"text_links":[]}
        with self.assertRaisesRegex(ValueError, "invalid tasks"):
            validate_record(row)

    def test_mask_prompt_padding_and_keep_eos(self):
        self.assertEqual(mask_labels([1,2,3,9,9],[1,2],[1,1,1,1,0]),[-100,-100,3,9,-100])

    def test_mask_mismatch_and_empty_response_fail(self):
        for full, prefix in (([1,2],[1,3]),([1,2],[1,2])):
            with self.assertRaises(ValueError):
                mask_labels(full,prefix,[1,1])

    def test_schedule_warmup_and_refuses_unreachable_eval(self):
        config = {"batch_size":1,"gradient_accumulation_steps":16,"epochs":3,"eval_steps":25,"save_steps":25}
        self.assertEqual(training_schedule(503,config),(96,2))
        self.assertEqual(training_schedule(503,config,max_steps=1),(1,0))
        with self.assertRaisesRegex(ValueError,"never run"):
            training_schedule(503,{**config,"eval_steps":100})

    def test_each_source_gets_equal_sampling_share(self):
        rows = [{"dataset":"big"}]*90 + [{"dataset":"small"}]*10
        weights = source_balanced_weights(rows)
        self.assertAlmostEqual(sum(weights[:90]), sum(weights[90:]))

    def test_only_language_attention_adapted(self):
        names = ["model.visual.blocks.0.attn.q_proj","model.language_model.layers.0.self_attn.q_proj", "model.language_model.layers.0.mlp.down_proj"]
        self.assertEqual(language_lora_targets(names), [names[1]])
        with self.assertRaises(ValueError):
            language_lora_targets([names[0]])


class CocoTests(unittest.TestCase):
    def fixture(self):
        coco = {"images":[{"id":1,"width":100,"height":80}], "categories":[{"id":3,"name":"process"}],
                "annotations":[{"id":10,"image_id":1,"category_id":3,"bbox":[10,10,20,20]},
                               {"id":11,"image_id":1,"category":"arrow","bbox":[30,20,20,1],"arrow_prev":10,"arrow_next":None}]}
        row = record()
        side = {key:row[key] for key in ("image","split","groups")}
        side.update(image_id=1,tasks=["nodes","arrows","relations"],coverage={k:"complete" for k in ["nodes","arrows","relations"]})
        return coco, side, {"process":{"type":"node","kind":"process"},"arrow":{"type":"arrow"}}

    def test_category_variants_source_ids_and_no_inferred_endpoints(self):
        coco, side, mapping = self.fixture()
        row = list(convert(coco,[side],mapping,"fc_a","v1"))[0]
        self.assertEqual(row["target"]["relations"][0]["source"],"n1")
        self.assertEqual(row["target"]["relations"][0]["arrow_id"],"a1")
        self.assertEqual(row["source_ids"],{"n1":"ann-10","a1":"ann-11"})
        self.assertIsNone(row["target"]["arrows"][0]["start"])

    def test_missing_connectivity_not_silently_empty(self):
        coco, side, mapping = self.fixture()
        del coco["annotations"][1]["arrow_prev"]
        with self.assertRaisesRegex(ValueError,"missing source connectivity"):
            list(convert(coco,[side],mapping,"fc_a","v1"))

    def test_ids_namespaced_across_source_splits(self):
        coco, side, mapping = self.fixture()
        first = list(convert(coco,[side],mapping,"fc_a","v1"))[0]
        other = copy.deepcopy(side)
        other["split"] = "test"
        second = list(convert(coco,[other],mapping,"fc_a","v1"))[0]
        self.assertEqual(first["id"], "train:1")
        self.assertEqual(second["id"], "test:1")


class PrepareTests(unittest.TestCase):
    def test_source_absence_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            registry = root / "configs" / "sources.json"
            registry.write_text(json.dumps({"datasets":[{"id":"fc_a","records":"missing.jsonl","status":"pending","license":"test","source":"fixture"}]}))
            try:
                import PIL
            except ImportError:
                self.skipTest("Pillow is required for preparation integration")
            with self.assertRaisesRegex(ValueError,"missing source"):
                prepare(registry,root / "out")

    def test_prepare_group_and_content_leakage(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is required for preparation integration")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            rows = []
            for i, split in enumerate(("train","validation","test")):
                Image.new("RGB",(100,80),(i*50,0,0)).save(root / f"{split}.png")
                row = record(split=split)
                row.update(id=str(i),image=f"{split}.png")
                rows.append(row)
            data = root / "normalized.jsonl"
            registry = root / "configs" / "sources.json"
            registry.write_text(json.dumps({"datasets":[{"id":"fc_a","records":"normalized.jsonl","status":"fixture","license":"test","source":"fixture"}]}))
            def write():
                data.write_text("".join(json.dumps(row)+"\n" for row in rows),encoding="utf-8")
            write()
            report = prepare(registry, root / "good")
            self.assertEqual(report["counts"], {"train":1,"validation":1,"test":1})
            rows[1]["groups"] = rows[0]["groups"]
            write()
            with self.assertRaisesRegex(ValueError,"group leaked"):
                prepare(registry, root / "bad")
            rows[1]["groups"] = ["new-group"]
            rows[1]["image"] = rows[0]["image"]
            write()
            with self.assertRaisesRegex(ValueError,"image leaked"):
                prepare(registry, root / "bad2")


if __name__ == "__main__":
    unittest.main()
