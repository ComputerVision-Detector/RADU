import unittest

from src.web.backend import local_dataset as ld

HAS_DATA = all((ld.PREPARED / f"{split}.jsonl").is_file() for split in ld.SPLITS)


@unittest.skipUnless(HAS_DATA, "prepared RADU splits are not on this machine")
class LocalDatasetTests(unittest.TestCase):
    def test_sources_and_samples_come_from_validation_and_test_only(self):
        listing = ld.list_sources()
        self.assertEqual(len(listing["sources"]), 16)
        self.assertEqual({item["kind"] for item in listing["sources"]}, {"diagram", "formula", "table"})
        samples = ld.list_samples("fc_a", "validation")
        self.assertEqual(len(samples), next(s for s in listing["sources"] if s["id"] == "fc_a")["splits"]["validation"])
        self.assertTrue(ld.image_path(samples[0]["id"]).is_file())
        with self.assertRaises(KeyError):
            ld.list_samples("fc_a", "train")

    def test_api_serves_listed_images_only(self):
        from fastapi.testclient import TestClient
        from src.web.backend.app import app
        client = TestClient(app)
        sample = client.get("/api/local-datasets/ibem/test").json()["samples"][0]
        image = client.get("/api/local-datasets/image", params={"id": sample["id"]})
        self.assertEqual((image.status_code, image.headers["content-type"].split("/")[0]), (200, "image"))
        self.assertEqual(client.get("/api/local-datasets/image", params={"id": "../../etc/passwd"}).status_code, 404)
        self.assertEqual(client.get("/api/local-datasets/nope/test").status_code, 404)


if __name__ == "__main__":
    unittest.main()
