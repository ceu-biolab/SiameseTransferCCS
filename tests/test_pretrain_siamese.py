from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.models.hmdb_pretraining import HMDBPretrainingMixin


class DummyHMDBPretrainer(HMDBPretrainingMixin):
    fingerprint_source = "rdkit"

    def __init__(self, results_dir):
        self.config = {
            "split": {
                "train_fraction": 0.90,
                "validation_fraction": 0.10,
                "stratify_by": "classification",
            },
            "pairs": {
                "train_pairs": 100,
                "validation_pairs": 20,
            },
        }
        self.split_cfg = self.config["split"]
        self.pairs_cfg = self.config["pairs"]
        self.train_fraction = 0.90
        self.validation_fraction = 0.10
        self.stratify_by = "classification"
        self.train_pairs = 100
        self.validation_pairs = 20
        self.random_seed = 42
        self.results_dir = results_dir
        self._validate_pretraining_config()


def make_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "inchi": [f"InChI={index}" for index in range(100)],
            "classification": ["A"] * 50 + ["B"] * 30 + ["C"] * 20,
        }
    )


class HMDBPretrainingTests(unittest.TestCase):
    def test_hmdb_split_is_disjoint_stratified_and_persisted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            results_dir = Path(temp_dir)
            trainer = DummyHMDBPretrainer(results_dir)
            source = make_dataframe()

            train_df, validation_df = trainer.make_or_load_split(source)

            self.assertEqual(len(train_df), 90)
            self.assertEqual(len(validation_df), 10)
            self.assertTrue(set(train_df["inchi"]).isdisjoint(validation_df["inchi"]))
            self.assertEqual(
                train_df["classification"].value_counts().to_dict(),
                {"A": 45, "B": 27, "C": 18},
            )
            self.assertEqual(
                validation_df["classification"].value_counts().to_dict(),
                {"A": 5, "B": 3, "C": 2},
            )

            persisted_train, persisted_validation = trainer.make_or_load_split(source)
            self.assertEqual(persisted_train["inchi"].tolist(), train_df["inchi"].tolist())
            self.assertEqual(
                persisted_validation["inchi"].tolist(),
                validation_df["inchi"].tolist(),
            )

            manifest = json.loads(
                (results_dir / "split" / "split_manifest.json").read_text()
            )
            self.assertEqual(manifest["strategy"], "stratified_hmdb_90_10")
            self.assertEqual(
                manifest["validation_usage"],
                "training_control_only_early_stopping_lr_schedule_and_checkpoint",
            )
            self.assertIsNone(manifest["test_partition"])

    def test_split_configuration_requires_complete_90_10_partition(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = DummyHMDBPretrainer(Path(temp_dir))
            trainer.validation_fraction = 0.20

            with self.assertRaisesRegex(ValueError, "must sum to 1.0"):
                trainer._validate_pretraining_config()


if __name__ == "__main__":
    unittest.main()
