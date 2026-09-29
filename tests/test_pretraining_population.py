from __future__ import annotations

import importlib
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import pandas as pd
import yaml

from src.models.ablations.run_siamese_ablations import validate_pretraining_split


PRETRAIN_MODULES = (
    "src.models.pretrain_siamese",
    "src.models.pretrain_siamese_alvadesc",
    "src.models.ablations.pretrain_siamese_wide_only",
    "src.models.ablations.pretrain_siamese_wide_only_alvadesc",
    "src.models.ablations.pretrain_siamese_deep_only",
    "src.models.ablations.pretrain_siamese_deep_only_alvadesc",
)


class PretrainingPopulationTests(unittest.TestCase):
    def test_all_objectives_and_architectures_keep_the_same_population_and_split(self):
        source = pd.DataFrame({
            "inchi": [f"molecule_{i}" for i in range(103)],
            "V1": np.ones(103, dtype=np.float32),
            "classification": ["A"] * 60 + ["B"] * 43,
        })
        # Two absent LogP values and one molecule absent from the descriptor cache.
        descriptors = pd.DataFrame({
            "inchi": source["inchi"].iloc[:-1],
            "logp": np.ones(102),
            "mol_volume_mean": np.full(102, 100.0),
        })
        descriptors.loc[[0, 60], "logp"] = np.nan
        descriptors.loc[1, "mol_volume_mean"] = np.nan
        expected = source.drop(index=[0, 60, 102])["inchi"].tolist()
        reference_splits = {}

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            descriptor_path = root / "descriptors.csv"
            descriptors.to_csv(descriptor_path, index=False)
            for module_name in PRETRAIN_MODULES:
                module = importlib.import_module(module_name)
                for use_logp, use_molvol in ((True, True), (True, False), (False, True), (False, False)):
                    with self.subTest(module=module_name, logp=use_logp, molvol=use_molvol):
                        config = yaml.safe_load(module.DEFAULT_CONFIG_PATH.read_text())
                        config["data"]["descriptor_cache_csv"] = str(descriptor_path)
                        config["data"]["random_seed"] = 42
                        config["tasks"] = {"use_logp": use_logp, "use_molvol": use_molvol}
                        output = root / f"{module_name}_{use_logp}_{use_molvol}"
                        config["training"]["results_name"] = str(output)
                        config_path = root / "config.yaml"
                        config_path.write_text(yaml.safe_dump(config))
                        trainer = module.FingerprintSiamesePretrainer(config_path)
                        loader_name = "load_hmdb_rdkit" if trainer.fingerprint_source == "rdkit" else "load_hmdb"
                        with patch.object(module, loader_name, return_value=source.copy()):
                            data = trainer.load_data()
                        self.assertEqual(data["inchi"].tolist(), expected)
                        self.assertIn("molecule_1", data["inchi"].tolist())
                        train, validation = trainer.make_or_load_split(data)
                        membership = (train["inchi"].tolist(), validation["inchi"].tolist())
                        reference = reference_splits.setdefault(trainer.fingerprint_source, membership)
                        self.assertEqual(membership, reference)
                        self.assertEqual((len(train), len(validation)), (90, 10))

    def test_ablation_split_validation_compares_membership_per_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            reference, candidate = Path(directory) / "reference", Path(directory) / "candidate"
            for root in (reference, candidate):
                (root / "split").mkdir(parents=True)
            for filename, molecules in (
                ("train_inchi.csv", ["A", "B"]),
                ("validation_inchi.csv", ["C", "D"]),
            ):
                pd.DataFrame({"inchi": molecules}).to_csv(reference / "split" / filename, index=False)
                pd.DataFrame({"inchi": molecules[::-1]}).to_csv(candidate / "split" / filename, index=False)
            validate_pretraining_split(candidate, reference)
            # The global population is unchanged, but molecules swapped partitions.
            pd.DataFrame({"inchi": ["A", "C"]}).to_csv(candidate / "split" / "train_inchi.csv", index=False)
            pd.DataFrame({"inchi": ["B", "D"]}).to_csv(candidate / "split" / "validation_inchi.csv", index=False)
            with self.assertRaisesRegex(ValueError, "HMDB split mismatch"):
                validate_pretraining_split(candidate, reference)


if __name__ == "__main__":
    unittest.main()
