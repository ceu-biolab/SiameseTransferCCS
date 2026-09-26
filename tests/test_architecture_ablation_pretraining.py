from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from src.models.ablations.pretrain_siamese_deep_only import (
    FingerprintSiamesePretrainer as RDKitDeepOnlyPretrainer,
)
from src.models.ablations.pretrain_siamese_deep_only_alvadesc import (
    FingerprintSiamesePretrainer as AlvaDescDeepOnlyPretrainer,
)
from src.models.ablations.pretrain_siamese_wide_only import (
    FingerprintSiamesePretrainer as RDKitWideOnlyPretrainer,
)
from src.models.ablations.pretrain_siamese_wide_only_alvadesc import (
    FingerprintSiamesePretrainer as AlvaDescWideOnlyPretrainer,
)
from src.models.ablations.run_siamese_deep_only import (
    SIAMESE_RESULTS_DIRS as DEEP_ONLY_RESULTS_DIRS,
    SiameseCCSRunner as DeepOnlyCCSRunner,
)
from src.models.ablations.run_siamese_wide_only import (
    SIAMESE_RESULTS_DIRS as WIDE_ONLY_RESULTS_DIRS,
    SiameseCCSRunner as WideOnlyCCSRunner,
)
from src.models.hmdb_pretraining import HMDBPretrainingMixin


PRETRAIN_CASES = (
    (
        RDKitWideOnlyPretrainer,
        "rdkit",
        "wide_only",
        Path("configs/pretrain_siamese_wide_only.yaml"),
    ),
    (
        AlvaDescWideOnlyPretrainer,
        "alvadesc",
        "wide_only",
        Path("configs/pretrain_siamese_wide_only_alvadesc.yaml"),
    ),
    (
        RDKitDeepOnlyPretrainer,
        "rdkit",
        "deep_only",
        Path("configs/pretrain_siamese_deep_only.yaml"),
    ),
    (
        AlvaDescDeepOnlyPretrainer,
        "alvadesc",
        "deep_only",
        Path("configs/pretrain_siamese_deep_only_alvadesc.yaml"),
    ),
)


class ArchitectureAblationPretrainingTests(unittest.TestCase):
    def test_all_architecture_pretrainers_use_the_hmdb_protocol(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            for index, (trainer_class, source, branch_mode, config_path) in enumerate(
                PRETRAIN_CASES
            ):
                with self.subTest(trainer=trainer_class.__module__):
                    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                    output_dir = Path(temp_dir) / f"case_{index}"
                    config["training"]["results_name"] = str(output_dir)
                    generated_config = Path(temp_dir) / f"config_{index}.yaml"
                    generated_config.write_text(
                        yaml.safe_dump(config, sort_keys=False),
                        encoding="utf-8",
                    )

                    trainer = trainer_class(config_path=generated_config)

                    self.assertIsInstance(trainer, HMDBPretrainingMixin)
                    self.assertEqual(trainer.fingerprint_source, source)
                    self.assertEqual(trainer.train_fraction, 0.90)
                    self.assertEqual(trainer.validation_fraction, 0.10)
                    self.assertEqual(trainer.train_pairs, 100_000)
                    self.assertEqual(trainer.validation_pairs, 10_000)
                    self.assertEqual(trainer.results_dir, output_dir)

                    trainer._write_model_manifest()
                    manifest = json.loads(
                        (output_dir / "model_manifest.json").read_text(encoding="utf-8")
                    )
                    self.assertEqual(manifest["fingerprint_source"], source)
                    self.assertEqual(manifest["branch_mode"], branch_mode)
                    self.assertEqual(
                        manifest["pretraining_protocol"],
                        "hmdb_90_10",
                    )
                    self.assertEqual(manifest["wide_dim"], 1536)
                    self.assertEqual(manifest["deep_dim"], 512)
                    self.assertEqual(manifest["embedding_dim"], 2048)

    def test_downstream_defaults_and_tags_are_protocol_specific(self):
        self.assertEqual(
            WIDE_ONLY_RESULTS_DIRS["rdkit"],
            Path("results/Siamese_physchem_wide_only"),
        )
        self.assertEqual(
            DEEP_ONLY_RESULTS_DIRS["alvadesc"],
            Path("results/Siamese_physchem_deep_only_alvadesc"),
        )

        wide_runner = WideOnlyCCSRunner(fingerprint_source="rdkit")
        deep_runner = DeepOnlyCCSRunner(fingerprint_source="alvadesc")
        self.assertIn(
            "siamesa_wide_only_hmdb90_10_v1_",
            wide_runner.result_subfolder(["ccsbase"], "ccsbase", "linear_regression"),
        )
        self.assertIn(
            "siamesa_deep_only_hmdb90_10_v1_",
            deep_runner.result_subfolder(["metlinccs"], "ccsbase", "linear_regression"),
        )


if __name__ == "__main__":
    unittest.main()
