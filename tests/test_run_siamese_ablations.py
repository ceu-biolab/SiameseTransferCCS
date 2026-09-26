from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from src.models.ablations.run_siamese_ablations import (
    ABLATIONS,
    PRETRAINING_PROTOCOL,
    build_job,
    run_pipeline,
)


class SiameseObjectiveAblationRunnerTests(unittest.TestCase):
    def build_test_job(self, temp_dir: str, source: str = "rdkit"):
        return build_job(
            source=source,
            ablation="tanimoto_only",
            results_prefix="Siamese_ablation",
            experiment_prefix="hmdb90_10_test",
            configs_dir=Path(temp_dir) / "configs",
            ccs_config_path=Path("configs/ccs_prediction_heads.yaml"),
            random_seed=17,
        )

    def test_generated_config_uses_hmdb_protocol_and_objective_ablation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            job = self.build_test_job(temp_dir)
            config = yaml.safe_load(job.config_path.read_text(encoding="utf-8"))
            ccs_config = yaml.safe_load(job.ccs_config_path.read_text(encoding="utf-8"))

            self.assertEqual(config["split"]["train_fraction"], 0.90)
            self.assertEqual(config["split"]["validation_fraction"], 0.10)
            self.assertEqual(config["data"]["random_seed"], 17)
            self.assertFalse(config["tasks"]["use_logp"])
            self.assertFalse(config["tasks"]["use_molvol"])
            self.assertEqual(config["loss"]["lambda_similarity"], 1.0)
            self.assertEqual(
                config["ablation"]["pretraining_protocol"],
                PRETRAINING_PROTOCOL,
            )
            self.assertEqual(
                ccs_config["ablation"]["pretraining_protocol"],
                PRETRAINING_PROTOCOL,
            )
            self.assertEqual(job.pretrain_module, "src.models.pretrain_siamese")
            self.assertEqual(job.experiment_tag, "hmdb90_10_test_tanimoto_only_mae")

    def test_runner_contains_only_the_four_objective_ablations(self):
        self.assertEqual(
            set(ABLATIONS),
            {
                "tanimoto_only",
                "tanimoto_logp",
                "tanimoto_molvol",
                "logp_molvol_no_tanimoto",
            },
        )
        self.assertNotIn("wide_only", ABLATIONS)
        self.assertNotIn("deep_only", ABLATIONS)

    def test_pretrain_command_selects_the_source_specific_module(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            job = self.build_test_job(temp_dir, source="alvadesc")
            args = argparse.Namespace(
                python="python",
                folds=5,
                random_seed=17,
                logs_dir=Path(temp_dir) / "logs",
                markers_dir=Path(temp_dir) / "markers",
                skip_existing_pretrain=False,
                skip_existing_ccs=False,
                dry_run=True,
            )

            with patch(
                "src.models.ablations.run_siamese_ablations.run_command"
            ) as run_command:
                run_pipeline(job, args)

            pretrain_command = run_command.call_args_list[0].args[0]
            ccs_command = run_command.call_args_list[1].args[0]
            self.assertIn("src.models.pretrain_siamese_alvadesc", pretrain_command)
            self.assertNotIn("--fingerprint-source", pretrain_command)
            self.assertEqual(
                ccs_command[ccs_command.index("--experiment-tag") + 1],
                job.experiment_tag,
            )


if __name__ == "__main__":
    unittest.main()
