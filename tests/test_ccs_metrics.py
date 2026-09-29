from __future__ import annotations

import csv
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np

from src.models.ablations.run_siamese_from_scratch import ScratchEndToEndRunner
from src.models.ablations.run_siamese_wide_only_dgr import WideOnlyPretrainedDGRRunner
from src.models.metrics import mspe_percent
from src.models.run_baseline import FingerprintCCSBaselineRunner
from src.models.run_siamese import SiameseCCSRunner


class CCSMetricTests(unittest.TestCase):
    def test_mspe_is_squared_relative_error_expressed_as_percent(self):
        self.assertAlmostEqual(mspe_percent([100.0], [110.0]), 1.0)
        self.assertAlmostEqual(mspe_percent([100.0, 200.0], [110.0, 160.0]), 2.5)
        self.assertEqual(mspe_percent([100.0, 200.0], [100.0, 200.0]), 0.0)

    def test_all_evaluators_write_the_same_mspe_units(self):
        arrays = SimpleNamespace(
            y_scaler=SimpleNamespace(inverse_transform=lambda values: values),
            y_test_scaled=np.array([100.0, 200.0]),
        )
        model = Mock()
        model.predict.return_value = np.array([[110.0], [180.0]])
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            for runner_class in (
                FingerprintCCSBaselineRunner, SiameseCCSRunner,
                ScratchEndToEndRunner, WideOnlyPretrainedDGRRunner,
            ):
                with self.subTest(runner=runner_class.__name__):
                    runner = runner_class.__new__(runner_class)
                    runner.model_type = "gated_residual_mlp"
                    runner.training_inputs = Mock(return_value=np.zeros((2, 1)))
                    runner.plot_scatter = Mock()
                    route_dir = Path(directory) / runner_class.__name__
                    fold_dir = route_dir / "fold_1"
                    fold_dir.mkdir(parents=True)
                    if runner_class in (FingerprintCCSBaselineRunner, SiameseCCSRunner):
                        metrics = runner.evaluate_fold(
                            model, arrays, 0, fold_dir, "test", runner.model_type
                        )
                        self.assertAlmostEqual(metrics["mspe"], 1.0)
                        with (route_dir / "test_results.csv").open(newline="") as handle:
                            saved = next(csv.DictReader(handle))
                    else:
                        evaluate = (
                            runner.evaluate_scratch_fold if runner_class is ScratchEndToEndRunner
                            else runner.evaluate_experiment_fold
                        )
                        metrics = evaluate(model, arrays, 0, fold_dir)
                        self.assertAlmostEqual(metrics["MSPE(%)"], 1.0)
                        saved = json.loads((fold_dir / "metrics.json").read_text())
                    self.assertAlmostEqual(float(saved["MSPE(%)"]), 1.0)
                    self.assertAlmostEqual(float(saved["MAE"]), 15.0)
                    self.assertAlmostEqual(float(saved["MAPE(%)"]), 10.0)


if __name__ == "__main__":
    unittest.main()
