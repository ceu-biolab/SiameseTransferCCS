from __future__ import annotations

import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np
import pandas as pd
import torch

from src.models.losses import scalar_absolute_error


PRETRAIN_MODULES = (
    "src.models.pretrain_siamese",
    "src.models.pretrain_siamese_alvadesc",
    "src.models.ablations.pretrain_siamese_wide_only",
    "src.models.ablations.pretrain_siamese_wide_only_alvadesc",
    "src.models.ablations.pretrain_siamese_deep_only",
    "src.models.ablations.pretrain_siamese_deep_only_alvadesc",
)
VOLUME_OUTPUTS = ("molvol_1", "molvol_2", "molvol_delta")


def scalar_model():
    inputs = keras.Input(shape=(1,))
    output = keras.layers.Dense(1, use_bias=False, kernel_initializer="ones")(inputs)
    model = keras.Model(inputs, keras.ops.squeeze(output, axis=-1))
    model.compile(optimizer=keras.optimizers.SGD(0.1), loss=scalar_absolute_error)
    return model


class VolumeLossMaskingTests(unittest.TestCase):
    def test_masked_loss_and_gradients_keep_full_batch_denominator(self):
        model = scalar_model()
        cases = (
            # The original bug returned 2.5 and a nonzero masked gradient here.
            ([0, 10], [1, 0], 0.0, [0, 0]),
            ([2, 4, 10, 20], [1, 1, 0, 0], 1.5, [0.25, 0.25, 0, 0]),
            ([2, -4], [0, 0], 0.0, [0, 0]),
            ([2, -4], [1, 1], 3.0, [0.5, -0.5]),
            ([10], [0], 0.0, [0]),
        )
        for predictions, mask, expected_loss, expected_gradient in cases:
            with self.subTest(predictions=predictions, mask=mask):
                predicted = torch.tensor(predictions, dtype=torch.float32, requires_grad=True)
                loss = model.compute_loss(
                    y=torch.zeros_like(predicted),
                    y_pred=predicted,
                    sample_weight=torch.tensor(mask, dtype=torch.float32),
                )
                loss.backward()
                self.assertAlmostEqual(float(loss.detach()), expected_loss)
                np.testing.assert_allclose(predicted.grad.numpy(), expected_gradient)

    def test_fit_validation_and_saved_model_exclude_masked_errors(self):
        model = scalar_model()
        x = np.array([[0], [10]], dtype=np.float32)
        y = np.zeros(2, dtype=np.float32)
        weights = np.array([1, 0], dtype=np.float32)
        history = model.fit(
            x, y, sample_weight=weights,
            validation_data=(x, y, weights),
            batch_size=2, epochs=1, shuffle=False, verbose=0,
        )
        self.assertEqual(history.history["loss"], [0.0])
        self.assertEqual(history.history["val_loss"], [0.0])
        np.testing.assert_array_equal(model.get_weights()[0], [[1.0]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.keras"
            model.save(path)
            restored = keras.models.load_model(path)
            self.assertEqual(
                restored.evaluate(x, y, sample_weight=weights, batch_size=2, verbose=0),
                0.0,
            )

    def test_all_pretrainers_mask_actual_volume_outputs_and_validation_losses(self):
        # Distinct fingerprints identify the molecule with a valid volume.
        frame = pd.DataFrame({
            "V1": [1.0, 0.0], "V2": [0.0, 1.0],
            "V3": [0.0, 0.0], "V4": [0.0, 0.0],
            "classification": ["A", "A"],
            "logp_scaled": [0.0, 1.0],
            "molvol_scaled": [2.0, 0.0],
            "molvol_valid": [1.0, 0.0],
        })
        for module_name in PRETRAIN_MODULES:
            with self.subTest(pretrainer=module_name):
                keras.backend.clear_session()
                keras.utils.set_random_seed(42)
                module = importlib.import_module(module_name)
                trainer = module.FingerprintSiamesePretrainer.__new__(
                    module.FingerprintSiamesePretrainer
                )
                trainer.use_logp = trainer.use_molvol = True
                trainer.loss_cfg = {}
                x1, x2, targets, weights = trainer.generate_pairs(
                    frame, ["V1", "V2", "V3", "V4"], 64
                )
                valid_1, valid_2 = x1[:, 0], x2[:, 0]
                np.testing.assert_array_equal(weights["molvol_1"], valid_1)
                np.testing.assert_array_equal(weights["molvol_2"], valid_2)
                np.testing.assert_array_equal(weights["molvol_delta"], valid_1 * valid_2)
                self.assertEqual(set(zip(valid_1, valid_2)), {(0, 0), (0, 1), (1, 0), (1, 1)})

                # Exercise the real architectures at small widths to keep tests fast.
                model_module = importlib.import_module(module.SiameseFingerprintModel.__module__)
                with patch.multiple(model_module, WIDE_DIM=8, DEEP_DIM=4, EMBEDDING_DIM=12):
                    model = module.SiameseFingerprintModel(4, use_logp=True, use_molvol=True)
                    predicted = model((x1, x2), training=False)
                trainer.compile_model(model)
                loss = model.compute_loss(y=targets, y_pred=predicted, sample_weight=weights)
                gradients = torch.autograd.grad(loss, [predicted[key] for key in VOLUME_OUTPUTS])
                for key, gradient in zip(VOLUME_OUTPUTS, gradients):
                    self.assertEqual(tuple(predicted[key].shape), (64,))
                    self.assertTrue(torch.isfinite(gradient).all())
                    np.testing.assert_array_equal(
                        gradient.detach().cpu().numpy()[weights[key] == 0], 0.0
                    )

                baseline = model.test_on_batch(
                    (x1, x2), targets, sample_weight=weights, return_dict=True
                )
                for key in VOLUME_OUTPUTS:
                    errors = np.abs(keras.ops.convert_to_numpy(predicted[key]) - targets[key])
                    self.assertAlmostEqual(
                        float(baseline[f"{key}_loss"]), float(np.mean(errors * weights[key])), places=5
                    )
                # Changing only masked targets must not change validation loss.
                changed_targets = {key: value.copy() for key, value in targets.items()}
                for key in VOLUME_OUTPUTS:
                    changed_targets[key][weights[key] == 0] += 1000.0
                model.reset_metrics()
                changed = model.test_on_batch(
                    (x1, x2), changed_targets, sample_weight=weights, return_dict=True
                )
                self.assertAlmostEqual(float(changed["loss"]), float(baseline["loss"]), places=5)


if __name__ == "__main__":
    unittest.main()
