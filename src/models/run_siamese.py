from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("KERAS_BACKEND", "torch")

import joblib
import keras
import numpy as np
import pandas as pd
from keras import callbacks, layers, models
from keras.optimizers import RMSprop
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

from src.models.pretrain_siamese import (
    BASE_EMBEDDING_DIM,
    DEEP_DIM,
    EMBEDDING_DIM,
    FINGERPRINT_DIM,
    WIDE_DIM,
    SiameseFingerprintModel,
)
from src.models.run_baseline import (
    DATASET_CONFIGS,
    DEFAULT_CONFIG_PATH,
    RESULTS_ROOT,
    VALID_FINGERPRINT_SOURCES,
    FingerprintCCSBaselineRunner,
    FoldArrays,
)


SIAMESE_RESULTS_DIRS = {
    "alvadesc": Path("results/Siamese_physchem_alvadesc"),
    "rdkit": Path("results/Siamese_physchem"),
}
MODEL_TYPES = ("linear_regression", "linear_regression_ft", "gated_residual_mlp")
MODEL_DISPLAY_NAMES = {
    "linear_regression": "Linear regression",
    "linear_regression_ft": "Linear regression FT",
    "gated_residual_mlp": "Gated residual MLP",
}


class SiameseCCSRunner(FingerprintCCSBaselineRunner):
    """Run CCS prediction using the pretrained fingerprint Siamese encoder."""

    def __init__(
        self,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        folds: int = 5,
        random_seed: int = 42,
        fingerprint_source: str = "rdkit",
        siamese_results_dir: str | Path | None = None,
        experiment_tag: str | None = None,
        model_types: tuple[str, ...] | list[str] | None = None,
    ):
        super().__init__(
            config_path=config_path,
            folds=folds,
            random_seed=random_seed,
            fingerprint_source=fingerprint_source,
        )
        self.siamese_results_dir = Path(siamese_results_dir) if siamese_results_dir else SIAMESE_RESULTS_DIRS[fingerprint_source]
        self.experiment_tag = str(experiment_tag).strip() if experiment_tag else ""
        self.results_root = RESULTS_ROOT
        self.splits_dir = self.results_root / "splits"
        selected_models = tuple(model_types) if model_types is not None else MODEL_TYPES
        unknown_models = sorted(set(selected_models) - set(MODEL_TYPES))
        if unknown_models:
            raise ValueError(
                f"Unknown model types: {unknown_models}. Expected a subset of {MODEL_TYPES}."
            )
        if not selected_models:
            raise ValueError("At least one model type must be selected.")
        self.model_types = selected_models

    def run(self) -> None:
        print(f"CCS siamesa runner | config={self.config_path} | folds={self.folds}")
        print(f"Fingerprint source: {self.fingerprint_source}")
        print(f"Siamese results dir: {self.siamese_results_dir}")
        if self.experiment_tag:
            print(f"Experiment tag: {self.experiment_tag}")
        print("CCS loss: MAE")
        print("Models:")
        for model_type in self.model_types:
            print(f"  - {MODEL_DISPLAY_NAMES[model_type]}: {self.model_run_description(model_type)}")
        print(f"Mass features: {'enabled' if self.use_mass else 'disabled'}")
        print(f"Molecular features: {'enabled' if self.use_molecular_features else 'disabled'}")

        for train_val_dbs, test_db in DATASET_CONFIGS:
            train_val_dbs = list(train_val_dbs)
            print(f"\nDataset config: train_val={train_val_dbs} -> test={test_db}")
            train_val_df = self.load_data(train_val_dbs)
            test_df_full = self.load_data([test_db])

            same_database = len(train_val_dbs) == 1 and train_val_dbs[0] == test_db
            config_key = self._split_key(train_val_dbs, test_db, same_database)
            if same_database:
                train_dfs, val_dfs, test_dfs = self.group_kfold_split(train_val_df, config_key)
            else:
                train_dfs, val_dfs = self.group_kfold_split_train_val_only(train_val_df, config_key)
                test_dfs = [test_df_full.copy() for _ in train_dfs]

            for fold_idx, (train_df, val_df, test_df) in enumerate(zip(train_dfs, val_dfs, test_dfs)):
                if fold_idx >= self.folds:
                    break
                for model_type in self.model_types:
                    if model_type == "gated_residual_mlp" and not self.aux_features_enabled():
                        print(
                            "  Skipping gated_residual_mlp: it requires mass.enabled=true "
                            "or molecular_features.enabled=true."
                        )
                        continue
                    self.train_one_fold(train_df, val_df, test_df, fold_idx, model_type, train_val_dbs, test_db)

            self.write_final_metrics(train_val_dbs, test_db)

    def train_one_fold(
        self,
        train_df,
        val_df,
        test_df,
        fold_idx: int,
        model_type: str,
        train_val_dbs: list[str],
        test_db: str,
    ) -> None:
        display_name = MODEL_DISPLAY_NAMES[model_type]
        print(
            f"  Training {display_name} | siamesa | fold {fold_idx + 1} "
            f"| train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}"
        )
        arrays = self.prepare_fold_arrays(train_df, val_df, test_df)
        subfolder = self.result_subfolder(train_val_dbs, test_db, model_type)
        fold_dir = self.results_root / subfolder / f"fold_{fold_idx + 1}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        results_csv_path = self.results_root / subfolder / f"{subfolder}_results.csv"
        if fold_idx == 0 and results_csv_path.exists():
            results_csv_path.unlink()

        (fold_dir / "aux_features_metadata.json").write_text(
            json.dumps(arrays.aux_metadata, indent=2),
            encoding="utf-8",
        )
        (fold_dir / "siamese_metadata.json").write_text(
            json.dumps(self.siamese_metadata(model_type), indent=2),
            encoding="utf-8",
        )

        model = self.build_model(
            model_type=model_type,
            fp_dim=arrays.x_train.shape[1],
            adduct_dim=arrays.adducts_train.shape[1],
            aux_dim=arrays.aux_train.shape[1] if arrays.aux_train.shape[1] else 1,
        )
        history = self.train_model(model, model_type, arrays, fold_idx, fold_dir)
        model.save(fold_dir / "model.keras")
        joblib.dump(arrays.y_scaler, fold_dir / "y_scaler.pkl")
        joblib.dump(arrays.adduct_encoder, fold_dir / "adduct_encoder.pkl")
        self.save_training_history(history, fold_idx, fold_dir)
        self.plot_loss_history(history, arrays.y_scaler, fold_idx, fold_dir, model_type)
        self.evaluate_fold(model, arrays, fold_idx, fold_dir, subfolder, model_type)

    def siamese_metadata(self, model_type: str) -> dict[str, Any]:
        return {
            "mode": "siamesa",
            "fingerprint_source": self.fingerprint_source,
            "siamese_results_dir": str(self.siamese_results_dir),
            "siamese_weights": str(self.weights_path()),
            "experiment_tag": self.experiment_tag,
            "ccs_loss": "mae",
            "fine_tune_siamese": self.fine_tunes_siamese(model_type),
            "uses_aux_features": self.uses_aux_features(model_type),
        }

    def build_model(self, model_type: str, fp_dim: int, adduct_dim: int, aux_dim: int = 1):
        if model_type in {"linear_regression", "linear_regression_ft"}:
            return self.build_linear_regression(fp_dim, adduct_dim, aux_dim, model_type=model_type)
        if model_type == "gated_residual_mlp":
            return self.build_gated_residual_mlp(fp_dim, adduct_dim, aux_dim, fine_tune_siamese=False)
        raise ValueError(f"Unknown model_type='{model_type}'. Expected one of: {MODEL_TYPES}")

    def build_linear_regression(
        self,
        fp_dim: int,
        adduct_dim: int,
        aux_dim: int = 1,
        model_type: str = "linear_regression",
    ):
        fp_input, adduct_input, aux_input = self.make_inputs(fp_dim, adduct_dim, aux_dim)
        features = self.siamese_embedding(fp_input, fp_dim=fp_dim, trainable=False)
        x = self.concat_inputs(features, adduct_input, aux_input, use_aux=self.uses_aux_features(model_type))
        output = layers.Dense(1, activation="linear")(x)
        return models.Model(self.model_inputs(fp_input, adduct_input, aux_input, model_type), output)

    def build_gated_residual_mlp(
        self,
        fp_dim: int,
        adduct_dim: int,
        aux_dim: int = 1,
        fine_tune_siamese: bool = False,
    ):
        if not self.aux_features_enabled():
            raise ValueError("gated_residual_mlp requires mass.enabled=true or molecular_features.enabled=true.")
        fp_input, adduct_input, aux_input = self.make_inputs(fp_dim, adduct_dim, aux_dim)
        features = self.siamese_embedding(fp_input, fp_dim=fp_dim, trainable=fine_tune_siamese)
        feature_dim = int(features.shape[-1])
        gamma = layers.Dense(
            feature_dim,
            activation="linear",
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="aux_gamma",
        )(aux_input)
        beta = layers.Dense(
            feature_dim,
            activation="linear",
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="aux_beta",
        )(aux_input)
        scaled_delta = layers.Multiply(name="aux_feature_interaction")([features, gamma])
        gated_features = layers.Add(name="aux_gated_features")([features, scaled_delta, beta])
        x = self.concat_inputs(gated_features, adduct_input, aux_input, use_aux=True)
        output = self.build_residual_core(x, self.config.get("heads", {}).get("gated_residual_mlp", {}))
        return models.Model(self.model_inputs(fp_input, adduct_input, aux_input, "gated_residual_mlp"), output)

    def siamese_embedding(self, fp_input, fp_dim: int, trainable: bool):
        siamese = self.load_siamese(fp_dim)
        encoder_input = layers.Input(shape=(fp_dim,), name="siamese_encoder_input")
        encoder_output = siamese.get_embedding(encoder_input)
        encoder = models.Model(encoder_input, encoder_output, name="siamese_encoder")
        encoder.trainable = bool(trainable)
        return encoder(fp_input)

    def load_siamese(self, fp_dim: int) -> SiameseFingerprintModel:
        weights_path = self.weights_path()
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Missing Siamese weights: {weights_path}. "
                "Run src.models.pretrain_siamese before run_siamese."
            )

        manifest = self.read_siamese_manifest()
        self.validate_siamese_manifest(manifest)
        model = SiameseFingerprintModel(
            input_dim=fp_dim,
            use_logp=bool(manifest.get("use_logp", True)),
            use_molvol=bool(manifest.get("use_molvol", True)),
        )
        dummy = np.zeros((1, fp_dim), dtype=np.float32)
        _ = model((dummy, dummy))
        model.load_weights(weights_path)
        return model

    def weights_path(self) -> Path:
        return self.siamese_results_dir / "fold_1" / "best.weights.h5"

    def read_siamese_manifest(self) -> dict[str, Any]:
        manifest_path = self.siamese_results_dir / "model_manifest.json"
        if not manifest_path.exists():
            return {}
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def validate_siamese_manifest(self, manifest: dict[str, Any]) -> None:
        manifest_path = self.siamese_results_dir / "model_manifest.json"
        if not manifest_path.exists():
            return
        manifest_source = manifest.get("fingerprint_source")
        if manifest_source != self.fingerprint_source:
            raise ValueError(
                f"Siamese manifest fingerprint_source mismatch at {manifest_path}: "
                f"expected '{self.fingerprint_source}', found '{manifest_source}'."
            )
        expected_dims = {
            "input_dim": FINGERPRINT_DIM,
            "embedding_dim": EMBEDDING_DIM,
            "base_embedding_dim": BASE_EMBEDDING_DIM,
            "wide_dim": WIDE_DIM,
            "deep_dim": DEEP_DIM,
        }
        for key, expected in expected_dims.items():
            actual = manifest.get(key)
            if actual is not None and int(actual) != expected:
                raise ValueError(
                    f"Siamese manifest {key} mismatch at {manifest_path}: "
                    f"expected {expected}, found {actual}. Retrain the Siamese model with the current architecture."
                )

    @staticmethod
    def fine_tunes_siamese(model_type: str) -> bool:
        return model_type in {"linear_regression_ft", "gated_residual_mlp"}

    def uses_aux_features(self, model_type: str) -> bool:
        if model_type == "linear_regression":
            return False
        return self.aux_features_enabled()

    def model_run_description(self, model_type: str) -> str:
        ft = "Siamese fine-tuning" if self.fine_tunes_siamese(model_type) else "Siamese frozen"
        aux = "uses mass/molecular features" if self.uses_aux_features(model_type) else "no mass/molecular features"
        return f"{ft}; {aux}"

    def train_model(self, model, model_type: str, arrays: FoldArrays, fold_idx: int, fold_dir: Path) -> dict[str, list[float]]:
        fold_dir.mkdir(parents=True, exist_ok=True)
        batch_size = int(self.training_cfg.get("batch_size", 32))
        phase1_epochs = int(self.training_cfg.get("phase1_epochs", 10000))
        phase2_epochs = int(self.training_cfg.get("phase2_epochs", 10000))
        fine_tune_epochs = int(self.training_cfg.get("fine_tune_epochs", 20))
        early_stopping_patience = int(self.training_cfg.get("early_stopping_patience", 20))

        model.compile(
            optimizer=RMSprop(
                learning_rate=float(self.training_cfg.get("phase1_learning_rate", 1e-3)),
                rho=0.7,
                momentum=0.7,
            ),
            loss="mae",
            metrics=["mae"],
        )
        history_phase1 = model.fit(
            self.training_inputs(arrays, "train", model_type),
            arrays.y_train_scaled,
            validation_data=(self.training_inputs(arrays, "val", model_type), arrays.y_val_scaled),
            epochs=phase1_epochs,
            batch_size=batch_size,
            callbacks=[
                callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=early_stopping_patience,
                    restore_best_weights=True,
                )
            ],
            verbose=0,
        )

        history_ft = {}
        if self.fine_tunes_siamese(model_type):
            model.get_layer("siamese_encoder").trainable = True
            model.compile(
                optimizer=RMSprop(
                    learning_rate=float(self.training_cfg.get("fine_tune_learning_rate", 1e-5)),
                    rho=0.7,
                    momentum=0.7,
                ),
                loss="mae",
                metrics=["mae"],
            )
            history_ft = model.fit(
                self.training_inputs(arrays, "train", model_type),
                arrays.y_train_scaled,
                validation_data=(self.training_inputs(arrays, "val", model_type), arrays.y_val_scaled),
                epochs=fine_tune_epochs,
                batch_size=batch_size,
                callbacks=[
                    callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=early_stopping_patience,
                        restore_best_weights=True,
                    )
                ],
                verbose=0,
            ).history

        model.compile(
            optimizer=RMSprop(
                learning_rate=float(self.training_cfg.get("phase2_learning_rate", 1e-4)),
                rho=0.7,
                momentum=0.7,
            ),
            loss="mae",
            metrics=["mae"],
        )
        csv_logger = callbacks.CSVLogger(fold_dir / f"training_log_fold_{fold_idx + 1}.csv", append=False, separator=",")
        history_phase2 = model.fit(
            self.training_inputs(arrays, "train", model_type),
            arrays.y_train_scaled,
            validation_data=(self.training_inputs(arrays, "val", model_type), arrays.y_val_scaled),
            epochs=phase2_epochs,
            batch_size=batch_size,
            callbacks=[
                callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=early_stopping_patience,
                    restore_best_weights=True,
                ),
                callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=float(self.training_cfg.get("reduce_lr_factor", 1 / 3)),
                    patience=int(self.training_cfg.get("reduce_lr_patience", 5)),
                    min_delta=float(self.training_cfg.get("reduce_lr_min_delta", 1e-5)),
                    verbose=0,
                ),
                csv_logger,
            ],
            verbose=0,
        )

        history = {}
        for key in set(history_phase1.history) | set(history_ft) | set(history_phase2.history):
            history[key] = (
                history_phase1.history.get(key, [])
                + history_ft.get(key, [])
                + history_phase2.history.get(key, [])
            )
        return {key: [float(value) for value in values] for key, values in history.items()}

    def result_subfolder(self, train_val_dbs: list[str], test_db: str, model_type: str) -> str:
        features_tag = "with_molfeatures" if self.use_molecular_features else "no_molfeatures"
        mass_tag = "with_mass" if self.use_mass else "no_mass"
        folds_tag = "single_fold" if self.folds == 1 else "five_folds"
        experiment_tag = f"{self.experiment_tag}_" if self.experiment_tag else ""
        return (
            f"train_val_{'_'.join(train_val_dbs)}_test_{test_db}_{self.fingerprint_source}_"
            f"siamesa_{experiment_tag}{features_tag}_{mass_tag}_{model_type}_{folds_tag}"
        )

    def write_final_metrics(self, train_val_dbs: list[str], test_db: str) -> None:
        for model_type in self.model_types:
            subfolder = self.result_subfolder(train_val_dbs, test_db, model_type)
            csv_path = self.results_root / subfolder / f"{subfolder}_results.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            df = df[df.iloc[:, 0].astype(str).str.strip() != "Total"].copy()
            if df.empty:
                continue
            numeric_data = df.iloc[:, 1:].astype(float).to_numpy()
            means = np.mean(numeric_data, axis=0)
            stds = np.std(numeric_data, axis=0)
            total_row = ["Total", *[f"{mean:.3f}+-{std:.3f}" for mean, std in zip(means, stds)]]
            with csv_path.open("a", newline="") as f:
                csv.writer(f).writerow(total_row)
            print(f"  Aggregated metrics -> {csv_path}")

    def plot_loss_history(self, history: dict[str, list[float]], y_scaler, fold_idx: int, fold_dir: Path, model_type: str) -> None:
        train_loss = np.asarray(history["loss"], dtype=np.float32)
        val_loss = np.asarray(history.get("val_loss", []), dtype=np.float32)
        scale_factor = y_scaler.scale_[0]
        train_loss = train_loss * scale_factor
        val_loss = val_loss * scale_factor

        plt.figure(figsize=(10, 6))
        plt.plot(train_loss, label="Training Loss", linewidth=2)
        if len(val_loss):
            plt.plot(val_loss, label="Validation Loss", linewidth=2)
        plt.title(f"CCS Prediction - siamesa + {MODEL_DISPLAY_NAMES[model_type]} - Fold {fold_idx + 1}")
        plt.xlabel("Epoch")
        plt.ylabel("MAE loss on original CCS scale")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(fold_dir / f"loss_curve_fold_{fold_idx + 1}.png", dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, fold_idx: int, fold_dir: Path, model_type: str) -> None:
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, s=4, alpha=0.6, edgecolors="w", linewidth=0.3)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Perfect prediction")
        plt.xlabel("True CCS")
        plt.ylabel("Predicted CCS")
        plt.title(
            f"siamesa + {MODEL_DISPLAY_NAMES[model_type]} - Fold {fold_idx + 1}\n"
            f"MAE: {mean_absolute_error(y_true, y_pred):.3f} | R2: {r2_score(y_true, y_pred):.3f}"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(fold_dir / f"scatter_fold_{fold_idx + 1}.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _split_key(self, train_val_dbs: list[str], test_db: str, same_database: bool) -> str:
        split_kind = "same_db" if same_database else "external_test"
        return (
            f"train_val_{'_'.join(train_val_dbs)}_test_{test_db}_"
            f"{self.fingerprint_source}_fingerprints_{split_kind}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CCS prediction on pretrained Siamese fingerprint embeddings.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--folds", type=int, default=5, help="Number of generated folds to execute, from 1 to 5.")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--fingerprint-source", choices=sorted(VALID_FINGERPRINT_SOURCES), default="rdkit")
    parser.add_argument("--siamese-results-dir", default=None)
    parser.add_argument("--experiment-tag", default=None, help="Optional tag added to CCS result folder names.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_TYPES,
        default=None,
        help="CCS heads to train. By default all three heads are trained.",
    )
    args = parser.parse_args()
    SiameseCCSRunner(
        config_path=args.config,
        folds=args.folds,
        random_seed=args.random_seed,
        fingerprint_source=args.fingerprint_source,
        siamese_results_dir=args.siamese_results_dir,
        experiment_tag=args.experiment_tag,
        model_types=args.models,
    ).run()


if __name__ == "__main__":
    main()
