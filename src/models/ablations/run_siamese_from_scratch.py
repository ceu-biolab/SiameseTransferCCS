from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("KERAS_BACKEND", "torch")

import joblib
import keras
import numpy as np
import pandas as pd
from keras import layers, models
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

from src.models.pretrain_siamese import SiameseFingerprintModel
from src.models.metrics import mspe_percent
from src.models.run_baseline import FoldArrays, set_seed
from src.models.run_siamese import SiameseCCSRunner


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "siamese_from_scratch"
PAPER_CONFIG_ROOT = PROJECT_ROOT / "configs" / "paper_experiments"
DEFAULT_CONFIG = PAPER_CONFIG_ROOT / "scratch_end_to_end.yaml"
SOURCE_SPLITS_DIR = PROJECT_ROOT / "results" / "CCSTrainer" / "splits"
INPUT_SPLITS_DIR = SOURCE_SPLITS_DIR
MANIFEST_PATH = PAPER_CONFIG_ROOT / "scratch_reference_manifest.json"
OUTPUT_ROOT = EXPERIMENT_ROOT / "outputs"
STATUS_ROOT = EXPERIMENT_ROOT / "status"

MODEL_ALIASES = {
    "lr_pc": "linear_regression_ft",
    "dgr_mlp": "gated_residual_mlp",
}
ROUTES = {
    "ccsbase_to_ccsbase": (("ccsbase",), "ccsbase"),
    "ccsbase_to_metlinccs": (("ccsbase",), "metlinccs"),
    "metlinccs_to_ccsbase": (("metlinccs",), "ccsbase"),
    "metlinccs_to_metlinccs": (("metlinccs",), "metlinccs"),
}
METRIC_COLUMNS = ["MAE", "MedAE", "MSE", "MAPE(%)", "MedAPE(%)", "MSPE(%)", "R2"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def stable_cell_seed(base_seed: int, source: str, route: str, fold: int, model: str) -> int:
    key = f"{base_seed}|{source}|{route}|{fold}|{model}".encode("utf-8")
    offset = int.from_bytes(hashlib.sha256(key).digest()[:4], "big")
    return int((base_seed + offset) % (2**31 - 1))


def prepare_split_inputs() -> dict[str, str]:
    """Validate and reuse the canonical downstream splits byte for byte."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    expected = manifest["split_sha256"]
    observed: dict[str, str] = {}
    for filename, expected_hash in expected.items():
        split_path = SOURCE_SPLITS_DIR / filename
        if not split_path.exists():
            raise FileNotFoundError(
                f"Missing canonical split {split_path}. "
                "Run the pretrained-reference downstream workflow first."
            )
        actual_hash = sha256(split_path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Split hash mismatch for {split_path}: expected {expected_hash}, "
                f"found {actual_hash}."
            )
        observed[filename] = actual_hash
    return observed


class ScratchEndToEndRunner(SiameseCCSRunner):
    """Train the reference wide+deep encoder and one CCS head from random weights."""

    def __init__(
        self,
        model_alias: str,
        fingerprint_source: str,
        config_path: str | Path = DEFAULT_CONFIG,
        folds: int = 5,
        random_seed: int = 42,
        routes: Iterable[str] | None = None,
        resume: bool = False,
    ):
        if model_alias not in MODEL_ALIASES:
            raise ValueError(f"Unknown model '{model_alias}'. Expected one of {sorted(MODEL_ALIASES)}")
        super().__init__(
            config_path=config_path,
            folds=folds,
            random_seed=random_seed,
            fingerprint_source=fingerprint_source,
            siamese_results_dir=EXPERIMENT_ROOT / "unused_pretrained_weights",
            experiment_tag="scratch_end_to_end",
        )
        self.model_alias = model_alias
        self.model_type = MODEL_ALIASES[model_alias]
        self.model_types = (self.model_type,)
        self.route_names = list(routes or ROUTES)
        unknown = sorted(set(self.route_names) - set(ROUTES))
        if unknown:
            raise ValueError(f"Unknown routes: {unknown}. Expected a subset of {sorted(ROUTES)}")
        self.resume = bool(resume)
        self.results_root = OUTPUT_ROOT / fingerprint_source / model_alias / f"seed_{random_seed}"
        self.splits_dir = INPUT_SPLITS_DIR
        self.status_root = STATUS_ROOT / fingerprint_source / model_alias / f"seed_{random_seed}"
        self.results_root.mkdir(parents=True, exist_ok=True)
        self.status_root.mkdir(parents=True, exist_ok=True)
        self.split_hashes = prepare_split_inputs()

    def load_siamese(self, fp_dim: int) -> SiameseFingerprintModel:
        """Return the reference architecture with random weights; never load a checkpoint."""
        return SiameseFingerprintModel(input_dim=fp_dim, use_logp=True, use_molvol=True)

    def siamese_embedding(self, fp_input, fp_dim: int, trainable: bool):
        # `trainable` is deliberately ignored: scratch_end_to_end trains the encoder
        # from the very first downstream batch.
        siamese = self.load_siamese(fp_dim)
        encoder_input = layers.Input(shape=(fp_dim,), name="siamese_encoder_input")
        encoder_output = siamese.get_embedding(encoder_input)
        encoder = models.Model(encoder_input, encoder_output, name="siamese_encoder")
        encoder.trainable = True
        return encoder(fp_input)

    def run(self) -> None:
        print(
            "Siamese scratch_end_to_end | "
            f"model={self.model_alias} ({self.model_type}) | source={self.fingerprint_source} | "
            f"folds={self.folds} | seed={self.random_seed}"
        )
        print("Encoder: reference wide+deep architecture, random initialization, trainable from batch 1")
        print(f"Outputs: {self.results_root}")

        for route_name in self.route_names:
            train_val_tuple, test_db = ROUTES[route_name]
            train_val_dbs = list(train_val_tuple)
            print(f"\nRoute {route_name}: train_val={train_val_dbs} -> test={test_db}")
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
                self.train_one_scratch_fold(
                    train_df, val_df, test_df, fold_idx, route_name, train_val_dbs, test_db
                )
            self.rebuild_route_results(route_name)

    def _read_split(self, config_key: str, suffix: str) -> list[dict[str, list[int]]]:
        split_path = self.splits_dir / f"{config_key}{suffix}"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing immutable experiment split: {split_path}")
        expected = self.split_hashes.get(split_path.name)
        actual = sha256(split_path)
        if expected is None or actual != expected:
            raise ValueError(f"Unrecognized or modified experiment split: {split_path}")
        return json.loads(split_path.read_text(encoding="utf-8"))

    def group_kfold_split_train_val_only(self, df, config_key: str, n_splits: int = 5):
        split_data = self._read_split(config_key, "_trainval_splits.json")
        return (
            [df.loc[item["train_idx"]].copy() for item in split_data],
            [df.loc[item["val_idx"]].copy() for item in split_data],
        )

    def group_kfold_split(self, df, config_key: str, n_splits: int = 5):
        split_data = self._read_split(config_key, "_splits.json")
        return (
            [df.loc[item["train_idx"]].copy() for item in split_data],
            [df.loc[item["val_idx"]].copy() for item in split_data],
            [df.loc[item["test_idx"]].copy() for item in split_data],
        )

    def route_dir(self, route_name: str) -> Path:
        return self.results_root / route_name

    def fold_dir(self, route_name: str, fold_idx: int) -> Path:
        return self.route_dir(route_name) / f"fold_{fold_idx + 1}"

    def fold_status_path(self, route_name: str, fold_idx: int) -> Path:
        return self.status_root / route_name / f"fold_{fold_idx + 1}.json"

    def fold_is_complete(self, route_name: str, fold_idx: int) -> bool:
        fold_dir = self.fold_dir(route_name, fold_idx)
        status_path = self.fold_status_path(route_name, fold_idx)
        required = [
            fold_dir / "metrics.json",
            fold_dir / "model.keras",
            fold_dir / "y_scaler.pkl",
            fold_dir / "adduct_encoder.pkl",
            fold_dir / f"training_history_fold_{fold_idx + 1}.json",
            fold_dir / "experiment_metadata.json",
            status_path,
        ]
        if not all(path.exists() for path in required):
            return False
        try:
            return json.loads(status_path.read_text(encoding="utf-8")).get("state") == "complete"
        except (OSError, json.JSONDecodeError):
            return False

    def train_one_scratch_fold(
        self,
        train_df,
        val_df,
        test_df,
        fold_idx: int,
        route_name: str,
        train_val_dbs: list[str],
        test_db: str,
    ) -> None:
        if self.resume and self.fold_is_complete(route_name, fold_idx):
            print(f"  Fold {fold_idx + 1}: complete, skipped (--resume)")
            return

        fold_dir = self.fold_dir(route_name, fold_idx)
        fold_dir.mkdir(parents=True, exist_ok=True)
        status_path = self.fold_status_path(route_name, fold_idx)
        cell_seed = stable_cell_seed(
            self.random_seed, self.fingerprint_source, route_name, fold_idx + 1, self.model_alias
        )
        started = utc_now()
        start_clock = time.monotonic()
        status = {
            "state": "running",
            "started_at": started,
            "source": self.fingerprint_source,
            "model": self.model_alias,
            "route": route_name,
            "fold": fold_idx + 1,
            "base_seed": self.random_seed,
            "cell_seed": cell_seed,
        }
        atomic_json(status_path, status)
        print(
            f"  Training {self.model_alias} | fold {fold_idx + 1} | seed={cell_seed} | "
            f"train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}"
        )

        try:
            keras.backend.clear_session()
            gc.collect()
            set_seed(cell_seed)
            arrays = self.prepare_fold_arrays(train_df, val_df, test_df)
            (fold_dir / "aux_features_metadata.json").write_text(
                json.dumps(arrays.aux_metadata, indent=2), encoding="utf-8"
            )
            metadata = {
                "protocol": "scratch_end_to_end",
                "pretrained_weights_loaded": False,
                "encoder_trainable_from_first_batch": True,
                "encoder_architecture": "reference_wide_deep",
                "model_alias": self.model_alias,
                "model_type": self.model_type,
                "fingerprint_source": self.fingerprint_source,
                "train_val_datasets": train_val_dbs,
                "test_dataset": test_db,
                "fold": fold_idx + 1,
                "base_seed": self.random_seed,
                "cell_seed": cell_seed,
                "split_sha256": self.split_hashes,
                "config_path": str(self.config_path),
                "config_sha256": sha256(Path(self.config_path)),
            }
            model = self.build_model(
                model_type=self.model_type,
                fp_dim=arrays.x_train.shape[1],
                adduct_dim=arrays.adducts_train.shape[1],
                aux_dim=arrays.aux_train.shape[1] if arrays.aux_train.shape[1] else 1,
            )
            if not model.get_layer("siamese_encoder").trainable:
                raise RuntimeError("Scratch encoder must be trainable before the first fit call.")
            metadata.update(
                {
                    "model_parameter_count": int(model.count_params()),
                    "encoder_parameter_count": int(model.get_layer("siamese_encoder").count_params()),
                }
            )
            atomic_json(fold_dir / "experiment_metadata.json", metadata)
            history = self.train_model(model, self.model_type, arrays, fold_idx, fold_dir)
            model.save(fold_dir / "model.keras")
            joblib.dump(arrays.y_scaler, fold_dir / "y_scaler.pkl")
            joblib.dump(arrays.adduct_encoder, fold_dir / "adduct_encoder.pkl")
            self.save_training_history(history, fold_idx, fold_dir)
            self.plot_loss_history(history, arrays.y_scaler, fold_idx, fold_dir, self.model_type)
            metrics = self.evaluate_scratch_fold(model, arrays, fold_idx, fold_dir)
            elapsed = time.monotonic() - start_clock
            atomic_json(
                status_path,
                {
                    **status,
                    "state": "complete",
                    "completed_at": utc_now(),
                    "elapsed_seconds": elapsed,
                    "epochs": len(history.get("loss", [])),
                    "metrics": metrics,
                },
            )
            print(f"    Fold {fold_idx + 1} done -> MAE: {metrics['MAE']:.3f}, R2: {metrics['R2']:.3f}")
        except Exception as exc:
            atomic_json(
                status_path,
                {
                    **status,
                    "state": "failed",
                    "failed_at": utc_now(),
                    "elapsed_seconds": time.monotonic() - start_clock,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise

    def evaluate_scratch_fold(
        self, model, arrays: FoldArrays, fold_idx: int, fold_dir: Path
    ) -> dict[str, float]:
        pred_scaled = model.predict(
            self.training_inputs(arrays, "test", self.model_type), batch_size=512, verbose=0
        ).flatten()
        y_pred = arrays.y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        y_test = arrays.y_scaler.inverse_transform(arrays.y_test_scaled.reshape(-1, 1)).flatten()
        metrics = {
            "MAE": float(mean_absolute_error(y_test, y_pred)),
            "MedAE": float(median_absolute_error(y_test, y_pred)),
            "MSE": float(mean_squared_error(y_test, y_pred)),
            "MAPE(%)": float(mean_absolute_percentage_error(y_test, y_pred) * 100),
            "MedAPE(%)": float(
                np.median(np.abs((y_test - y_pred) / np.clip(y_test, 1e-8, None))) * 100
            ),
            "MSPE(%)": mspe_percent(y_test, y_pred),
            "R2": float(r2_score(y_test, y_pred)),
        }
        atomic_json(fold_dir / "metrics.json", {"Fold": fold_idx + 1, **metrics})
        self.plot_scatter(y_test, y_pred, fold_idx, fold_dir, self.model_type)
        return metrics

    def rebuild_route_results(self, route_name: str) -> Path | None:
        rows: list[dict[str, Any]] = []
        for fold_idx in range(self.folds):
            path = self.fold_dir(route_name, fold_idx) / "metrics.json"
            if path.exists():
                rows.append(json.loads(path.read_text(encoding="utf-8")))
        if not rows:
            return None
        rows.sort(key=lambda row: int(row["Fold"]))
        frame = pd.DataFrame(rows, columns=["Fold", *METRIC_COLUMNS])
        output_path = self.route_dir(route_name) / "results.csv"
        temporary = output_path.with_suffix(f".{os.getpid()}.tmp")
        frame.to_csv(temporary, index=False, float_format="%.6f", quoting=csv.QUOTE_MINIMAL)
        temporary.replace(output_path)
        print(f"  Route metrics ({len(frame)}/{self.folds} folds) -> {output_path}")
        return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train one CCS head end-to-end with the reference Siamese encoder "
            "initialized from scratch."
        )
    )
    parser.add_argument("--model", required=True, choices=sorted(MODEL_ALIASES))
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=("rdkit", "alvadesc"),
        default=["rdkit", "alvadesc"],
    )
    parser.add_argument("--routes", nargs="+", choices=sorted(ROUTES), default=list(ROUTES))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for source in args.sources:
        ScratchEndToEndRunner(
            model_alias=args.model,
            fingerprint_source=source,
            config_path=args.config,
            folds=args.folds,
            random_seed=args.random_seed,
            routes=args.routes,
            resume=args.resume,
        ).run()


if __name__ == "__main__":
    main()
