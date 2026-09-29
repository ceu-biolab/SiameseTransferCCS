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
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

from src.models.ablations.run_siamese_wide_only import SiameseCCSRunner as WideOnlyCCSRunner
from src.models.run_baseline import FoldArrays, set_seed
from src.models.metrics import mspe_percent


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "siamese_wide_only_pretrained_dgr"
PAPER_CONFIG_ROOT = PROJECT_ROOT / "configs" / "paper_experiments"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "ccs_prediction_heads.yaml"
MANIFEST_PATH = PAPER_CONFIG_ROOT / "wide_only_dgr_manifest.json"
SOURCE_SPLITS_DIR = PROJECT_ROOT / "results" / "CCSTrainer" / "splits"
INPUT_SPLITS_DIR = SOURCE_SPLITS_DIR
OUTPUT_ROOT = EXPERIMENT_ROOT / "outputs"
STATUS_ROOT = EXPERIMENT_ROOT / "status"
MODEL_TYPE = "gated_residual_mlp"
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


def stable_cell_seed(base_seed: int, source: str, route: str, fold: int) -> int:
    key = f"{base_seed}|{source}|{route}|{fold}|wide_only_pretrained_dgr".encode("utf-8")
    offset = int.from_bytes(hashlib.sha256(key).digest()[:4], "big")
    return int((base_seed + offset) % (2**31 - 1))


def load_experiment_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def validate_experiment_inputs(source: str, config_path: Path) -> dict[str, str]:
    manifest = load_experiment_manifest()
    if source not in manifest["checkpoints"]:
        raise ValueError(f"No checkpoint is registered for fingerprint source '{source}'.")

    expected_config = Path(manifest["config_path"])
    try:
        config_relative = config_path.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        config_relative = config_path.resolve()
    if Path(config_relative) != expected_config:
        raise ValueError(
            f"This frozen experiment requires config {expected_config}, found {config_path}."
        )
    config_hash = sha256(config_path)
    if config_hash != manifest["config_sha256"]:
        raise ValueError(
            f"Config hash mismatch for {config_path}: expected {manifest['config_sha256']}, "
            f"found {config_hash}."
        )

    checkpoint = manifest["checkpoints"][source]
    observed: dict[str, str] = {"config_sha256": config_hash}
    for path_key, hash_key in (
        ("weights", "weights_sha256"),
        ("model_manifest", "model_manifest_sha256"),
    ):
        path = PROJECT_ROOT / checkpoint[path_key]
        if not path.exists():
            raise FileNotFoundError(f"Missing registered {path_key}: {path}")
        actual = sha256(path)
        expected = checkpoint[hash_key]
        if actual != expected:
            raise ValueError(
                f"Hash mismatch for {path}: expected {expected}, found {actual}."
            )
        observed[hash_key] = actual

    model_manifest_path = PROJECT_ROOT / checkpoint["model_manifest"]
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    expected_fields = {
        "fingerprint_source": source,
        "branch_mode": "wide_only",
        "model_type": "fingerprint_siamese_pretrain_wide_only",
        "pretraining_protocol": "hmdb_90_10",
        "wide_dim": 1536,
        "embedding_dim": 2048,
    }
    for key, expected in expected_fields.items():
        if model_manifest.get(key) != expected:
            raise ValueError(
                f"Checkpoint manifest mismatch for {key}: expected {expected!r}, "
                f"found {model_manifest.get(key)!r}."
            )
    return observed


def prepare_split_inputs() -> dict[str, str]:
    manifest = load_experiment_manifest()
    expected = manifest["split_sha256"]
    observed: dict[str, str] = {}
    for filename, expected_hash in expected.items():
        split_path = SOURCE_SPLITS_DIR / filename
        if not split_path.exists():
            raise FileNotFoundError(f"Missing canonical split: {split_path}")
        actual_hash = sha256(split_path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Split hash mismatch for {split_path}: expected {expected_hash}, "
                f"found {actual_hash}."
            )
        observed[filename] = actual_hash
    return observed


class WideOnlyPretrainedDGRRunner(WideOnlyCCSRunner):
    """Fine-tune an HMDB-pretrained wide-only encoder with the existing DGR-MLP head."""

    def __init__(
        self,
        fingerprint_source: str,
        config_path: str | Path = DEFAULT_CONFIG,
        folds: int = 5,
        random_seed: int = 42,
        routes: Iterable[str] | None = None,
        resume: bool = False,
    ):
        config_path = Path(config_path).resolve()
        manifest = load_experiment_manifest()
        checkpoint = manifest["checkpoints"].get(fingerprint_source)
        if checkpoint is None:
            raise ValueError(f"Unknown fingerprint source '{fingerprint_source}'.")
        checkpoint_dir = PROJECT_ROOT / checkpoint["directory"]
        super().__init__(
            config_path=config_path,
            folds=folds,
            random_seed=random_seed,
            fingerprint_source=fingerprint_source,
            siamese_results_dir=checkpoint_dir,
            experiment_tag="wide_only_pretrained_hmdb_dgr",
        )
        self.model_type = MODEL_TYPE
        self.route_names = list(routes or ROUTES)
        unknown = sorted(set(self.route_names) - set(ROUTES))
        if unknown:
            raise ValueError(f"Unknown routes: {unknown}. Expected a subset of {sorted(ROUTES)}")
        self.resume = bool(resume)
        self.results_root = OUTPUT_ROOT / fingerprint_source / f"seed_{random_seed}"
        self.status_root = STATUS_ROOT / fingerprint_source / f"seed_{random_seed}"
        self.splits_dir = INPUT_SPLITS_DIR
        self.results_root.mkdir(parents=True, exist_ok=True)
        self.status_root.mkdir(parents=True, exist_ok=True)
        self.input_hashes = validate_experiment_inputs(fingerprint_source, config_path)
        self.split_hashes = prepare_split_inputs()

    def run(self) -> None:
        print(
            "Siamese wide-only pretrained HMDB + DGR-MLP | "
            f"source={self.fingerprint_source} | folds={self.folds} | seed={self.random_seed}"
        )
        print(f"Checkpoint: {self.weights_path()}")
        print("Schedule: frozen phase -> encoder fine-tuning -> joint phase")
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

            for fold_idx, (train_df, val_df, test_df) in enumerate(
                zip(train_dfs, val_dfs, test_dfs)
            ):
                if fold_idx >= self.folds:
                    break
                self.train_one_experiment_fold(
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

    def train_one_experiment_fold(
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
            self.random_seed, self.fingerprint_source, route_name, fold_idx + 1
        )
        start_clock = time.monotonic()
        status = {
            "state": "running",
            "started_at": utc_now(),
            "source": self.fingerprint_source,
            "model": MODEL_TYPE,
            "route": route_name,
            "fold": fold_idx + 1,
            "base_seed": self.random_seed,
            "cell_seed": cell_seed,
        }
        atomic_json(status_path, status)
        print(
            f"  Training {MODEL_TYPE} | fold {fold_idx + 1} | seed={cell_seed} | "
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
            model = self.build_model(
                model_type=MODEL_TYPE,
                fp_dim=arrays.x_train.shape[1],
                adduct_dim=arrays.adducts_train.shape[1],
                aux_dim=arrays.aux_train.shape[1] if arrays.aux_train.shape[1] else 1,
            )
            encoder = model.get_layer("siamese_encoder")
            if encoder.trainable:
                raise RuntimeError("The pretrained encoder must be frozen before phase 1.")

            metadata = {
                "protocol": "wide_only_pretrained_hmdb_dgr",
                "pretrained_weights_loaded": True,
                "pretraining_protocol": "hmdb_90_10",
                "encoder_architecture": "wide_only",
                "encoder_frozen_during_phase1": True,
                "encoder_fine_tuned_after_phase1": True,
                "model_type": MODEL_TYPE,
                "fingerprint_source": self.fingerprint_source,
                "train_val_datasets": train_val_dbs,
                "test_dataset": test_db,
                "route": route_name,
                "fold": fold_idx + 1,
                "base_seed": self.random_seed,
                "cell_seed": cell_seed,
                "checkpoint_path": str(self.weights_path()),
                "checkpoint_sha256": self.input_hashes["weights_sha256"],
                "checkpoint_manifest_sha256": self.input_hashes["model_manifest_sha256"],
                "split_sha256": self.split_hashes,
                "config_path": str(self.config_path),
                "config_sha256": self.input_hashes["config_sha256"],
                "model_parameter_count": int(model.count_params()),
                "encoder_parameter_count": int(encoder.count_params()),
            }
            atomic_json(fold_dir / "experiment_metadata.json", metadata)

            history = self.train_model(model, MODEL_TYPE, arrays, fold_idx, fold_dir)
            if not model.get_layer("siamese_encoder").trainable:
                raise RuntimeError("The encoder was not unfrozen during DGR-MLP fine-tuning.")
            model.save(fold_dir / "model.keras")
            joblib.dump(arrays.y_scaler, fold_dir / "y_scaler.pkl")
            joblib.dump(arrays.adduct_encoder, fold_dir / "adduct_encoder.pkl")
            self.save_training_history(history, fold_idx, fold_dir)
            self.plot_loss_history(history, arrays.y_scaler, fold_idx, fold_dir, MODEL_TYPE)
            metrics = self.evaluate_experiment_fold(model, arrays, fold_idx, fold_dir)
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
            print(
                f"    Fold {fold_idx + 1} done -> MAE: {metrics['MAE']:.3f}, "
                f"R2: {metrics['R2']:.3f}"
            )
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

    def evaluate_experiment_fold(
        self, model, arrays: FoldArrays, fold_idx: int, fold_dir: Path
    ) -> dict[str, float]:
        pred_scaled = model.predict(
            self.training_inputs(arrays, "test", MODEL_TYPE), batch_size=512, verbose=0
        ).flatten()
        y_pred = arrays.y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        y_test = arrays.y_scaler.inverse_transform(
            arrays.y_test_scaled.reshape(-1, 1)
        ).flatten()
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
        self.plot_scatter(y_test, y_pred, fold_idx, fold_dir, MODEL_TYPE)
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
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(f".{os.getpid()}.tmp")
        frame.to_csv(temporary, index=False, float_format="%.6f", quoting=csv.QUOTE_MINIMAL)
        temporary.replace(output_path)
        print(f"  Route metrics ({len(frame)}/{self.folds} folds) -> {output_path}")
        return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune the HMDB-pretrained wide-only Siamese encoder with DGR-MLP."
    )
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
        WideOnlyPretrainedDGRRunner(
            fingerprint_source=source,
            config_path=args.config,
            folds=args.folds,
            random_seed=args.random_seed,
            routes=args.routes,
            resume=args.resume,
        ).run()


if __name__ == "__main__":
    main()
