"""Pretrain one Siamese encoder on a reproducible 90/10 HMDB split.

The canonical source-specific pretraining entry points reuse this mixin together
with their existing model, losses, pair generation, normalization, and callbacks.
It replaces the superclass-grouped cross-validation procedure with one
molecule-level stratified training/validation split.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import pandas as pd
from keras import callbacks
from sklearn.model_selection import StratifiedShuffleSplit

MONITOR_METRIC = "val_loss"
MONITOR_MODE = "min"


class HMDBPretrainingMixin:
    """Replace five-fold pretraining with one stratified HMDB split."""

    fingerprint_source: str
    pair_dataset_class: type
    epoch_logger_class: type

    def __init__(self, config_path: str | Path):
        super().__init__(config_path=config_path)
        self.split_cfg = self.config.get("split", {})
        self.pairs_cfg = self.config.get("pairs", {})
        self.train_fraction = float(self.split_cfg.get("train_fraction", 0.90))
        self.validation_fraction = float(self.split_cfg.get("validation_fraction", 0.10))
        self.stratify_by = str(self.split_cfg.get("stratify_by", "classification"))
        self.train_pairs = int(self.pairs_cfg.get("train_pairs", 100_000))
        self.validation_pairs = int(self.pairs_cfg.get("validation_pairs", 10_000))
        self._validate_pretraining_config()

    def _validate_pretraining_config(self) -> None:
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("split.train_fraction must be between 0 and 1.")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("split.validation_fraction must be between 0 and 1.")
        if not np.isclose(self.train_fraction + self.validation_fraction, 1.0):
            raise ValueError(
                "split.train_fraction and split.validation_fraction must sum to 1.0 "
                "for HMDB 90/10 pretraining."
            )
        for name, value in (
            ("pairs.train_pairs", self.train_pairs),
            ("pairs.validation_pairs", self.validation_pairs),
        ):
            if value <= 0 or value % 2:
                raise ValueError(f"{name} must be a positive even integer.")

    def run(self) -> None:
        self._write_model_manifest()
        df = self.load_data()
        train_df, validation_df = self.make_or_load_split(df)
        self.train_single_encoder(train_df, validation_df)

    def make_or_load_split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if "inchi" not in df.columns:
            raise ValueError("HMDB dataframe must contain an 'inchi' column.")
        if self.stratify_by not in df.columns:
            raise ValueError(
                f"Configured stratification column '{self.stratify_by}' is missing from HMDB."
            )
        if df["inchi"].duplicated().any():
            raise ValueError("HMDB pretraining requires one row per InChI.")

        split_dir = self.results_dir / "split"
        split_dir.mkdir(parents=True, exist_ok=True)
        train_path = split_dir / "train_inchi.csv"
        validation_path = split_dir / "validation_inchi.csv"
        manifest_path = split_dir / "split_manifest.json"

        if train_path.exists() or validation_path.exists() or manifest_path.exists():
            if not (train_path.exists() and validation_path.exists() and manifest_path.exists()):
                raise RuntimeError(
                    f"Incomplete persisted split in {split_dir}; use a new training.results_name."
                )
            return self._load_persisted_split(
                df,
                train_path=train_path,
                validation_path=validation_path,
                manifest_path=manifest_path,
            )

        splitter = StratifiedShuffleSplit(
            n_splits=1,
            train_size=self.train_fraction,
            test_size=self.validation_fraction,
            random_state=self.random_seed,
        )
        train_idx, validation_idx = next(
            splitter.split(df, df[self.stratify_by].astype(str))
        )
        train_df = df.iloc[train_idx].reset_index(drop=True)
        validation_df = df.iloc[validation_idx].reset_index(drop=True)
        self._validate_split_membership(df, train_df, validation_df)

        train_df[["inchi"]].to_csv(train_path, index=False)
        validation_df[["inchi"]].to_csv(validation_path, index=False)
        manifest = self._split_manifest(df, train_df, validation_df)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        self._print_split_summary(train_df, validation_df)
        return train_df, validation_df

    def _load_persisted_split(
        self,
        df: pd.DataFrame,
        train_path: Path,
        validation_path: Path,
        manifest_path: Path,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_hash = self._inchi_hash(df["inchi"])
        if manifest.get("dataset_inchi_sha256") != expected_hash:
            raise RuntimeError(
                "The persisted split was created from a different HMDB molecule set; "
                "use a new training.results_name."
            )
        expected_settings = {
            "random_seed": self.random_seed,
            "train_fraction_requested": self.train_fraction,
            "validation_fraction_requested": self.validation_fraction,
            "stratify_by": self.stratify_by,
            "fingerprint_source": self.fingerprint_source,
        }
        for key, expected in expected_settings.items():
            if manifest.get(key) != expected:
                raise RuntimeError(
                    f"Persisted split setting '{key}' is {manifest.get(key)!r}, "
                    f"but the current configuration requests {expected!r}. "
                    "Use a new training.results_name."
                )

        train_inchis = pd.read_csv(train_path, usecols=["inchi"])["inchi"].astype(str)
        validation_inchis = pd.read_csv(validation_path, usecols=["inchi"])["inchi"].astype(str)
        by_inchi = df.set_index("inchi", drop=False)
        missing_train = set(train_inchis) - set(by_inchi.index)
        missing_validation = set(validation_inchis) - set(by_inchi.index)
        if missing_train or missing_validation:
            raise RuntimeError("Persisted split contains InChI values absent from the current HMDB data.")

        train_df = by_inchi.loc[train_inchis.tolist()].reset_index(drop=True)
        validation_df = by_inchi.loc[validation_inchis.tolist()].reset_index(drop=True)
        self._validate_split_membership(df, train_df, validation_df)
        self._print_split_summary(train_df, validation_df, reused=True)
        return train_df, validation_df

    @staticmethod
    def _validate_split_membership(
        df: pd.DataFrame,
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
    ) -> None:
        train_inchis = set(train_df["inchi"].astype(str))
        validation_inchis = set(validation_df["inchi"].astype(str))
        all_inchis = set(df["inchi"].astype(str))
        if train_inchis & validation_inchis:
            raise RuntimeError("Training and validation InChI sets overlap.")
        if train_inchis | validation_inchis != all_inchis:
            raise RuntimeError("Training and validation do not cover the complete HMDB molecule set.")

    def _split_manifest(
        self,
        df: pd.DataFrame,
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
    ) -> dict[str, Any]:
        total = len(df)
        return {
            "strategy": "stratified_hmdb_90_10",
            "fingerprint_source": self.fingerprint_source,
            "random_seed": self.random_seed,
            "stratify_by": self.stratify_by,
            "train_fraction_requested": self.train_fraction,
            "validation_fraction_requested": self.validation_fraction,
            "total_molecules": total,
            "train_molecules": len(train_df),
            "validation_molecules": len(validation_df),
            "train_fraction_actual": len(train_df) / total,
            "validation_fraction_actual": len(validation_df) / total,
            "dataset_inchi_sha256": self._inchi_hash(df["inchi"]),
            "train_class_counts": self._class_counts(train_df),
            "validation_class_counts": self._class_counts(validation_df),
            "train_pairs": self.train_pairs,
            "validation_pairs": self.validation_pairs,
            "validation_usage": "training_control_only_early_stopping_lr_schedule_and_checkpoint",
            "test_partition": None,
        }

    def _class_counts(self, df: pd.DataFrame) -> dict[str, int]:
        counts = df[self.stratify_by].astype(str).value_counts().sort_index()
        return {str(label): int(count) for label, count in counts.items()}

    @staticmethod
    def _inchi_hash(inchis: pd.Series) -> str:
        digest = hashlib.sha256()
        for inchi in sorted(inchis.astype(str)):
            digest.update(inchi.encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()

    @staticmethod
    def _print_split_summary(
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        reused: bool = False,
    ) -> None:
        total = len(train_df) + len(validation_df)
        action = "Reused" if reused else "Created"
        print(
            f"{action} HMDB 90/10 split | "
            f"train={len(train_df):,} ({len(train_df) / total:.2%}) | "
            f"validation={len(validation_df):,} ({len(validation_df) / total:.2%})"
        )

    def train_single_encoder(
        self,
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
    ) -> None:
        stats_by_name = {}
        if self.use_logp:
            stats_by_name["logp"] = self._fit_task_stats(train_df["logp"])
        if self.use_molvol:
            stats_by_name["molvol"] = self._fit_task_stats(
                train_df["mol_volume_mean"].dropna()
            )

        train_scaled = self._apply_normalization(train_df, stats_by_name)
        validation_scaled = self._apply_normalization(validation_df, stats_by_name)
        fingerprint_columns = self.fingerprint_columns(train_df)
        fingerprint_dim = len(fingerprint_columns)

        x1_train, x2_train, train_targets, train_weights = self.generate_pairs(
            train_scaled,
            fingerprint_columns,
            self.train_pairs,
        )
        train_dataset = self.pair_dataset_class(
            x1_train,
            x2_train,
            train_targets,
            sample_weights=train_weights,
            batch_size=self.batch_size,
        )
        x1_validation, x2_validation, validation_targets, validation_weights = self.generate_pairs(
            validation_scaled,
            fingerprint_columns,
            self.validation_pairs,
        )

        model = self.build_model(fingerprint_dim)
        self.compile_model(model)
        fold_dir = self.results_dir / "fold_1"
        fold_dir.mkdir(parents=True, exist_ok=True)
        weights_path = fold_dir / "best.weights.h5"
        epoch_logger = self.epoch_logger_class(
            use_logp=self.use_logp,
            use_molvol=self.use_molvol,
        )
        training_callbacks = [
            epoch_logger,
            callbacks.ModelCheckpoint(
                str(weights_path),
                save_best_only=True,
                save_weights_only=True,
                monitor=MONITOR_METRIC,
                mode=MONITOR_MODE,
            ),
            callbacks.EarlyStopping(
                monitor=MONITOR_METRIC,
                mode=MONITOR_MODE,
                min_delta=float(self.training_cfg.get("early_stopping_min_delta", 0.000002)),
                patience=int(self.training_cfg.get("early_stopping_patience", 10)),
                restore_best_weights=True,
            ),
            callbacks.ReduceLROnPlateau(
                monitor=MONITOR_METRIC,
                factor=float(self.training_cfg.get("reduce_lr_factor", 0.3333333)),
                patience=int(self.training_cfg.get("reduce_lr_patience", 4)),
                min_delta=float(self.training_cfg.get("early_stopping_min_delta", 0.000002)),
                verbose=1,
            ),
        ]

        history = model.fit(
            train_dataset,
            epochs=int(self.training_cfg.get("max_epochs", 2000)),
            validation_data=(
                (x1_validation, x2_validation),
                validation_targets,
                validation_weights,
            ),
            callbacks=training_callbacks,
            verbose=0,
        )
        history.history["epoch_time"] = epoch_logger.epoch_times
        self._save_training_log(history, 1, fold_dir)
        self._plot_loss_history(history, 1, fold_dir)
        self._write_training_summary(
            history=history.history,
            train_df=train_df,
            validation_df=validation_df,
            stats_by_name=stats_by_name,
            weights_path=weights_path,
            fold_dir=fold_dir,
        )

    def _write_training_summary(
        self,
        history: dict[str, list[Any]],
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        stats_by_name: dict[str, Any],
        weights_path: Path,
        fold_dir: Path,
    ) -> None:
        validation_loss = np.asarray(history.get("val_loss", []), dtype=np.float64)
        best_index = int(np.argmin(validation_loss)) if validation_loss.size else None
        payload = {
            "mode": "single_hmdb_pretraining_split",
            "fingerprint_source": self.fingerprint_source,
            "train_molecules": len(train_df),
            "validation_molecules": len(validation_df),
            "train_pairs": self.train_pairs,
            "validation_pairs": self.validation_pairs,
            "epochs_completed": len(history.get("loss", [])),
            "best_epoch": best_index + 1 if best_index is not None else None,
            "best_validation_loss": (
                float(validation_loss[best_index]) if best_index is not None else None
            ),
            "weights_path": str(weights_path),
            "task_normalization": {
                name: {"mean": float(stats.mean), "std": float(stats.std)}
                for name, stats in stats_by_name.items()
            },
            "validation_usage": "training_control_only_early_stopping_lr_schedule_and_checkpoint",
        }
        (fold_dir / "training_summary.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
