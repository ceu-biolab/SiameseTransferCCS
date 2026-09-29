from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("KERAS_BACKEND", "torch")

import joblib
import keras
import numpy as np
import pandas as pd
import yaml
from keras import callbacks, layers, models
from keras.optimizers import RMSprop
from matplotlib import pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.data import load_ccsbase, load_metlinccs
from src.data.load_ccsbase import (
    CCSBASE_DESCRIPTORS_CSV,
    MOLECULAR_FEATURE_COLUMNS,
    MOLECULAR_FEATURE_SCALES,
    merge_physchem_descriptors,
)
from src.data.load_metlinccs import METLINCCS_DESCRIPTORS_CSV
from src.models.metrics import mspe_percent


DEFAULT_CONFIG_PATH = Path("configs/ccs_prediction_heads.yaml")
RESULTS_ROOT = Path("results/CCSTrainer")
VALID_FINGERPRINT_SOURCES = {"alvadesc", "rdkit"}
CCSBASE_RDKIT_CSV = Path("resources/fingerprints/ccsbase_cleaned_rdkit.csv")
METLINCCS_RDKIT_CSV = Path("resources/fingerprints/metlinccs_cleaned_rdkit.csv")
MODEL_TYPES = ("linear_regression_fingerprints_only", "linear_regression", "gated_residual_mlp")
MODEL_DISPLAY_NAMES = {
    "linear_regression_fingerprints_only": "Linear regression fingerprints only",
    "linear_regression": "Linear regression",
    "gated_residual_mlp": "Gated residual MLP",
}
DATASET_CONFIGS = (
    (("ccsbase",), "ccsbase"),
    (("ccsbase",), "metlinccs"),
    (("metlinccs",), "ccsbase"),
    (("metlinccs",), "metlinccs"),
)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        keras.utils.set_random_seed(seed)
    except Exception:
        pass


@contextmanager
def locked_split_file(splits_path: Path):
    lock_path = splits_path.with_suffix(splits_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@dataclass
class FoldArrays:
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    adducts_train: np.ndarray
    adducts_val: np.ndarray
    adducts_test: np.ndarray
    aux_train: np.ndarray
    aux_val: np.ndarray
    aux_test: np.ndarray
    y_train_scaled: np.ndarray
    y_val_scaled: np.ndarray
    y_test_scaled: np.ndarray
    y_scaler: RobustScaler
    adduct_encoder: OneHotEncoder
    aux_metadata: dict[str, Any]


class FingerprintCCSBaselineRunner:
    """Run CCS prediction directly on molecular fingerprints."""

    def __init__(
        self,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        folds: int = 5,
        random_seed: int = 42,
        fingerprint_source: str = "rdkit",
        experiment_tag: str = "",
    ):
        if folds < 1 or folds > 5:
            raise ValueError("--folds must be between 1 and 5.")
        if fingerprint_source not in VALID_FINGERPRINT_SOURCES:
            raise ValueError(
                f"Invalid fingerprint_source='{fingerprint_source}'. "
                f"Expected one of: {sorted(VALID_FINGERPRINT_SOURCES)}"
            )
        self.config_path = Path(config_path)
        self.config = self._with_defaults(load_yaml(self.config_path))
        self.training_cfg = self.config["training"]
        self.mass_cfg = self.config["mass"]
        self.molecular_features_cfg = self.config["molecular_features"]
        self.folds = int(folds)
        self.random_seed = int(random_seed)
        self.fingerprint_source = fingerprint_source
        self.experiment_tag = str(experiment_tag).strip()
        self.use_mass = bool(self.mass_cfg.get("enabled", False))
        self.use_molecular_features = bool(self.molecular_features_cfg.get("enabled", False))
        self.molecular_feature_columns = list(
            self.molecular_features_cfg.get("features", MOLECULAR_FEATURE_COLUMNS)
        )
        self.molecular_feature_scales = dict(
            self.molecular_features_cfg.get("scale", MOLECULAR_FEATURE_SCALES)
        )
        self.mass_scale = float(self.mass_cfg.get("scale", 100.0))
        self.mass_column = self.mass_cfg.get("column")
        self._molecular_weight_cache: dict[str, float] = {}
        self.results_root = RESULTS_ROOT
        self.splits_dir = self.results_root / "splits"
        self.results_root.mkdir(parents=True, exist_ok=True)
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        set_seed(self.random_seed)

    @staticmethod
    def _with_defaults(config: dict[str, Any]) -> dict[str, Any]:
        defaults = {
            "mass": {
                "enabled": False,
                "column": None,
                "scale": 100.0,
            },
            "molecular_features": {
                "enabled": False,
                "descriptor_csvs": {
                    "ccsbase": "resources/descriptors/ccsbase_physchem.csv",
                    "metlinccs": "resources/descriptors/metlinccs_physchem.csv",
                },
                "features": MOLECULAR_FEATURE_COLUMNS,
                "scale": MOLECULAR_FEATURE_SCALES,
                "missing": {
                    "add_indicators": True,
                },
            },
            "heads": {
                "gated_residual_mlp": {
                    "hidden_dim": 256,
                    "num_blocks": 3,
                    "dropout": 0.1,
                    "activation": "gelu",
                    "residual_zero_init": True,
                },
            },
            "training": {
                "batch_size": 32,
                "phase1_learning_rate": 1e-3,
                "phase2_learning_rate": 1e-4,
                "fine_tune_learning_rate": 1e-5,
                "phase1_epochs": 10000,
                "phase2_epochs": 10000,
                "fine_tune_epochs": 20,
                "early_stopping_patience": 20,
                "reduce_lr_patience": 5,
                "reduce_lr_factor": 1 / 3,
                "reduce_lr_min_delta": 1e-5,
            },
        }
        return deep_update(defaults, config)

    def run(self) -> None:
        print(f"CCS fingerprints runner | config={self.config_path} | folds={self.folds}")
        print(f"Fingerprint source: {self.fingerprint_source}")
        print(f"Models: {', '.join(MODEL_DISPLAY_NAMES[m] for m in MODEL_TYPES)}")
        print("CCS loss: MAE")
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
                for model_type in MODEL_TYPES:
                    if model_type == "gated_residual_mlp" and not self.aux_features_enabled():
                        print(
                            "  Skipping gated_residual_mlp: it requires mass.enabled=true "
                            "or molecular_features.enabled=true."
                        )
                        continue
                    self.train_one_fold(train_df, val_df, test_df, fold_idx, model_type, train_val_dbs, test_db)

            self.write_final_metrics(train_val_dbs, test_db)

    def load_data(self, db_names: list[str]) -> pd.DataFrame:
        dfs = []
        for db_name in db_names:
            if self.fingerprint_source == "rdkit":
                df = self.load_rdkit_dataset(db_name)
            elif db_name == "ccsbase":
                df = load_ccsbase(
                    descriptor_csv=self.config["molecular_features"]["descriptor_csvs"].get("ccsbase"),
                    add_descriptors=self.use_molecular_features,
                )
            elif db_name == "metlinccs":
                df = load_metlinccs(
                    descriptor_csv=self.config["molecular_features"]["descriptor_csvs"].get("metlinccs"),
                    add_descriptors=self.use_molecular_features,
                )
            else:
                raise ValueError(f"Unknown CCS dataset: {db_name}")
            df = df.copy()
            df["dataset"] = db_name
            dfs.append(df)
        if not dfs:
            raise ValueError("No datasets requested.")
        return pd.concat(dfs, ignore_index=True)

    def load_rdkit_dataset(self, db_name: str) -> pd.DataFrame:
        if db_name == "ccsbase":
            path = CCSBASE_RDKIT_CSV
            descriptor_path = self.config["molecular_features"]["descriptor_csvs"].get(
                "ccsbase",
                str(CCSBASE_DESCRIPTORS_CSV),
            )
            merge_descriptors = merge_physchem_descriptors
        elif db_name == "metlinccs":
            path = METLINCCS_RDKIT_CSV
            descriptor_path = self.config["molecular_features"]["descriptor_csvs"].get(
                "metlinccs",
                str(METLINCCS_DESCRIPTORS_CSV),
            )
            merge_descriptors = merge_physchem_descriptors
        else:
            raise ValueError(f"Unknown CCS dataset: {db_name}")

        if not path.exists():
            raise FileNotFoundError(
                f"RDKit fingerprint CSV not found: {path}. "
                "Run: python -m src.data.generate_rdkit_fingerprints"
            )

        df = pd.read_csv(path, low_memory=False)
        self.validate_fingerprint_dataframe(df, path)
        if self.use_molecular_features:
            df = merge_descriptors(df, descriptor_csv=descriptor_path)
        return df

    @staticmethod
    def validate_fingerprint_dataframe(df: pd.DataFrame, path: Path) -> None:
        required = {"inchi", "adduct", "ccs"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        fp_cols = [column for column in df.columns if column.startswith("V")]
        if not fp_cols:
            raise ValueError(f"{path} must contain fingerprint columns named V*.")

    def group_kfold_split_train_val_only(
        self,
        df: pd.DataFrame,
        config_key: str,
        n_splits: int = 5,
    ) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
        splits_path = self.splits_dir / f"{config_key}_trainval_splits.json"
        with locked_split_file(splits_path):
            if splits_path.exists():
                split_data = json.loads(splits_path.read_text(encoding="utf-8"))
                return (
                    [df.loc[item["train_idx"]].copy() for item in split_data],
                    [df.loc[item["val_idx"]].copy() for item in split_data],
                )

            groups = df.get("inchi", df.index.astype(str))
            train_dfs, val_dfs, split_data = [], [], []
            for train_idx, val_idx in GroupKFold(n_splits=n_splits).split(df, groups=groups):
                train_dfs.append(df.iloc[train_idx].copy())
                val_dfs.append(df.iloc[val_idx].copy())
                split_data.append({
                    "train_idx": df.index[train_idx].tolist(),
                    "val_idx": df.index[val_idx].tolist(),
                })
            splits_path.write_text(json.dumps(split_data), encoding="utf-8")
            return train_dfs, val_dfs

    def group_kfold_split(
        self,
        df: pd.DataFrame,
        config_key: str,
        n_splits: int = 5,
    ) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame]]:
        splits_path = self.splits_dir / f"{config_key}_splits.json"
        with locked_split_file(splits_path):
            if splits_path.exists():
                split_data = json.loads(splits_path.read_text(encoding="utf-8"))
                train_dfs = [df.loc[item["train_idx"]].copy() for item in split_data]
                val_dfs = [df.loc[item["val_idx"]].copy() for item in split_data]
                test_dfs = [df.loc[item["test_idx"]].copy() for item in split_data]
                if not self._split_has_overlap(train_dfs, val_dfs, test_dfs):
                    return train_dfs, val_dfs, test_dfs
                splits_path.unlink()

            groups = df.get("inchi", df.index.astype(str))
            train_dfs, val_dfs, test_dfs, split_data = [], [], [], []
            for train_val_idx, test_idx in GroupKFold(n_splits=n_splits).split(df, groups=groups):
                train_val_df = df.iloc[train_val_idx].copy()
                test_df = df.iloc[test_idx].copy()
                inner_groups = train_val_df.get("inchi", train_val_df.index.astype(str))
                train_idx, val_idx = next(GroupKFold(n_splits=5).split(train_val_df, groups=inner_groups))
                train_dfs.append(train_val_df.iloc[train_idx].copy())
                val_dfs.append(train_val_df.iloc[val_idx].copy())
                test_dfs.append(test_df)
                split_data.append({
                    "train_idx": train_val_df.index[train_idx].tolist(),
                    "val_idx": train_val_df.index[val_idx].tolist(),
                    "test_idx": df.index[test_idx].tolist(),
                })
            splits_path.write_text(json.dumps(split_data), encoding="utf-8")
            return train_dfs, val_dfs, test_dfs

    @staticmethod
    def _split_has_overlap(train_dfs, val_dfs, test_dfs) -> bool:
        return any(
            (set(train.index) & set(val.index))
            or (set(train.index) & set(test.index))
            or (set(val.index) & set(test.index))
            for train, val, test in zip(train_dfs, val_dfs, test_dfs)
        )

    def train_one_fold(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        fold_idx: int,
        model_type: str,
        train_val_dbs: list[str],
        test_db: str,
    ) -> None:
        display_name = MODEL_DISPLAY_NAMES[model_type]
        print(
            f"  Training {display_name} | fingerprints | fold {fold_idx + 1} "
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
        (fold_dir / "fingerprint_baseline_metadata.json").write_text(
            json.dumps(
                {
                    "mode": "fingerprint_ccs_baseline",
                    "fingerprint_source": self.fingerprint_source,
                    "experiment_tag": self.experiment_tag,
                    "ccs_loss": "mae",
                    "uses_aux_features": self.uses_aux_features(model_type),
                    "use_mass": self.use_mass,
                    "use_molecular_features": self.use_molecular_features,
                },
                indent=2,
            ),
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

    def prepare_fold_arrays(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> FoldArrays:
        fp_cols = [column for column in train_df.columns if column.startswith("V")]
        if not fp_cols:
            raise ValueError("No fingerprint columns named V* were found.")

        x_train = train_df[fp_cols].to_numpy(dtype=np.float32)
        x_val = val_df[fp_cols].to_numpy(dtype=np.float32)
        x_test = test_df[fp_cols].to_numpy(dtype=np.float32)
        x_train = np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0)
        x_val = np.nan_to_num(x_val, nan=0.0, posinf=0.0, neginf=0.0)
        x_test = np.nan_to_num(x_test, nan=0.0, posinf=0.0, neginf=0.0)

        aux_train, aux_val, aux_test, aux_metadata = self.prepare_aux_features(train_df, val_df, test_df)

        y_scaler = RobustScaler()
        y_train_scaled = y_scaler.fit_transform(train_df["ccs"].to_numpy(dtype=np.float32).reshape(-1, 1)).flatten()
        y_val_scaled = y_scaler.transform(val_df["ccs"].to_numpy(dtype=np.float32).reshape(-1, 1)).flatten()
        y_test_scaled = y_scaler.transform(test_df["ccs"].to_numpy(dtype=np.float32).reshape(-1, 1)).flatten()

        adduct_encoder = OneHotEncoder(sparse_output=False)
        adducts_train = adduct_encoder.fit_transform(train_df["adduct"].values.reshape(-1, 1))
        adducts_val = adduct_encoder.transform(val_df["adduct"].values.reshape(-1, 1))
        adducts_test = adduct_encoder.transform(test_df["adduct"].values.reshape(-1, 1))

        return FoldArrays(
            x_train=x_train,
            x_val=x_val,
            x_test=x_test,
            adducts_train=adducts_train.astype(np.float32),
            adducts_val=adducts_val.astype(np.float32),
            adducts_test=adducts_test.astype(np.float32),
            aux_train=aux_train,
            aux_val=aux_val,
            aux_test=aux_test,
            y_train_scaled=y_train_scaled.astype(np.float32),
            y_val_scaled=y_val_scaled.astype(np.float32),
            y_test_scaled=y_test_scaled.astype(np.float32),
            y_scaler=y_scaler,
            adduct_encoder=adduct_encoder,
            aux_metadata=aux_metadata,
        )

    def prepare_aux_features(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        if self.use_molecular_features:
            return self.prepare_molecular_aux_features(train_df, val_df, test_df)
        if self.use_mass:
            return (
                self.get_mass_features(train_df),
                self.get_mass_features(val_df),
                self.get_mass_features(test_df),
                {"mode": "mass", "columns": ["MolWt_scaled"], "scale": {"MolWt": self.mass_scale}},
            )
        empty = (
            np.zeros((len(train_df), 0), dtype=np.float32),
            np.zeros((len(val_df), 0), dtype=np.float32),
            np.zeros((len(test_df), 0), dtype=np.float32),
        )
        return (*empty, {"mode": "none", "columns": []})

    def prepare_molecular_aux_features(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        feature_cols = self.molecular_feature_columns
        missing_cfg = self.molecular_features_cfg.get("missing", {})
        add_indicators = bool(missing_cfg.get("add_indicators", True))
        scale_cfg = self.molecular_feature_scales

        train_raw = train_df[feature_cols].apply(pd.to_numeric, errors="coerce")
        val_raw = val_df[feature_cols].apply(pd.to_numeric, errors="coerce")
        test_raw = test_df[feature_cols].apply(pd.to_numeric, errors="coerce")

        fill_values = train_raw.median(axis=0, skipna=True).reindex(feature_cols).fillna(0.0).astype(float)

        scales = {}
        for column in feature_cols:
            scale = float(scale_cfg.get(column, 1.0))
            if scale == 0:
                raise ValueError(f"molecular_features.scale.{column} cannot be 0.")
            scales[column] = scale

        def transform(raw: pd.DataFrame) -> np.ndarray:
            missing = raw.isna().astype(np.float32)
            filled = raw.fillna(fill_values)
            scaled = filled.copy()
            for column, scale in scales.items():
                scaled[column] = scaled[column].astype(np.float32) / np.float32(scale)
            parts = [scaled[feature_cols].to_numpy(dtype=np.float32)]
            if add_indicators:
                parts.append(missing[feature_cols].to_numpy(dtype=np.float32))
            return np.concatenate(parts, axis=1).astype(np.float32)

        output_columns = list(feature_cols)
        if add_indicators:
            output_columns += [f"{column}_missing" for column in feature_cols]
        metadata = {
            "mode": "molecular_features",
            "raw_feature_columns": list(feature_cols),
            "columns": output_columns,
            "missing_strategy": "median_train",
            "add_missing_indicators": add_indicators,
            "fill_values": {column: float(fill_values[column]) for column in feature_cols},
            "scale": scales,
        }
        return transform(train_raw), transform(val_raw), transform(test_raw), metadata

    def get_mass_features(self, df: pd.DataFrame) -> np.ndarray:
        if self.mass_scale == 0:
            raise ValueError("mass.scale cannot be 0.")
        mass_series = self.resolve_mass_series(df)
        missing = mass_series.isna()
        if missing.any():
            examples = df.loc[missing, "inchi"].head(5).astype(str).tolist() if "inchi" in df.columns else []
            raise ValueError(f"Could not resolve molecular mass for {int(missing.sum())} rows. Examples: {examples}")
        return (mass_series.astype(np.float32).to_numpy().reshape(-1, 1) / np.float32(self.mass_scale)).astype(np.float32)

    def resolve_mass_series(self, df: pd.DataFrame) -> pd.Series:
        if self.mass_column:
            if self.mass_column not in df.columns:
                raise ValueError(f"Configured mass.column='{self.mass_column}' is not present in the dataframe.")
            return pd.to_numeric(df[self.mass_column], errors="coerce")
        for column in ("molecular_weight", "mol_weight", "mol_wt", "exact_mass", "monoisotopic_mass", "mass", "mw", "MolWt"):
            if column in df.columns:
                return pd.to_numeric(df[column], errors="coerce")
        if "inchi" in df.columns:
            return df["inchi"].apply(self.molecular_weight_from_inchi)
        raise ValueError("Molecular mass is enabled, but no mass column or InChI column is available.")

    def molecular_weight_from_inchi(self, inchi: str) -> float:
        if pd.isna(inchi) or not isinstance(inchi, str) or not inchi.strip():
            return np.nan
        key = inchi.strip()
        if key in self._molecular_weight_cache:
            return self._molecular_weight_cache[key]
        try:
            RDLogger.DisableLog("rdApp.*")
            mol = Chem.MolFromInchi(key)
            value = float(Descriptors.MolWt(mol)) if mol is not None else np.nan
        except Exception:
            value = np.nan
        self._molecular_weight_cache[key] = value
        return value

    def build_model(self, model_type: str, fp_dim: int, adduct_dim: int, aux_dim: int = 1):
        if model_type in {"linear_regression_fingerprints_only", "linear_regression"}:
            return self.build_linear_regression(fp_dim, adduct_dim, aux_dim, model_type=model_type)
        if model_type == "gated_residual_mlp":
            return self.build_gated_residual_mlp(fp_dim, adduct_dim, aux_dim)
        raise ValueError(f"Unknown model_type='{model_type}'. Expected one of: {MODEL_TYPES}")

    def build_linear_regression(
        self,
        fp_dim: int,
        adduct_dim: int,
        aux_dim: int = 1,
        model_type: str = "linear_regression",
    ):
        fp_input, adduct_input, aux_input = self.make_inputs(fp_dim, adduct_dim, aux_dim)
        x = self.concat_inputs(fp_input, adduct_input, aux_input, use_aux=self.uses_aux_features(model_type))
        output = layers.Dense(1, activation="linear")(x)
        return models.Model(self.model_inputs(fp_input, adduct_input, aux_input, model_type), output)

    def build_gated_residual_mlp(self, fp_dim: int, adduct_dim: int, aux_dim: int = 1):
        if not self.aux_features_enabled():
            raise ValueError("gated_residual_mlp requires mass.enabled=true or molecular_features.enabled=true.")
        fp_input, adduct_input, aux_input = self.make_inputs(fp_dim, adduct_dim, aux_dim)
        feature_dim = int(fp_input.shape[-1])
        gamma = layers.Dense(feature_dim, activation="linear", kernel_initializer="zeros", bias_initializer="zeros", name="aux_gamma")(aux_input)
        beta = layers.Dense(feature_dim, activation="linear", kernel_initializer="zeros", bias_initializer="zeros", name="aux_beta")(aux_input)
        scaled_delta = layers.Multiply(name="aux_feature_interaction")([fp_input, gamma])
        gated_features = layers.Add(name="aux_gated_features")([fp_input, scaled_delta, beta])
        x = self.concat_inputs(gated_features, adduct_input, aux_input, use_aux=True)
        output = self.build_residual_core(x, self.config.get("heads", {}).get("gated_residual_mlp", {}))
        return models.Model(self.model_inputs(fp_input, adduct_input, aux_input, "gated_residual_mlp"), output)

    @staticmethod
    def make_inputs(fp_dim: int, adduct_dim: int, aux_dim: int):
        fp_input = layers.Input(shape=(fp_dim,), name="fingerprints")
        adduct_input = layers.Input(shape=(adduct_dim,), name="adduct")
        aux_input = layers.Input(shape=(aux_dim,), name="aux_features")
        return fp_input, adduct_input, aux_input

    def model_inputs(self, fp_input, adduct_input, aux_input, model_type: str):
        if self.uses_aux_features(model_type):
            return [fp_input, adduct_input, aux_input]
        return [fp_input, adduct_input]

    def training_inputs(self, arrays: FoldArrays, split: str, model_type: str):
        x = getattr(arrays, f"x_{split}")
        adducts = getattr(arrays, f"adducts_{split}")
        aux = getattr(arrays, f"aux_{split}")
        if self.uses_aux_features(model_type):
            return [x, adducts, aux]
        return [x, adducts]

    def uses_aux_features(self, model_type: str) -> bool:
        if model_type == "linear_regression_fingerprints_only":
            return False
        return self.aux_features_enabled() or model_type == "gated_residual_mlp"

    def aux_features_enabled(self) -> bool:
        return self.use_mass or self.use_molecular_features

    @staticmethod
    def concat_inputs(features, adduct_input, aux_input, use_aux: bool):
        tensors = [features, adduct_input]
        if use_aux:
            tensors.append(aux_input)
        return layers.Concatenate()(tensors)

    @staticmethod
    def activation(name: str):
        return name or "relu"

    def residual_block(self, x, hidden_dim: int, activation: str, dropout: float, block_idx: int):
        shortcut = x
        y = layers.Dense(
            hidden_dim,
            activation=self.activation(activation),
            kernel_initializer="he_normal",
            name=f"res_block_{block_idx}_dense_1",
        )(x)
        if dropout > 0:
            y = layers.Dropout(dropout, name=f"res_block_{block_idx}_dropout")(y)
        y = layers.Dense(
            hidden_dim,
            activation=None,
            kernel_initializer="he_normal",
            name=f"res_block_{block_idx}_dense_2",
        )(y)
        y = layers.Add(name=f"res_block_{block_idx}_add")([shortcut, y])
        return layers.Activation(self.activation(activation), name=f"res_block_{block_idx}_activation")(y)

    def build_residual_core(self, x, head_cfg: dict[str, Any]):
        hidden_dim = int(head_cfg.get("hidden_dim", 256))
        num_blocks = int(head_cfg.get("num_blocks", 3))
        dropout = float(head_cfg.get("dropout", 0.1))
        activation = head_cfg.get("activation", "gelu")
        zero_init = bool(head_cfg.get("residual_zero_init", True))
        skip = layers.Dense(1, activation="linear", name="linear_skip")(x)
        residual = layers.Dense(
            hidden_dim,
            activation=self.activation(activation),
            kernel_initializer="he_normal",
            name="residual_projection",
        )(x)
        for block_idx in range(num_blocks):
            residual = self.residual_block(residual, hidden_dim, activation, dropout, block_idx + 1)
        final_initializer = "zeros" if zero_init else "glorot_uniform"
        residual = layers.Dense(1, activation="linear", kernel_initializer=final_initializer, name="residual_output")(residual)
        return layers.Add(name="linear_plus_residual")([skip, residual])

    def train_model(self, model, model_type: str, arrays: FoldArrays, fold_idx: int, fold_dir: Path) -> dict[str, list[float]]:
        fold_dir.mkdir(parents=True, exist_ok=True)
        batch_size = int(self.training_cfg.get("batch_size", 32))
        phase1_epochs = int(self.training_cfg.get("phase1_epochs", 10000))
        phase2_epochs = int(self.training_cfg.get("phase2_epochs", 10000))
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
        for key in set(history_phase1.history) | set(history_phase2.history):
            history[key] = history_phase1.history.get(key, []) + history_phase2.history.get(key, [])
        return {key: [float(value) for value in values] for key, values in history.items()}

    def evaluate_fold(
        self,
        model,
        arrays: FoldArrays,
        fold_idx: int,
        fold_dir: Path,
        subfolder: str,
        model_type: str,
    ) -> dict[str, float]:
        pred_scaled = model.predict(self.training_inputs(arrays, "test", model_type), batch_size=512, verbose=0).flatten()
        y_pred = arrays.y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        y_test = arrays.y_scaler.inverse_transform(arrays.y_test_scaled.reshape(-1, 1)).flatten()

        mae = mean_absolute_error(y_test, y_pred)
        medae = median_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        mspe = mspe_percent(y_test, y_pred)
        medape = np.median(np.abs((y_test - y_pred) / np.clip(y_test, 1e-8, None))) * 100
        r2 = r2_score(y_test, y_pred)

        csv_path = fold_dir.parent / f"{subfolder}_results.csv"
        write_header = not csv_path.exists()
        with csv_path.open("a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["Fold", "MAE", "MedAE", "MSE", "MAPE(%)", "MedAPE(%)", "MSPE(%)", "R2"])
            writer.writerow([
                fold_idx + 1,
                f"{mae:.3f}",
                f"{medae:.3f}",
                f"{mse:.3f}",
                f"{mape:.3f}",
                f"{medape:.3f}",
                f"{mspe:.3f}",
                f"{r2:.3f}",
            ])

        print(f"    Fold {fold_idx + 1} done -> MAE: {mae:.3f}, R2: {r2:.3f}")
        self.plot_scatter(y_test, y_pred, fold_idx, fold_dir, model_type)
        return {"mae": mae, "medae": medae, "mse": mse, "mape": mape, "medape": medape, "mspe": mspe, "r2": r2}

    @staticmethod
    def save_training_history(history: dict[str, list[float]], fold_idx: int, fold_dir: Path) -> None:
        with (fold_dir / f"training_history_fold_{fold_idx + 1}.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    def plot_loss_history(self, history: dict[str, list[float]], y_scaler: RobustScaler, fold_idx: int, fold_dir: Path, model_type: str) -> None:
        train_loss_scaled = np.asarray(history["loss"], dtype=np.float32)
        val_loss_scaled = np.asarray(history.get("val_loss", []), dtype=np.float32)
        scale_factor = y_scaler.scale_[0]
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss_scaled * scale_factor, label="Training Loss", linewidth=2)
        if len(val_loss_scaled):
            plt.plot(val_loss_scaled * scale_factor, label="Validation Loss", linewidth=2)
        plt.title(f"CCS Prediction - fingerprints + {MODEL_DISPLAY_NAMES[model_type]} - Fold {fold_idx + 1}")
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
            f"fingerprints + {MODEL_DISPLAY_NAMES[model_type]} - Fold {fold_idx + 1}\n"
            f"MAE: {mean_absolute_error(y_true, y_pred):.3f} | R2: {r2_score(y_true, y_pred):.3f}"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(fold_dir / f"scatter_fold_{fold_idx + 1}.png", dpi=300, bbox_inches="tight")
        plt.close()

    def write_final_metrics(self, train_val_dbs: list[str], test_db: str) -> None:
        for model_type in MODEL_TYPES:
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

    def result_subfolder(self, train_val_dbs: list[str], test_db: str, model_type: str) -> str:
        features_tag = "with_molfeatures" if self.use_molecular_features else "no_molfeatures"
        mass_tag = "with_mass" if self.use_mass else "no_mass"
        folds_tag = "single_fold" if self.folds == 1 else "five_folds"
        experiment_tag = f"{self.experiment_tag}_" if self.experiment_tag else ""
        return (
            f"train_val_{'_'.join(train_val_dbs)}_test_{test_db}_"
            f"{self.fingerprint_source}_fingerprints_{experiment_tag}{features_tag}_{mass_tag}_{model_type}_{folds_tag}"
        )

    def _split_key(self, train_val_dbs: list[str], test_db: str, same_database: bool) -> str:
        split_kind = "same_db" if same_database else "external_test"
        return (
            f"train_val_{'_'.join(train_val_dbs)}_test_{test_db}_"
            f"{self.fingerprint_source}_fingerprints_{split_kind}"
        )


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CCS prediction directly on fingerprints.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--folds", type=int, default=5, help="Number of generated folds to execute, from 1 to 5.")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--fingerprint-source", choices=sorted(VALID_FINGERPRINT_SOURCES), default="rdkit")
    parser.add_argument("--experiment-tag", default="", help="Optional tag inserted into output folder names.")
    args = parser.parse_args()
    FingerprintCCSBaselineRunner(
        config_path=args.config,
        folds=args.folds,
        random_seed=args.random_seed,
        fingerprint_source=args.fingerprint_source,
        experiment_tag=args.experiment_tag,
    ).run()


if __name__ == "__main__":
    main()
