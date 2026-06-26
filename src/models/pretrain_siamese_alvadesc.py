from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np
import pandas as pd
import yaml
from keras import callbacks, layers, ops, optimizers, regularizers
from keras.utils import PyDataset
from matplotlib import pyplot as plt
from rdkit import DataStructs
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

from src.data import load_hmdb


DEFAULT_CONFIG_PATH = Path("configs/pretrain_siamese_alvadesc.yaml")
RESULTS_ROOT = Path("results")
FINGERPRINT_DIM = 2214
WIDE_DIM = 1536
DEEP_DIM = 512
BASE_EMBEDDING_DIM = WIDE_DIM + DEEP_DIM
EMBEDDING_DIM = BASE_EMBEDDING_DIM
TRAIN_PAIRS = 100_000
VAL_PAIRS = 10_000
TEST_PAIRS = 100_000
LEARNING_RATE = 1e-4
RMSPROP_RHO = 0.7
RMSPROP_MOMENTUM = 0.7
MONITOR_METRIC = "val_loss"
MONITOR_MODE = "min"


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


def format_epoch_time(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(total_seconds, 60)
    return f"{minutes:02d}:{secs:02d}"


@keras.saving.register_keras_serializable()
class SiameseFingerprintModel(keras.Model):
    """Wide/deep fingerprint Siamese encoder."""

    def __init__(self, input_dim: int, use_logp: bool = True, use_molvol: bool = True):
        super().__init__()
        self.input_dim = int(input_dim)
        self.use_logp = bool(use_logp)
        self.use_molvol = bool(use_molvol)

        self.wide = keras.Sequential([
            layers.Dense(WIDE_DIM, activation="relu", kernel_initializer="he_normal"),
            layers.Dropout(0.01),
            layers.Dense(
                WIDE_DIM,
                activation="relu",
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.L2(0.0001),
            ),
        ])

        self.deep = keras.Sequential([
            layers.Dense(DEEP_DIM, activation="relu", kernel_initializer="he_normal"),
            layers.Dense(DEEP_DIM, activation="relu", kernel_initializer="he_normal"),
            layers.Dropout(0.1),
            layers.Dense(
                DEEP_DIM,
                activation="relu",
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.L2(0.001),
            ),
            layers.Dropout(0.1),
            layers.Dense(
                DEEP_DIM,
                activation="relu",
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.L2(0.001),
            ),
        ])

        self.embedding_head = keras.Sequential([
            layers.Dense(EMBEDDING_DIM, activation="linear"),
        ])
        self.logp_head = keras.Sequential([layers.Dense(1, activation="linear")])
        self.molvol_head = keras.Sequential([layers.Dense(1, activation="linear")])

    def _encode_base(self, x):
        wide_emb = self.wide(x)
        deep_emb = self.deep(x)
        return layers.Concatenate(name="concat_deep_wide")([wide_emb, deep_emb])

    def _encode_projected(self, x):
        base_emb = self._encode_base(x)
        emb = self.embedding_head(base_emb)
        return ops.normalize(emb, axis=1), base_emb

    def call(self, inputs):
        x1, x2 = inputs
        e1, base1 = self._encode_projected(x1)
        e2, base2 = self._encode_projected(x2)
        outputs = {
            "similarity": ops.sum(e1 * e2, axis=1),
        }
        if self.use_logp:
            logp_1 = ops.squeeze(self.logp_head(base1), axis=-1)
            logp_2 = ops.squeeze(self.logp_head(base2), axis=-1)
            outputs["logp_1"] = logp_1
            outputs["logp_2"] = logp_2
            outputs["logp_delta"] = logp_1 - logp_2
        if self.use_molvol:
            molvol_1 = ops.squeeze(self.molvol_head(base1), axis=-1)
            molvol_2 = ops.squeeze(self.molvol_head(base2), axis=-1)
            outputs["molvol_1"] = molvol_1
            outputs["molvol_2"] = molvol_2
            outputs["molvol_delta"] = molvol_1 - molvol_2
        return outputs

    def get_embedding(self, x):
        return self._encode_base(x)

    def get_config(self):
        return {
            "input_dim": self.input_dim,
            "use_logp": self.use_logp,
            "use_molvol": self.use_molvol,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class SiamesePairDataset(PyDataset):
    def __init__(
        self,
        x1,
        x2,
        targets: dict[str, np.ndarray],
        sample_weights: dict[str, np.ndarray] | None = None,
        batch_size: int = 64,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.x1 = keras.ops.convert_to_tensor(x1, dtype="float32")
        self.x2 = keras.ops.convert_to_tensor(x2, dtype="float32")
        self.targets = {key: keras.ops.convert_to_tensor(value, dtype="float32") for key, value in targets.items()}
        self.sample_weights = None
        if sample_weights is not None:
            self.sample_weights = {
                key: keras.ops.convert_to_tensor(value, dtype="float32")
                for key, value in sample_weights.items()
            }
        self.batch_size = int(batch_size)

    def __len__(self):
        return len(next(iter(self.targets.values()))) // self.batch_size

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = start + self.batch_size
        batch = (
            (self.x1[start:end], self.x2[start:end]),
            {key: value[start:end] for key, value in self.targets.items()},
        )
        if self.sample_weights is None:
            return batch
        return (*batch, {key: value[start:end] for key, value in self.sample_weights.items()})


class EpochLogger(callbacks.Callback):
    def __init__(self, use_logp: bool, use_molvol: bool):
        super().__init__()
        self.use_logp = use_logp
        self.use_molvol = use_molvol
        self._epoch_start = None
        self.epoch_times: list[str] = []

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start = time.perf_counter()

    @staticmethod
    def _value(logs, key):
        return float(logs.get(key, 0.0)) if logs else 0.0

    def _current_lr(self, logs):
        if logs and "learning_rate" in logs:
            return float(logs["learning_rate"])
        lr = getattr(self.model.optimizer, "learning_rate", None)
        if lr is None:
            return float("nan")
        try:
            return float(ops.convert_to_numpy(lr))
        except Exception:
            try:
                return float(lr)
            except Exception:
                return float("nan")

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.perf_counter() - self._epoch_start if self._epoch_start is not None else 0.0
        epoch_time = format_epoch_time(elapsed)
        self.epoch_times.append(epoch_time)
        lr = self._current_lr(logs)

        train_parts = [f"tan={self._value(logs, 'similarity_loss'):.4f}"]
        val_parts = [f"tan={self._value(logs, 'val_similarity_loss'):.4f}"]
        train_tail = [f"train_tan_mae={self._value(logs, 'similarity_mae'):.4f}"]
        val_tail = [f"val_tan_mae={self._value(logs, 'val_similarity_mae'):.4f}"]

        if self.use_logp:
            train_parts.extend([
                f"logp={0.5 * (self._value(logs, 'logp_1_loss') + self._value(logs, 'logp_2_loss')):.4f}",
                f"logp_delta={self._value(logs, 'logp_delta_loss'):.4f}",
            ])
            val_parts.extend([
                f"logp={0.5 * (self._value(logs, 'val_logp_1_loss') + self._value(logs, 'val_logp_2_loss')):.4f}",
                f"logp_delta={self._value(logs, 'val_logp_delta_loss'):.4f}",
            ])
        if self.use_molvol:
            train_parts.extend([
                f"molvol={0.5 * (self._value(logs, 'molvol_1_loss') + self._value(logs, 'molvol_2_loss')):.4f}",
                f"molvol_delta={self._value(logs, 'molvol_delta_loss'):.4f}",
            ])
            val_parts.extend([
                f"molvol={0.5 * (self._value(logs, 'val_molvol_1_loss') + self._value(logs, 'val_molvol_2_loss')):.4f}",
                f"molvol_delta={self._value(logs, 'val_molvol_delta_loss'):.4f}",
            ])

        print(
            f"Epoch {epoch + 1}: train_loss={self._value(logs, 'loss'):.4f} "
            f"({', '.join(train_parts)}) {' '.join(train_tail)} | "
            f"val_loss={self._value(logs, 'val_loss'):.4f} "
            f"({', '.join(val_parts)}) {' '.join(val_tail)} "
            f"lr={lr:.2e} time={epoch_time}"
        )


@dataclass
class TaskStats:
    mean: float
    std: float


class FingerprintSiamesePretrainer:
    def __init__(self, config_path: str | Path = DEFAULT_CONFIG_PATH):
        self.config_path = Path(config_path)
        self.config = load_yaml(self.config_path)
        self.data_cfg = self.config.get("data", {})
        self.task_cfg = self.config.get("tasks", {})
        self.loss_cfg = self.config.get("loss", {})
        self.training_cfg = self.config.get("training", {})

        self.use_logp = bool(self.task_cfg.get("use_logp", True))
        self.use_molvol = bool(self.task_cfg.get("use_molvol", True))
        self.descriptor_cache_csv = Path(
            self.data_cfg.get("descriptor_cache_csv", "resources/descriptors/hmdb_physchem.csv")
        )
        self.random_seed = int(self.data_cfg.get("random_seed", 42))
        self.batch_size = int(self.training_cfg.get("batch_size", 64))
        self.results_name = str(self.training_cfg.get("results_name", "Siamese_physchem_alvadesc"))
        self.results_dir = RESULTS_ROOT / self.results_name
        self.csv_path = self.results_dir / "metrics.csv"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        set_seed(self.random_seed)

    def run(self) -> None:
        self._write_model_manifest()
        df = self.load_data()
        self.train(df)
        self.aggregate_results()

    def load_data(self) -> pd.DataFrame:
        df = load_hmdb()
        if not self.use_logp and not self.use_molvol:
            return df
        if not self.descriptor_cache_csv.exists():
            raise FileNotFoundError(
                f"Descriptor cache not found: {self.descriptor_cache_csv}. "
                "Run src/data/generate_hmdb_physchem_descriptors.py first."
            )

        desc_df = pd.read_csv(self.descriptor_cache_csv)
        desc_df = desc_df.drop_duplicates(subset="inchi", keep="last")
        keep_cols = ["inchi"]
        if self.use_logp:
            keep_cols.append("logp")
        if self.use_molvol:
            keep_cols.append("mol_volume_mean")

        merged = df.merge(desc_df[keep_cols], on="inchi", how="left")
        before = len(merged)
        if self.use_logp:
            merged = merged[merged["logp"].notna()]
        if self.use_molvol:
            merged["mol_volume_valid"] = merged["mol_volume_mean"].notna().astype(np.float32)
        merged = merged.reset_index(drop=True)
        removed = before - len(merged)
        if removed > 0:
            print(f"Removing {removed:,} molecules with missing physchem descriptors")
        return merged

    def fingerprint_columns(self, df: pd.DataFrame) -> list[str]:
        return [column for column in df.columns if column.startswith("V")]

    def similarity(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        fingerprint_length = x1.shape[1]
        sim = np.zeros(x1.shape[0], dtype=np.float32)
        for i in range(x1.shape[0]):
            bv1 = DataStructs.ExplicitBitVect(fingerprint_length)
            for j, bit in enumerate(x1[i]):
                if bit:
                    bv1.SetBit(j)
            bv2 = DataStructs.ExplicitBitVect(fingerprint_length)
            for j, bit in enumerate(x2[i]):
                if bit:
                    bv2.SetBit(j)
            sim[i] = DataStructs.TanimotoSimilarity(bv1, bv2)
        return sim

    def build_model(self, input_dim: int) -> SiameseFingerprintModel:
        if int(input_dim) != FINGERPRINT_DIM:
            raise ValueError(f"Expected {FINGERPRINT_DIM} fingerprint features, found {input_dim}.")
        return SiameseFingerprintModel(input_dim=input_dim, use_logp=self.use_logp, use_molvol=self.use_molvol)

    def _fit_task_stats(self, values: pd.Series) -> TaskStats:
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        if not np.isfinite(mean):
            mean = 0.0
        if not np.isfinite(std) or std <= 1e-8:
            std = 1.0
        return TaskStats(mean=mean, std=std)

    def _apply_normalization(self, df: pd.DataFrame, stats_by_name: dict[str, TaskStats]) -> pd.DataFrame:
        df = df.copy()
        if self.use_logp:
            stats = stats_by_name["logp"]
            df["logp_scaled"] = ((df["logp"].astype(np.float32) - stats.mean) / stats.std).astype(np.float32)
        if self.use_molvol:
            stats = stats_by_name["molvol"]
            df["molvol_scaled"] = ((df["mol_volume_mean"].astype(np.float32) - stats.mean) / stats.std).astype(np.float32)
            df["molvol_valid"] = df["mol_volume_mean"].notna().astype(np.float32)
            df["molvol_scaled"] = df["molvol_scaled"].fillna(0.0).astype(np.float32)
        return df

    @staticmethod
    def _denormalize(values: np.ndarray, stats: TaskStats) -> np.ndarray:
        return (values.astype(np.float32) * np.float32(stats.std)) + np.float32(stats.mean)

    @staticmethod
    def _denormalize_delta(values: np.ndarray, stats: TaskStats) -> np.ndarray:
        return values.astype(np.float32) * np.float32(stats.std)

    def generate_pairs(self, df: pd.DataFrame, cols: list[str], n: int):
        x = df[cols].to_numpy(dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, 0.0, 1.0)

        classifications = df["classification"].values
        unique_classes = np.unique(classifications)
        class_to_indices = {cls: np.where(classifications == cls)[0] for cls in unique_classes}

        n_half = n // 2
        i1_rand = np.random.choice(len(x), n_half, replace=True)
        i2_rand = np.random.choice(len(x), n_half, replace=True)
        i1_same = np.zeros(n_half, dtype=int)
        i2_same = np.zeros(n_half, dtype=int)

        for i in range(n_half):
            cls = np.random.choice(unique_classes)
            idxs = class_to_indices[cls]
            if len(idxs) < 2:
                i1_same[i] = np.random.choice(len(x))
                i2_same[i] = np.random.choice(len(x))
            else:
                i1_same[i] = np.random.choice(idxs)
                i2_same[i] = np.random.choice(idxs)

        i1 = np.concatenate([i1_rand, i1_same])
        i2 = np.concatenate([i2_rand, i2_same])
        shuffle_idx = np.random.permutation(n)
        i1 = i1[shuffle_idx]
        i2 = i2[shuffle_idx]

        x1, x2 = x[i1], x[i2]
        targets = {
            "similarity": np.nan_to_num(self.similarity(x1, x2), nan=0.0).astype(np.float32),
        }
        sample_weights = {
            "similarity": np.ones(n, dtype=np.float32),
        }

        if self.use_logp:
            logp = df["logp_scaled"].to_numpy(dtype=np.float32)
            targets["logp_1"] = logp[i1].astype(np.float32)
            targets["logp_2"] = logp[i2].astype(np.float32)
            targets["logp_delta"] = (targets["logp_1"] - targets["logp_2"]).astype(np.float32)
            sample_weights["logp_1"] = np.ones(n, dtype=np.float32)
            sample_weights["logp_2"] = np.ones(n, dtype=np.float32)
            sample_weights["logp_delta"] = np.ones(n, dtype=np.float32)

        if self.use_molvol:
            molvol = df["molvol_scaled"].to_numpy(dtype=np.float32)
            molvol_valid = df["molvol_valid"].to_numpy(dtype=np.float32)
            valid_1 = molvol_valid[i1].astype(np.float32)
            valid_2 = molvol_valid[i2].astype(np.float32)
            targets["molvol_1"] = molvol[i1].astype(np.float32)
            targets["molvol_2"] = molvol[i2].astype(np.float32)
            targets["molvol_delta"] = (targets["molvol_1"] - targets["molvol_2"]).astype(np.float32)
            sample_weights["molvol_1"] = valid_1
            sample_weights["molvol_2"] = valid_2
            sample_weights["molvol_delta"] = (valid_1 * valid_2).astype(np.float32)

        return x1, x2, targets, sample_weights

    def compile_model(self, model: SiameseFingerprintModel) -> None:
        loss = {"similarity": "mae"}
        loss_weights = {"similarity": float(self.loss_cfg.get("lambda_similarity", 1.0))}
        metrics = {"similarity": ["mae"]}

        if self.use_logp:
            loss["logp_1"] = "mae"
            loss["logp_2"] = "mae"
            loss["logp_delta"] = "mae"
            loss_weights["logp_1"] = float(self.loss_cfg.get("lambda_logp_abs", 0.20))
            loss_weights["logp_2"] = float(self.loss_cfg.get("lambda_logp_abs", 0.20))
            loss_weights["logp_delta"] = float(self.loss_cfg.get("lambda_logp_delta", 0.1))
            metrics["logp_1"] = ["mae"]
            metrics["logp_2"] = ["mae"]
            metrics["logp_delta"] = ["mae"]

        if self.use_molvol:
            loss["molvol_1"] = "mae"
            loss["molvol_2"] = "mae"
            loss["molvol_delta"] = "mae"
            loss_weights["molvol_1"] = float(self.loss_cfg.get("lambda_molvol_abs", 0.3))
            loss_weights["molvol_2"] = float(self.loss_cfg.get("lambda_molvol_abs", 0.3))
            loss_weights["molvol_delta"] = float(self.loss_cfg.get("lambda_molvol_delta", 0.1))
            metrics["molvol_1"] = ["mae"]
            metrics["molvol_2"] = ["mae"]
            metrics["molvol_delta"] = ["mae"]

        model.compile(
            optimizer=optimizers.RMSprop(
                learning_rate=LEARNING_RATE,
                rho=RMSPROP_RHO,
                momentum=RMSPROP_MOMENTUM,
            ),
            loss=loss,
            loss_weights=loss_weights,
            metrics=metrics,
        )

    def train(self, df: pd.DataFrame) -> None:
        outer_gkf = GroupKFold(n_splits=5)
        for fold, (train_val_idx, test_idx) in enumerate(outer_gkf.split(df, groups=df["classification"])):
            print(f"Fold {fold + 1}")
            train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
            test_df = df.iloc[test_idx].reset_index(drop=True)

            inner_gkf = GroupKFold(n_splits=5)
            train_idx, val_idx = next(inner_gkf.split(train_val_df, groups=train_val_df["classification"]))
            train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
            val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

            stats_by_name: dict[str, TaskStats] = {}
            if self.use_logp:
                stats_by_name["logp"] = self._fit_task_stats(train_df["logp"])
            if self.use_molvol:
                stats_by_name["molvol"] = self._fit_task_stats(train_df["mol_volume_mean"].dropna())

            train_df_scaled = self._apply_normalization(train_df, stats_by_name)
            val_df_scaled = self._apply_normalization(val_df, stats_by_name)
            test_df_scaled = self._apply_normalization(test_df, stats_by_name)

            cols = self.fingerprint_columns(train_df)
            dim = len(cols)

            x1_train, x2_train, train_targets, train_sample_weights = self.generate_pairs(train_df_scaled, cols, TRAIN_PAIRS)
            dataset = SiamesePairDataset(
                x1_train,
                x2_train,
                train_targets,
                sample_weights=train_sample_weights,
                batch_size=self.batch_size,
            )
            x1_val, x2_val, val_targets, val_sample_weights = self.generate_pairs(val_df_scaled, cols, VAL_PAIRS)

            model = self.build_model(dim)
            self.compile_model(model)

            fold_dir = self.results_dir / f"fold_{fold + 1}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            weights = fold_dir / "best.weights.h5"
            epoch_logger = EpochLogger(use_logp=self.use_logp, use_molvol=self.use_molvol)
            cb = [
                epoch_logger,
                callbacks.ModelCheckpoint(
                    str(weights),
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
                dataset,
                epochs=int(self.training_cfg.get("max_epochs", 2000)),
                validation_data=((x1_val, x2_val), val_targets, val_sample_weights),
                callbacks=cb,
                verbose=0,
            )
            history.history["epoch_time"] = epoch_logger.epoch_times
            self._save_training_log(history, fold + 1, fold_dir)
            self._plot_loss_history(history, fold + 1, fold_dir)

            model = self.build_model(dim)
            dummy = (np.zeros((1, dim), dtype=np.float32), np.zeros((1, dim), dtype=np.float32))
            _ = model(dummy)
            model.load_weights(weights)

            x1_test, x2_test, test_targets, test_sample_weights = self.generate_pairs(test_df_scaled, cols, TEST_PAIRS)
            self.evaluate(model, x1_test, x2_test, test_targets, test_sample_weights, stats_by_name, fold, fold_dir)

            gc.collect()
            print(f"      -> Memory cleared after fold {fold + 1}")

    @staticmethod
    def _masked_mae(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> str:
        mask = np.asarray(mask).astype(bool)
        if mask.sum() == 0:
            return ""
        return f"{mean_absolute_error(y_true[mask], y_pred[mask]):.4f}"

    def evaluate(self, model, x1, x2, targets, sample_weights, stats_by_name: dict[str, TaskStats], fold, folder: Path):
        pred = model.predict((x1, x2), batch_size=512, verbose=0)
        pred_similarity = np.asarray(pred["similarity"]).flatten()
        y_similarity = np.asarray(targets["similarity"])
        metrics_row = {
            "Fold": fold + 1,
            "MAE": f"{mean_absolute_error(y_similarity, pred_similarity):.4f}",
            "R2": f"{r2_score(y_similarity, pred_similarity):.4f}",
            "LogP1_MAE": "",
            "LogP2_MAE": "",
            "LogPDelta_MAE": "",
            "MolVol1_MAE": "",
            "MolVol2_MAE": "",
            "MolVolDelta_MAE": "",
        }
        if self.use_logp:
            stats = stats_by_name["logp"]
            metrics_row["LogP1_MAE"] = f"{mean_absolute_error(self._denormalize(targets['logp_1'], stats), self._denormalize(np.asarray(pred['logp_1']).flatten(), stats)):.4f}"
            metrics_row["LogP2_MAE"] = f"{mean_absolute_error(self._denormalize(targets['logp_2'], stats), self._denormalize(np.asarray(pred['logp_2']).flatten(), stats)):.4f}"
            metrics_row["LogPDelta_MAE"] = f"{mean_absolute_error(self._denormalize_delta(targets['logp_delta'], stats), self._denormalize_delta(np.asarray(pred['logp_delta']).flatten(), stats)):.4f}"
        if self.use_molvol:
            stats = stats_by_name["molvol"]
            pred_molvol_1 = self._denormalize(np.asarray(pred["molvol_1"]).flatten(), stats)
            pred_molvol_2 = self._denormalize(np.asarray(pred["molvol_2"]).flatten(), stats)
            pred_molvol_delta = self._denormalize_delta(np.asarray(pred["molvol_delta"]).flatten(), stats)
            y_molvol_1 = self._denormalize(np.asarray(targets["molvol_1"]), stats)
            y_molvol_2 = self._denormalize(np.asarray(targets["molvol_2"]), stats)
            y_molvol_delta = self._denormalize_delta(np.asarray(targets["molvol_delta"]), stats)
            metrics_row["MolVol1_MAE"] = self._masked_mae(y_molvol_1, pred_molvol_1, sample_weights["molvol_1"])
            metrics_row["MolVol2_MAE"] = self._masked_mae(y_molvol_2, pred_molvol_2, sample_weights["molvol_2"])
            metrics_row["MolVolDelta_MAE"] = self._masked_mae(
                y_molvol_delta,
                pred_molvol_delta,
                sample_weights["molvol_delta"],
            )

        plt.figure()
        plt.scatter(y_similarity, pred_similarity, s=1, alpha=0.5)
        plt.plot([0, 1], [0, 1])
        plt.xlabel("True Similarity")
        plt.ylabel("Predicted Similarity")
        plt.title(f"Fold {fold + 1}")
        plt.savefig(folder / "scatter.png")
        plt.close()

        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics_row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(metrics_row)

    def _plot_loss_history(self, history, fold_number: int, fold_dir: Path) -> None:
        plt.figure(figsize=(10, 6))
        plt.plot(history.history["loss"], label="Training Loss", linewidth=2)
        plt.plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
        plt.title(f"Fingerprint Siamese - Fold {fold_number} Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plot_path = fold_dir / f"loss_curve_fold_{fold_number}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _save_training_log(self, history, fold_number: int, fold_dir: Path) -> None:
        log_dir = fold_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / f"training_history_fold_{fold_number}.json").open("w", encoding="utf-8") as f:
            json.dump(history.history, f, indent=2)
        pd.DataFrame(history.history).to_csv(log_dir / f"training_log_fold_{fold_number}.csv", index_label="epoch")

    def aggregate_results(self) -> None:
        if not self.csv_path.exists():
            return
        df = pd.read_csv(self.csv_path)
        df = df[df["Fold"].astype(str) != "Total"].copy()
        if df.empty:
            return
        numeric_cols = [
            "MAE", "R2",
            "LogP1_MAE", "LogP2_MAE", "LogPDelta_MAE",
            "MolVol1_MAE", "MolVol2_MAE", "MolVolDelta_MAE",
        ]
        summary_row = {"Fold": "Total"}
        for col in numeric_cols:
            values = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=np.float64)
            summary_row[col] = "" if values.size == 0 else f"{values.mean():.4f}±{values.std():.4f}"
        with self.csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Fold", *numeric_cols])
            writer.writerow(summary_row)

    def _write_model_manifest(self) -> None:
        payload = {
            "model_type": "fingerprint_siamese_pretrain",
            "fingerprint_source": "alvadesc",
            "embedding_model_class": "SiameseFingerprintModel",
            "input_dim": FINGERPRINT_DIM,
            "embedding_dim": EMBEDDING_DIM,
            "base_embedding_dim": BASE_EMBEDDING_DIM,
            "wide_dim": WIDE_DIM,
            "deep_dim": DEEP_DIM,
            "use_logp": self.use_logp,
            "use_molvol": self.use_molvol,
            "loss": "mae",
            "config_path": str(self.config_path),
        }
        with (self.results_dir / "model_manifest.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain the HMDB AlvaDesc fingerprint Siamese model.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()
    FingerprintSiamesePretrainer(config_path=args.config).run()


if __name__ == "__main__":
    main()
