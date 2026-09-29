from __future__ import annotations

import argparse
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

from src.data import load_hmdb_rdkit
from src.models.hmdb_pretraining import HMDBPretrainingMixin
from src.models.losses import scalar_absolute_error


DEFAULT_CONFIG_PATH = Path("configs/pretrain_siamese_wide_only.yaml")
RESULTS_ROOT = Path("results")
WIDE_DIM = 1536
DEEP_DIM = 512
EMBEDDING_DIM = WIDE_DIM + DEEP_DIM
LEARNING_RATE = 1e-4
RMSPROP_RHO = 0.7
RMSPROP_MOMENTUM = 0.7


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
    """Fingerprint Siamese architecture with only the wide branch active."""

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

    def build(self, input_shape):
        fp_shape = (None, self.input_dim)
        self.wide.build(fp_shape)
        self.embedding_head.build((None, WIDE_DIM))
        if self.use_logp:
            self.logp_head.build((None, WIDE_DIM))
        if self.use_molvol:
            self.molvol_head.build((None, WIDE_DIM))
        super().build(input_shape)

    def _encode_base(self, x):
        return self.wide(x)

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


class FingerprintPretrainerBase:
    pair_dataset_class = SiamesePairDataset
    epoch_logger_class = EpochLogger

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
        self.results_name = str(self.training_cfg.get("results_name", "Siamese_physchem_wide_only_test"))
        self.results_dir = RESULTS_ROOT / self.results_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        set_seed(self.random_seed)

    def load_data(self) -> pd.DataFrame:
        df = load_hmdb_rdkit()
        if not self.descriptor_cache_csv.exists():
            raise FileNotFoundError(
                f"Descriptor cache not found: {self.descriptor_cache_csv}. "
                "Run src/data/generate_hmdb_physchem_descriptors.py first."
            )

        desc_df = pd.read_csv(self.descriptor_cache_csv)
        desc_df = desc_df.drop_duplicates(subset="inchi", keep="last")
        # Keep the molecular population fixed across pretraining objectives.
        keep_cols = ["inchi", "logp"]
        if self.use_molvol:
            keep_cols.append("mol_volume_mean")

        merged = df.merge(desc_df[keep_cols], on="inchi", how="left")
        before = len(merged)
        merged = merged[merged["logp"].notna()].copy()
        if self.use_molvol:
            merged["mol_volume_valid"] = merged["mol_volume_mean"].notna().astype(np.float32)
        merged = merged.reset_index(drop=True)
        removed = before - len(merged)
        if removed > 0:
            print(f"Removing {removed:,} molecules with missing LogP (all objectives)")
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
            loss["molvol_1"] = scalar_absolute_error
            loss["molvol_2"] = scalar_absolute_error
            loss["molvol_delta"] = scalar_absolute_error
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

    def _write_model_manifest(self) -> None:
        payload = {
            "model_type": "fingerprint_siamese_pretrain_wide_only",
            "pretraining_protocol": "hmdb_90_10",
            "fingerprint_source": "rdkit",
            "embedding_model_class": "SiameseFingerprintModel",
            "branch_mode": "wide_only",
            "embedding_dim": EMBEDDING_DIM,
            "wide_dim": WIDE_DIM,
            "deep_dim": DEEP_DIM,
            "use_logp": self.use_logp,
            "use_molvol": self.use_molvol,
            "config_path": str(self.config_path),
        }
        with (self.results_dir / "model_manifest.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


class FingerprintSiamesePretrainer(
    HMDBPretrainingMixin,
    FingerprintPretrainerBase,
):
    """Pretrain one RDKit wide-only encoder on the shared HMDB 90/10 protocol."""

    fingerprint_source = "rdkit"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pretrain one HMDB RDKit fingerprint Siamese wide-only model on a 90/10 split."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()
    FingerprintSiamesePretrainer(config_path=args.config).run()


if __name__ == "__main__":
    main()
