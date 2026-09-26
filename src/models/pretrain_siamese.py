from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_hmdb_rdkit
from src.models.pretrain_siamese_alvadesc import (
    BASE_EMBEDDING_DIM,
    DEEP_DIM,
    EMBEDDING_DIM,
    FINGERPRINT_DIM,
    RESULTS_ROOT,
    WIDE_DIM,
    FiveFoldFingerprintSiamesePretrainer as AlvaDescFiveFoldFingerprintSiamesePretrainer,
    SiameseFingerprintModel,
)
from src.models.hmdb_pretraining import HMDBPretrainingMixin


DEFAULT_CONFIG_PATH = Path("configs/pretrain_siamese.yaml")


class FiveFoldFingerprintSiamesePretrainer(
    AlvaDescFiveFoldFingerprintSiamesePretrainer
):
    """Legacy five-fold pretrainer using HMDB RDKit fingerprints."""

    fingerprint_source = "rdkit"
    pretraining_protocol = "five_fold_grouped"

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG_PATH):
        super().__init__(config_path=config_path)

    def load_data(self) -> pd.DataFrame:
        df = load_hmdb_rdkit()
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

    def _write_model_manifest(self) -> None:
        payload = {
            "model_type": "fingerprint_siamese_pretrain",
            "pretraining_protocol": self.pretraining_protocol,
            "fingerprint_source": "rdkit",
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


class FingerprintSiamesePretrainer(
    HMDBPretrainingMixin,
    FiveFoldFingerprintSiamesePretrainer,
):
    """Pretrain one RDKit encoder on the canonical HMDB 90/10 split."""

    pretraining_protocol = "hmdb_90_10"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pretrain one HMDB RDKit fingerprint Siamese model on a 90/10 split."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()
    FingerprintSiamesePretrainer(config_path=args.config).run()


if __name__ == "__main__":
    main()
