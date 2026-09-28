from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from src.models.ablations.pretrain_siamese_wide_only import (
    EMBEDDING_DIM,
    WIDE_DIM,
    SiameseFingerprintModel,
)
from src.models.run_baseline import DEFAULT_CONFIG_PATH, VALID_FINGERPRINT_SOURCES
from src.models.run_siamese import SiameseCCSRunner as ReferenceCCSRunner


MODEL_TYPE = "gated_residual_mlp"
MODEL_TYPES = (MODEL_TYPE,)
SIAMESE_RESULTS_DIRS = {
    "alvadesc": Path("results/Siamese_physchem_wide_only_alvadesc"),
    "rdkit": Path("results/Siamese_physchem_wide_only"),
}


class SiameseCCSRunner(ReferenceCCSRunner):
    """Run only the DGR-MLP head on a pretrained 1536-dimensional wide encoder."""

    def __init__(
        self,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        folds: int = 5,
        random_seed: int = 42,
        fingerprint_source: str = "rdkit",
        siamese_results_dir: str | Path | None = None,
        experiment_tag: str = "hmdb90_10_v1",
    ):
        super().__init__(
            config_path=config_path,
            folds=folds,
            random_seed=random_seed,
            fingerprint_source=fingerprint_source,
            siamese_results_dir=(
                siamese_results_dir or SIAMESE_RESULTS_DIRS[fingerprint_source]
            ),
            experiment_tag=experiment_tag,
            model_types=MODEL_TYPES,
        )

    def load_siamese(self, fp_dim: int) -> SiameseFingerprintModel:
        weights_path = self.weights_path()
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Missing Siamese weights: {weights_path}. Run "
                "src.models.ablations.pretrain_siamese_wide_only first."
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

    def validate_siamese_manifest(self, manifest: dict[str, Any]) -> None:
        manifest_path = self.siamese_results_dir / "model_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing Siamese manifest: {manifest_path}")
        expected = {
            "fingerprint_source": self.fingerprint_source,
            "branch_mode": "wide_only",
            "model_type": "fingerprint_siamese_pretrain_wide_only",
            "wide_dim": WIDE_DIM,
            "embedding_dim": EMBEDDING_DIM,
        }
        for key, expected_value in expected.items():
            actual = manifest.get(key)
            if actual != expected_value:
                raise ValueError(
                    f"Siamese manifest {key} mismatch at {manifest_path}: "
                    f"expected {expected_value!r}, found {actual!r}."
                )

    def result_subfolder(self, train_val_dbs: list[str], test_db: str, model_type: str) -> str:
        base = super().result_subfolder(train_val_dbs, test_db, model_type)
        return base.replace("_siamesa_", "_siamesa_wide_only_", 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DGR-MLP CCS prediction on pretrained wide-only Siamese embeddings."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--fingerprint-source",
        choices=sorted(VALID_FINGERPRINT_SOURCES),
        default="rdkit",
    )
    parser.add_argument("--siamese-results-dir", default=None)
    parser.add_argument("--experiment-tag", default="hmdb90_10_v1")
    args = parser.parse_args()
    SiameseCCSRunner(
        config_path=args.config,
        folds=args.folds,
        random_seed=args.random_seed,
        fingerprint_source=args.fingerprint_source,
        siamese_results_dir=args.siamese_results_dir,
        experiment_tag=args.experiment_tag,
    ).run()


if __name__ == "__main__":
    main()
