from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
from keras import layers, models

from src.models.ablations.run_siamese_deep_only import (
    SiameseCCSRunner as DeepOnlyCCSRunner,
)
from src.models.ablations.run_siamese_wide_only import (
    SiameseCCSRunner as WideOnlyCCSRunner,
)
from src.models.pretrain_siamese import SiameseFingerprintModel
from src.models.run_baseline import DEFAULT_CONFIG_PATH, RESULTS_ROOT, set_seed
from src.models.run_siamese import SiameseCCSRunner


DGR_MODEL = "gated_residual_mlp"
ARCHITECTURES = ("wide_deep", "wide_only", "deep_only")
RUNNER_CLASSES = {
    "wide_deep": SiameseCCSRunner,
    "wide_only": WideOnlyCCSRunner,
    "deep_only": DeepOnlyCCSRunner,
}


class ScratchDGRRunner(SiameseCCSRunner):
    """Train the reference wide+deep encoder and DGR-MLP jointly from scratch."""

    def load_siamese(self, fp_dim: int) -> SiameseFingerprintModel:
        return SiameseFingerprintModel(input_dim=fp_dim, use_logp=True, use_molvol=True)

    def siamese_embedding(self, fp_input, fp_dim: int, trainable: bool):
        del trainable
        siamese = self.load_siamese(fp_dim)
        encoder_input = layers.Input(shape=(fp_dim,), name="siamese_encoder_input")
        encoder_output = siamese.get_embedding(encoder_input)
        encoder = models.Model(encoder_input, encoder_output, name="siamese_encoder")
        encoder.trainable = True
        return encoder(fp_input)

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
        route_name = f"{'_'.join(train_val_dbs)}_to_{test_db}"
        key = (
            f"{self.random_seed}|{self.fingerprint_source}|{route_name}|"
            f"{fold_idx + 1}|dgr_mlp"
        ).encode("utf-8")
        offset = int.from_bytes(hashlib.sha256(key).digest()[:4], "big")
        self.current_cell_seed = int((self.random_seed + offset) % (2**31 - 1))
        keras.backend.clear_session()
        gc.collect()
        set_seed(self.current_cell_seed)
        super().train_one_fold(
            train_df,
            val_df,
            test_df,
            fold_idx,
            model_type,
            train_val_dbs,
            test_db,
        )

    def siamese_metadata(self, model_type: str) -> dict[str, Any]:
        return {
            "mode": "siamesa_scratch_end_to_end",
            "fingerprint_source": self.fingerprint_source,
            "pretrained_weights_loaded": False,
            "encoder_architecture": "wide_deep",
            "encoder_trainable_from_first_batch": True,
            "experiment_tag": self.experiment_tag,
            "ccs_loss": "mae",
            "model_type": model_type,
            "cell_seed": getattr(self, "current_cell_seed", None),
            "uses_aux_features": self.uses_aux_features(model_type),
        }


def build_runner(
    *,
    architecture: str,
    config_path: str | Path,
    folds: int,
    random_seed: int,
    fingerprint_source: str,
    siamese_results_dir: str | Path | None,
    output_root: str | Path,
    experiment_tag: str,
    scratch: bool = False,
):
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture '{architecture}'. Expected one of {ARCHITECTURES}.")
    if scratch and architecture != "wide_deep":
        raise ValueError("Random initialization is defined only for the wide_deep architecture.")
    if not scratch and siamese_results_dir is None:
        raise ValueError("A pretrained checkpoint directory is required unless --scratch is used.")

    runner_class = ScratchDGRRunner if scratch else RUNNER_CLASSES[architecture]
    runner = runner_class(
        config_path=config_path,
        folds=folds,
        random_seed=random_seed,
        fingerprint_source=fingerprint_source,
        siamese_results_dir=(
            Path("results/ablations/unused_pretrained_checkpoint")
            if scratch
            else siamese_results_dir
        ),
        experiment_tag=experiment_tag,
        **({"model_types": [DGR_MODEL]} if runner_class in {SiameseCCSRunner, ScratchDGRRunner} else {}),
    )
    runner.model_types = (DGR_MODEL,)
    if not scratch:
        manifest = runner.read_siamese_manifest()
        if manifest.get("pretraining_protocol") != "hmdb_90_10":
            raise ValueError(
                "The ablation checkpoint must use the canonical HMDB 90/10 pretraining protocol."
            )
    runner.results_root = Path(output_root)
    runner.results_root.mkdir(parents=True, exist_ok=False)
    # Keep the canonical CCSTrainer splits so every ablation is paired fold by fold
    # with the reference experiment.
    runner.splits_dir = RESULTS_ROOT / "splits"
    return runner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one pretrained or randomly initialized Siamese ablation with DGR-MLP."
    )
    parser.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--fingerprint-source", choices=("rdkit",), default="rdkit")
    parser.add_argument("--siamese-results-dir", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--experiment-tag", required=True)
    parser.add_argument("--scratch", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.folds < 1 or args.folds > 5:
        raise ValueError("--folds must be between 1 and 5.")
    if args.output_root.exists():
        raise FileExistsError(
            f"Downstream output directory already exists: {args.output_root}. "
            "This one-shot runner never overwrites or resumes an experiment."
        )
    runner = build_runner(
        architecture=args.architecture,
        config_path=args.config,
        folds=args.folds,
        random_seed=args.random_seed,
        fingerprint_source=args.fingerprint_source,
        siamese_results_dir=args.siamese_results_dir,
        output_root=args.output_root,
        experiment_tag=args.experiment_tag,
        scratch=args.scratch,
    )
    runner.run()
    summary = {
        "architecture": args.architecture,
        "scratch": args.scratch,
        "fingerprint_source": args.fingerprint_source,
        "model_type": DGR_MODEL,
        "folds": args.folds,
        "random_seed": args.random_seed,
        "siamese_results_dir": (
            None if args.scratch else str(args.siamese_results_dir)
        ),
    }
    (args.output_root / "run_manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
