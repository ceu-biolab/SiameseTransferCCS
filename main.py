from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

from src.models.pretrain_siamese import FingerprintSiamesePretrainer
from src.models.run_baseline import FingerprintCCSBaselineRunner
from src.models.run_siamese import SiameseCCSRunner


PRETRAIN_CONFIG = Path("configs/pretrain_siamese.yaml")
CCS_CONFIG = Path("configs/ccs_prediction_heads.yaml")
SIAMESE_RESULTS_DIR = Path("results/Siamese_physchem")
FINGERPRINT_SOURCE = "rdkit"
FOLDS = 5


def main() -> None:
    FingerprintSiamesePretrainer(config_path=PRETRAIN_CONFIG).run()
    FingerprintCCSBaselineRunner(
        config_path=CCS_CONFIG,
        folds=FOLDS,
        fingerprint_source=FINGERPRINT_SOURCE,
    ).run()
    SiameseCCSRunner(
        config_path=CCS_CONFIG,
        folds=FOLDS,
        fingerprint_source=FINGERPRINT_SOURCE,
        siamese_results_dir=SIAMESE_RESULTS_DIR,
    ).run()


if __name__ == "__main__":
    main()
