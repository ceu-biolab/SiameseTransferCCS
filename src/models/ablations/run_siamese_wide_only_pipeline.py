from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.models.ablations.run_siamese_wide_only_dgr import (
    EXPERIMENT_ROOT,
    MANIFEST_PATH,
    PAPER_CONFIG_ROOT,
    PROJECT_ROOT,
    ROUTES,
)


PRETRAINING = {
    "rdkit": {
        "module": "src.models.ablations.pretrain_siamese_wide_only",
        "config": PAPER_CONFIG_ROOT / "pretrain_wide_only_1536_rdkit.yaml",
        "directory": PROJECT_ROOT
        / "results"
        / "Siamese_physchem_wide_only_1536",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_logged(command: list[str], log_path: Path, dry_run: bool) -> None:
    printable = " ".join(command)
    print(f"$ {printable}", flush=True)
    print(f"  log: {log_path}", flush=True)
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {printable}\n\n")
        log_file.flush()
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError(f"Could not capture output for command: {printable}")
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(
                return_code,
                command,
            )


def validate_and_register_checkpoint(source: str) -> dict[str, str]:
    spec = PRETRAINING[source]
    directory = Path(spec["directory"])
    weights = directory / "fold_1" / "best.weights.h5"
    model_manifest_path = directory / "model_manifest.json"
    for path in (weights, model_manifest_path):
        if not path.exists():
            raise FileNotFoundError(f"Pretraining did not produce the required artifact: {path}")

    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    expected = {
        "fingerprint_source": source,
        "branch_mode": "wide_only",
        "model_type": "fingerprint_siamese_pretrain_wide_only",
        "pretraining_protocol": "hmdb_90_10",
        "wide_dim": 1536,
        "deep_dim": 512,
        "embedding_dim": 2048,
    }
    for key, value in expected.items():
        if model_manifest.get(key) != value:
            raise ValueError(
                f"Checkpoint manifest mismatch for {key}: expected {value!r}, "
                f"found {model_manifest.get(key)!r} in {model_manifest_path}."
            )

    weights_hash = sha256(weights)
    manifest_hash = sha256(model_manifest_path)
    experiment_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    checkpoint = experiment_manifest["checkpoints"][source]
    checkpoint.update(
        {
            "directory": str(directory.relative_to(PROJECT_ROOT)),
            "weights": str(weights.relative_to(PROJECT_ROOT)),
            "weights_sha256": weights_hash,
            "model_manifest": str(model_manifest_path.relative_to(PROJECT_ROOT)),
            "model_manifest_sha256": manifest_hash,
        }
    )
    experiment_manifest["encoder_architecture"] = {
        "branch_mode": "wide_only",
        "wide_dim": 1536,
        "deep_dim_reference": 512,
        "embedding_dim": 2048,
    }
    atomic_json(MANIFEST_PATH, experiment_manifest)
    print(
        f"Registered {source} checkpoint | weights_sha256={weights_hash}",
        flush=True,
    )
    return {
        "weights_sha256": weights_hash,
        "model_manifest_sha256": manifest_hash,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pretrain the 1536-unit HMDB wide-only Siamese encoder "
            "for RDKit, register its hashes, and run all CCS DGR-MLP tasks."
        )
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.random_seed != 42:
        raise ValueError("The frozen HMDB pretraining protocol requires --random-seed 42.")
    if not 1 <= args.folds <= 5:
        raise ValueError("--folds must be between 1 and 5.")

    logs_dir = EXPERIMENT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    sources = list(PRETRAINING)

    for source in sources:
        spec = PRETRAINING[source]
        run_logged(
            [
                sys.executable,
                "-m",
                str(spec["module"]),
                "--config",
                str(spec["config"]),
            ],
            logs_dir / f"pretrain_wide1536_{source}.log",
            args.dry_run,
        )
        if not args.dry_run:
            validate_and_register_checkpoint(source)

    downstream_command = [
        sys.executable,
        "-m",
        "src.models.ablations.run_siamese_wide_only_dgr",
        "--sources",
        *sources,
        "--routes",
        *ROUTES,
        "--folds",
        str(args.folds),
        "--random-seed",
        str(args.random_seed),
    ]
    run_logged(
        downstream_command,
        logs_dir / "wide1536_downstream_all_seed_42.log",
        args.dry_run,
    )
    if not args.dry_run:
        print("Wide-only 1536 HMDB + CCS pipeline completed successfully.", flush=True)


if __name__ == "__main__":
    main()
