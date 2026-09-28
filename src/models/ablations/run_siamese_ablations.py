from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = PROJECT_ROOT / "results"
ABLATION_ROOT = RESULTS_ROOT / "ablations" / "dgr_hmdb"
DEFAULT_CCS_CONFIG = PROJECT_ROOT / "configs" / "ccs_prediction_heads.yaml"
DEFAULT_REFERENCE_SIAMESE_DIR = PROJECT_ROOT / "results" / "Siamese_physchem"
DEFAULT_EXPERIMENT_TAG = "dgr_hmdb_ablations"
DEFAULT_REFERENCE_TAG = "hmdb90_10_v1"
PRETRAINING_PROTOCOL = "hmdb_90_10"
MODEL_TYPE = "gated_residual_mlp"
FINGERPRINT_SOURCE = "rdkit"

ROUTES = {
    "ccsbase_to_ccsbase": ("ccsbase", "ccsbase"),
    "ccsbase_to_metlinccs": ("ccsbase", "metlinccs"),
    "metlinccs_to_ccsbase": ("metlinccs", "ccsbase"),
    "metlinccs_to_metlinccs": ("metlinccs", "metlinccs"),
}

SPLIT_FILENAMES = {
    "ccsbase_to_ccsbase": "train_val_ccsbase_test_ccsbase_rdkit_fingerprints_same_db_splits.json",
    "ccsbase_to_metlinccs": "train_val_ccsbase_test_metlinccs_rdkit_fingerprints_external_test_trainval_splits.json",
    "metlinccs_to_ccsbase": "train_val_metlinccs_test_ccsbase_rdkit_fingerprints_external_test_trainval_splits.json",
    "metlinccs_to_metlinccs": "train_val_metlinccs_test_metlinccs_rdkit_fingerprints_same_db_splits.json",
}

ARCHITECTURES = {
    "wide_deep": {
        "base_config": PROJECT_ROOT / "configs" / "pretrain_siamese.yaml",
        "pretrain_module": "src.models.pretrain_siamese",
        "wide_dim": 1536,
        "deep_dim": 512,
    },
    "wide_only": {
        "base_config": PROJECT_ROOT / "configs" / "pretrain_siamese_wide_only.yaml",
        "pretrain_module": "src.models.ablations.pretrain_siamese_wide_only",
        "wide_dim": 1536,
        "deep_dim": 0,
    },
    "deep_only": {
        "base_config": PROJECT_ROOT / "configs" / "pretrain_siamese_deep_only.yaml",
        "pretrain_module": "src.models.ablations.pretrain_siamese_deep_only",
        "wide_dim": 0,
        "deep_dim": 512,
    },
}


@dataclass(frozen=True)
class AblationSpec:
    name: str
    label: str
    architecture: str
    use_tanimoto: bool
    use_logp: bool
    use_molvol: bool
    scratch: bool = False

    @property
    def lambda_similarity(self) -> float:
        return 1.0 if self.use_tanimoto else 0.0


ABLATIONS = {
    spec.name: spec
    for spec in (
        AblationSpec(
            "random_initialization",
            "Random initialization",
            "wide_deep",
            True,
            True,
            True,
            scratch=True,
        ),
        AblationSpec("tanimoto_only", "Tanimoto only", "wide_deep", True, False, False),
        AblationSpec("tanimoto_logp", "Tanimoto + logP", "wide_deep", True, True, False),
        AblationSpec(
            "tanimoto_molvol", "Tanimoto + MolVol", "wide_deep", True, False, True
        ),
        AblationSpec("logp_molvol", "logP + MolVol", "wide_deep", False, True, True),
        AblationSpec("logp_only", "logP only", "wide_deep", False, True, False),
        AblationSpec("molvol_only", "MolVol only", "wide_deep", False, False, True),
        AblationSpec("wide_only", "Wide only", "wide_only", True, True, True),
        AblationSpec("deep_only", "Deep only", "deep_only", True, True, True),
    )
}


@dataclass(frozen=True)
class AblationJob:
    spec: AblationSpec
    pretrain_config_path: Path | None
    checkpoint_dir: Path | None
    downstream_dir: Path
    pretrain_module: str | None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def build_pretrain_config(
    spec: AblationSpec,
    checkpoint_dir: Path,
    random_seed: int,
) -> dict[str, Any]:
    if spec.scratch:
        raise ValueError("Random initialization does not have a pretraining config.")
    architecture = ARCHITECTURES[spec.architecture]
    base_config_path = Path(architecture["base_config"])
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base pretraining config not found: {base_config_path}")
    config = load_yaml(base_config_path)
    split = config.get("split", {})
    if float(split.get("train_fraction", 0.0)) != 0.90 or float(
        split.get("validation_fraction", 0.0)
    ) != 0.10:
        raise ValueError(f"Expected the canonical HMDB 90/10 split in {base_config_path}.")

    config.setdefault("data", {})["random_seed"] = int(random_seed)
    config.setdefault("tasks", {})["use_logp"] = spec.use_logp
    config["tasks"]["use_molvol"] = spec.use_molvol
    config.setdefault("loss", {})["lambda_similarity"] = spec.lambda_similarity
    config.setdefault("training", {})["results_name"] = str(checkpoint_dir.resolve())
    config["ablation"] = {
        "name": spec.name,
        "label": spec.label,
        "architecture": spec.architecture,
        "fingerprint_source": FINGERPRINT_SOURCE,
        "pretraining_protocol": PRETRAINING_PROTOCOL,
        "use_tanimoto": spec.use_tanimoto,
        "use_logp": spec.use_logp,
        "use_molvol": spec.use_molvol,
        "model_type": MODEL_TYPE,
        "base_pretrain_config": str(base_config_path.relative_to(PROJECT_ROOT)),
    }
    return config


def build_jobs(experiment_root: Path) -> list[AblationJob]:
    jobs: list[AblationJob] = []
    for spec in ABLATIONS.values():
        if spec.scratch:
            jobs.append(
                AblationJob(
                    spec=spec,
                    pretrain_config_path=None,
                    checkpoint_dir=None,
                    downstream_dir=experiment_root / "downstream" / spec.name,
                    pretrain_module=None,
                )
            )
            continue
        architecture = ARCHITECTURES[spec.architecture]
        jobs.append(
            AblationJob(
                spec=spec,
                pretrain_config_path=experiment_root / "configs" / f"pretrain_{spec.name}.yaml",
                checkpoint_dir=experiment_root / "checkpoints" / spec.name,
                downstream_dir=experiment_root / "downstream" / spec.name,
                pretrain_module=str(architecture["pretrain_module"]),
            )
        )
    return jobs


def reference_result_paths(
    reference_root: Path,
    reference_tag: str,
    folds: int,
) -> dict[str, str]:
    folds_tag = "single_fold" if folds == 1 else "five_folds"
    paths: dict[str, str] = {}
    for route_name, (train_db, test_db) in ROUTES.items():
        folder_name = (
            f"train_val_{train_db}_test_{test_db}_{FINGERPRINT_SOURCE}_"
            f"siamesa_{reference_tag}_with_molfeatures_with_mass_{MODEL_TYPE}_{folds_tag}"
        )
        csv_path = reference_root / folder_name / f"{folder_name}_results.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Missing reference result for {route_name}: {csv_path}. "
                "Run the standard pretrained Wide+Deep CCS workflow first."
            )
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = [
                row
                for row in csv.DictReader(handle)
                if str(row.get("Fold", "")).strip().lower() != "total"
            ]
        if len(rows) < folds:
            raise ValueError(
                f"Reference result {csv_path} contains {len(rows)} folds; expected at least {folds}."
            )
        if any(not row.get("MAE") for row in rows[:folds]):
            raise ValueError(f"Reference result has missing MAE values: {csv_path}")
        paths[route_name] = str(csv_path.resolve())
    return paths


def canonical_split_hashes(reference_root: Path) -> dict[str, str]:
    split_root = reference_root / "splits"
    hashes: dict[str, str] = {}
    for route_name, filename in SPLIT_FILENAMES.items():
        path = split_root / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Missing canonical downstream split for {route_name}: {path}. "
                "The ablations must reuse the reference folds."
            )
        hashes[filename] = sha256(path)
    return hashes


def validate_reference_encoder(reference_dir: Path, random_seed: int) -> dict[str, str]:
    manifest_path = reference_dir / "model_manifest.json"
    split_manifest_path = reference_dir / "split" / "split_manifest.json"
    weights_path = reference_dir / "fold_1" / "best.weights.h5"
    for path in (manifest_path, split_manifest_path, weights_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing reference encoder artifact: {path}")

    model_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_model = {
        "model_type": "fingerprint_siamese_pretrain",
        "fingerprint_source": FINGERPRINT_SOURCE,
        "wide_dim": 1536,
        "deep_dim": 512,
        "use_logp": True,
        "use_molvol": True,
    }
    for key, expected in expected_model.items():
        if model_manifest.get(key) != expected:
            raise ValueError(
                f"Reference encoder manifest mismatch for {key}: "
                f"expected {expected!r}, found {model_manifest.get(key)!r}."
            )

    config_path = Path(model_manifest.get("config_path", ""))
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Missing reference pretraining config: {config_path}")
    pretrain_config = load_yaml(config_path)
    if float(pretrain_config.get("loss", {}).get("lambda_similarity", 0.0)) <= 0:
        raise ValueError("The reference encoder config does not enable the Tanimoto objective.")
    if not pretrain_config.get("tasks", {}).get("use_logp", False):
        raise ValueError("The reference encoder config does not enable the logP objective.")
    if not pretrain_config.get("tasks", {}).get("use_molvol", False):
        raise ValueError("The reference encoder config does not enable the MolVol objective.")

    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    expected_split = {
        "strategy": "stratified_hmdb_90_10",
        "fingerprint_source": FINGERPRINT_SOURCE,
        "random_seed": random_seed,
        "train_fraction_requested": 0.9,
        "validation_fraction_requested": 0.1,
        "test_partition": None,
    }
    for key, expected in expected_split.items():
        if split_manifest.get(key) != expected:
            raise ValueError(
                f"Reference HMDB split mismatch for {key}: "
                f"expected {expected!r}, found {split_manifest.get(key)!r}."
            )
    return {
        "model_manifest": str(manifest_path.resolve()),
        "model_manifest_sha256": sha256(manifest_path),
        "split_manifest": str(split_manifest_path.resolve()),
        "split_manifest_sha256": sha256(split_manifest_path),
        "weights": str(weights_path.resolve()),
        "weights_sha256": sha256(weights_path),
        "pretrain_config": str(config_path.resolve()),
        "pretrain_config_sha256": sha256(config_path),
    }


def pretrain_command(job: AblationJob, python: str) -> list[str] | None:
    if job.spec.scratch:
        return None
    assert job.pretrain_module is not None
    assert job.pretrain_config_path is not None
    return [python, "-m", job.pretrain_module, "--config", str(job.pretrain_config_path)]


def validate_pretrained_checkpoint(job: AblationJob) -> dict[str, str]:
    if job.spec.scratch or job.checkpoint_dir is None:
        raise ValueError("Random initialization has no pretrained checkpoint to validate.")
    manifest_path = job.checkpoint_dir / "model_manifest.json"
    weights_path = job.checkpoint_dir / "fold_1" / "best.weights.h5"
    for path in (manifest_path, weights_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing pretrained ablation artifact: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "fingerprint_source": FINGERPRINT_SOURCE,
        "pretraining_protocol": PRETRAINING_PROTOCOL,
        "use_logp": job.spec.use_logp,
        "use_molvol": job.spec.use_molvol,
    }
    if job.spec.architecture == "wide_deep":
        expected.update(
            {
                "model_type": "fingerprint_siamese_pretrain",
                "wide_dim": 1536,
                "deep_dim": 512,
            }
        )
    elif job.spec.architecture == "wide_only":
        expected.update(
            {
                "model_type": "fingerprint_siamese_pretrain_wide_only",
                "branch_mode": "wide_only",
                "wide_dim": 1536,
            }
        )
    else:
        expected.update(
            {
                "model_type": "fingerprint_siamese_pretrain_deep_only",
                "branch_mode": "deep_only",
                "deep_dim": 512,
            }
        )
    for key, expected_value in expected.items():
        if manifest.get(key) != expected_value:
            raise ValueError(
                f"Ablation checkpoint {job.spec.name} has invalid {key}: "
                f"expected {expected_value!r}, found {manifest.get(key)!r}."
            )
    return {
        "model_manifest_sha256": sha256(manifest_path),
        "weights_sha256": sha256(weights_path),
    }


def downstream_command(
    job: AblationJob,
    *,
    python: str,
    ccs_config: Path,
    folds: int,
    random_seed: int,
) -> list[str]:
    command = [
        python,
        "-m",
        "src.models.ablations.run_dgr_ablation",
        "--architecture",
        job.spec.architecture,
        "--config",
        str(ccs_config),
        "--folds",
        str(folds),
        "--random-seed",
        str(random_seed),
        "--fingerprint-source",
        FINGERPRINT_SOURCE,
        "--output-root",
        str(job.downstream_dir),
        "--experiment-tag",
        job.spec.name,
    ]
    if job.spec.scratch:
        command.append("--scratch")
    else:
        assert job.checkpoint_dir is not None
        command.extend(["--siamese-results-dir", str(job.checkpoint_dir)])
    return command


def run_command(command: list[str], log_path: Path, dry_run: bool = False) -> None:
    print(f"$ {shlex.join(command)}")
    print(f"  log: {log_path}")
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"$ {shlex.join(command)}\n\n")
        log_handle.flush()
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_handle.write(line)
            log_handle.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def discover_downstream_results(job: AblationJob, folds: int) -> dict[str, str]:
    folds_tag = "single_fold" if folds == 1 else "five_folds"
    results: dict[str, str] = {}
    for route_name, (train_db, test_db) in ROUTES.items():
        pattern = (
            f"train_val_{train_db}_test_{test_db}_{FINGERPRINT_SOURCE}_*"
            f"{MODEL_TYPE}_{folds_tag}/*_results.csv"
        )
        matches = sorted(job.downstream_dir.glob(pattern))
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one downstream result for {job.spec.name}/{route_name}; "
                f"found {len(matches)} using {pattern!r}."
            )
        with matches[0].open(newline="", encoding="utf-8") as handle:
            rows = [
                row
                for row in csv.DictReader(handle)
                if str(row.get("Fold", "")).strip().lower() != "total"
            ]
        if len(rows) != folds:
            raise RuntimeError(
                f"Expected {folds} folds in {matches[0]}; found {len(rows)}."
            )
        results[route_name] = str(matches[0].resolve())
    return results


def job_manifest(job: AblationJob) -> dict[str, Any]:
    architecture = ARCHITECTURES[job.spec.architecture]
    return {
        **asdict(job.spec),
        "lambda_similarity": job.spec.lambda_similarity,
        "model_type": MODEL_TYPE,
        "pretraining_protocol": None if job.spec.scratch else PRETRAINING_PROTOCOL,
        "wide_dim": architecture["wide_dim"],
        "deep_dim": architecture["deep_dim"],
        "pretrain_config": (
            None if job.pretrain_config_path is None else str(job.pretrain_config_path.resolve())
        ),
        "checkpoint_dir": None if job.checkpoint_dir is None else str(job.checkpoint_dir.resolve()),
        "downstream_dir": str(job.downstream_dir.resolve()),
        "results": {},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run all nine RDKit Siamese ablations sequentially with the DGR-MLP CCS head. "
            "The pretrained all-task Wide+Deep reference is validated and reused, not retrained."
        )
    )
    parser.add_argument("--fingerprint-source", choices=(FINGERPRINT_SOURCE,), default=FINGERPRINT_SOURCE)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--experiment-tag", default=DEFAULT_EXPERIMENT_TAG)
    parser.add_argument("--ccs-config", type=Path, default=DEFAULT_CCS_CONFIG)
    parser.add_argument(
        "--reference-results-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "CCSTrainer",
    )
    parser.add_argument(
        "--reference-siamese-results-dir",
        type=Path,
        default=DEFAULT_REFERENCE_SIAMESE_DIR,
    )
    parser.add_argument("--reference-experiment-tag", default=DEFAULT_REFERENCE_TAG)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the complete sequential command list without writing files.",
    )
    args = parser.parse_args()
    if args.folds < 1 or args.folds > 5:
        raise ValueError("--folds must be between 1 and 5.")
    if not args.experiment_tag.strip():
        raise ValueError("--experiment-tag cannot be empty.")
    if "/" in args.experiment_tag or "\\" in args.experiment_tag:
        raise ValueError("--experiment-tag must be a directory name, not a path.")
    if not args.ccs_config.exists():
        raise FileNotFoundError(f"CCS config not found: {args.ccs_config}")
    return args


def main() -> None:
    args = parse_args()
    experiment_root = ABLATION_ROOT / args.experiment_tag
    reference_results = reference_result_paths(
        args.reference_results_root,
        args.reference_experiment_tag,
        args.folds,
    )
    split_hashes = canonical_split_hashes(args.reference_results_root)
    reference_encoder = validate_reference_encoder(
        args.reference_siamese_results_dir, args.random_seed
    )
    jobs = build_jobs(experiment_root)

    print("DGR-MLP Siamese ablation suite")
    print(f"Reference: {args.reference_experiment_tag} (validated, not retrained)")
    print(f"Output: {experiment_root}")
    print(
        f"Jobs: {len(jobs)} sequential ablations; "
        f"{len(jobs) * len(ROUTES) * args.folds} downstream folds"
    )

    if args.dry_run:
        for job in jobs:
            print(f"\n[{job.spec.name}] {job.spec.label}")
            pretrain = pretrain_command(job, args.python)
            if pretrain is not None:
                run_command(pretrain, experiment_root / "logs" / f"{job.spec.name}_pretrain.log", True)
            run_command(
                downstream_command(
                    job,
                    python=args.python,
                    ccs_config=args.ccs_config,
                    folds=args.folds,
                    random_seed=args.random_seed,
                ),
                experiment_root / "logs" / f"{job.spec.name}_downstream.log",
                True,
            )
        return

    if experiment_root.exists():
        raise FileExistsError(
            f"Experiment directory already exists: {experiment_root}. "
            "This suite is one-shot and does not overwrite or resume previous runs."
        )
    experiment_root.mkdir(parents=True)

    manifest_path = experiment_root / "experiment_manifest.json"
    manifest: dict[str, Any] = {
        "state": "running",
        "started_at": utc_now(),
        "experiment_tag": args.experiment_tag,
        "fingerprint_source": FINGERPRINT_SOURCE,
        "model_type": MODEL_TYPE,
        "pretraining_protocol": PRETRAINING_PROTOCOL,
        "folds": args.folds,
        "random_seed": args.random_seed,
        "ccs_config": str(args.ccs_config.resolve()),
        "ccs_config_sha256": sha256(args.ccs_config),
        "reference": {
            "label": "All pretraining tasks",
            "experiment_tag": args.reference_experiment_tag,
            "results": reference_results,
            "result_sha256": {
                route: sha256(Path(path)) for route, path in reference_results.items()
            },
            "split_sha256": split_hashes,
            "encoder": reference_encoder,
        },
        "ablations": [job_manifest(job) for job in jobs],
    }
    write_json(manifest_path, manifest)

    try:
        for index, job in enumerate(jobs):
            print(f"\n[{index + 1}/{len(jobs)}] {job.spec.label} ({job.spec.name})")
            if not job.spec.scratch:
                assert job.pretrain_config_path is not None
                assert job.checkpoint_dir is not None
                config = build_pretrain_config(job.spec, job.checkpoint_dir, args.random_seed)
                write_yaml(job.pretrain_config_path, config)
                manifest["ablations"][index]["pretrain_config_sha256"] = sha256(
                    job.pretrain_config_path
                )
                command = pretrain_command(job, args.python)
                assert command is not None
                run_command(
                    command,
                    experiment_root / "logs" / f"{job.spec.name}_pretrain.log",
                )
                manifest["ablations"][index]["checkpoint_sha256"] = (
                    validate_pretrained_checkpoint(job)
                )
            run_command(
                downstream_command(
                    job,
                    python=args.python,
                    ccs_config=args.ccs_config,
                    folds=args.folds,
                    random_seed=args.random_seed,
                ),
                experiment_root / "logs" / f"{job.spec.name}_downstream.log",
            )
            observed_split_hashes = canonical_split_hashes(args.reference_results_root)
            if observed_split_hashes != split_hashes:
                raise RuntimeError(
                    "Canonical CCS split files changed during the ablation run; "
                    "fold pairing with the reference is no longer guaranteed."
                )
            manifest["ablations"][index]["results"] = discover_downstream_results(
                job, args.folds
            )
            manifest["ablations"][index]["state"] = "complete"
            write_json(manifest_path, manifest)
    except Exception as exc:
        manifest["state"] = "failed"
        manifest["failed_at"] = utc_now()
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        write_json(manifest_path, manifest)
        raise

    manifest["state"] = "complete"
    manifest["completed_at"] = utc_now()
    write_json(manifest_path, manifest)
    print(f"\nAll nine DGR-MLP ablations completed: {manifest_path}")


if __name__ == "__main__":
    main()
