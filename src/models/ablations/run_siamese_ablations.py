from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


RESULTS_ROOT = Path("results")
ABLATION_ROOT = RESULTS_ROOT / "ablations"
TRAINING_LOSS = "mae"

SOURCES = {
    "alvadesc": {
        "base_pretrain_config": Path("configs/pretrain_siamese_alvadesc.yaml"),
        "pretrain_module": "src.models.pretrain_siamese_alvadesc",
    },
    "rdkit": {
        "base_pretrain_config": Path("configs/pretrain_siamese.yaml"),
        "pretrain_module": "src.models.pretrain_siamese",
    },
}

ABLATIONS = {
    "tanimoto_only": {
        "use_logp": False,
        "use_molvol": False,
        "lambda_similarity": 1.0,
    },
    "tanimoto_logp": {
        "use_logp": True,
        "use_molvol": False,
        "lambda_similarity": 1.0,
    },
    "tanimoto_molvol": {
        "use_logp": False,
        "use_molvol": True,
        "lambda_similarity": 1.0,
    },
    "logp_molvol_no_tanimoto": {
        "use_logp": True,
        "use_molvol": True,
        "lambda_similarity": 0.0,
    },
}


@dataclass(frozen=True)
class AblationJob:
    source: str
    ablation: str
    config_path: Path
    ccs_config_path: Path
    results_name: str
    siamese_results_dir: Path
    pretrain_module: str

    @property
    def label(self) -> str:
        return f"{self.source}_{self.ablation}_mae"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def build_job(
    source: str,
    ablation: str,
    results_prefix: str,
    configs_dir: Path,
    ccs_config_path: Path,
) -> AblationJob:
    source_cfg = SOURCES[source]
    ablation_cfg = ABLATIONS[ablation]
    base_config_path = source_cfg["base_pretrain_config"]
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base pretrain config not found: {base_config_path}")
    if not ccs_config_path.exists():
        raise FileNotFoundError(f"Base CCS config not found: {ccs_config_path}")

    config = load_yaml(base_config_path)
    config.setdefault("tasks", {})
    config.setdefault("loss", {})
    config.setdefault("training", {})

    config["tasks"]["use_logp"] = bool(ablation_cfg["use_logp"])
    config["tasks"]["use_molvol"] = bool(ablation_cfg["use_molvol"])
    config["loss"]["lambda_similarity"] = float(ablation_cfg["lambda_similarity"])

    experiment_tag = f"{ablation}_mae"
    results_name = f"{results_prefix}_{source}_{experiment_tag}"
    config["training"]["results_name"] = results_name
    config["ablation"] = {
        "source": source,
        "name": ablation,
        "training_loss": TRAINING_LOSS,
        "use_logp": bool(ablation_cfg["use_logp"]),
        "use_molvol": bool(ablation_cfg["use_molvol"]),
        "lambda_similarity": float(ablation_cfg["lambda_similarity"]),
        "base_pretrain_config": str(base_config_path),
    }

    config_path = configs_dir / f"pretrain_{source}_{experiment_tag}.yaml"
    write_yaml(config_path, config)

    ccs_config = load_yaml(ccs_config_path)
    ccs_config.setdefault("ablation", {})
    ccs_config["ablation"].update({
        "source": source,
        "name": ablation,
        "training_loss": TRAINING_LOSS,
        "base_ccs_config": str(ccs_config_path),
        "siamese_results_dir": str(RESULTS_ROOT / results_name),
    })
    generated_ccs_config_path = configs_dir / f"ccs_{source}_{experiment_tag}.yaml"
    write_yaml(generated_ccs_config_path, ccs_config)

    return AblationJob(
        source=source,
        ablation=ablation,
        config_path=config_path,
        ccs_config_path=generated_ccs_config_path,
        results_name=results_name,
        siamese_results_dir=RESULTS_ROOT / results_name,
        pretrain_module=str(source_cfg["pretrain_module"]),
    )


def command_to_string(command: list[str]) -> str:
    return " ".join(command)


def run_command(command: list[str], log_path: Path, dry_run: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"    $ {command_to_string(command)}")
    print(f"    log: {log_path}")
    if dry_run:
        return

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {command_to_string(command)}\n\n")
        log_file.flush()
        subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
        )


def pretrain_weights_exist(job: AblationJob) -> bool:
    return (job.siamese_results_dir / "fold_1" / "best.weights.h5").exists()


def completion_marker_path(markers_dir: Path, job: AblationJob) -> Path:
    return markers_dir / f"{job.label}.json"


def write_completion_marker(path: Path, job: AblationJob, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": job.source,
        "ablation": job.ablation,
        "training_loss": TRAINING_LOSS,
        "pretrain_config": str(job.config_path),
        "ccs_config": str(job.ccs_config_path),
        "siamese_results_dir": str(job.siamese_results_dir),
        "folds": args.folds,
        "random_seed": args.random_seed,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_pipeline(job: AblationJob, args: argparse.Namespace) -> None:
    print(f"[{job.label}] Starting")

    pretrain_command = [
        args.python,
        "-m",
        job.pretrain_module,
        "--config",
        str(job.config_path),
    ]
    ccs_command = [
        args.python,
        "-m",
        "src.models.run_siamese",
        "--config",
        str(job.ccs_config_path),
        "--folds",
        str(args.folds),
        "--random-seed",
        str(args.random_seed),
        "--fingerprint-source",
        job.source,
        "--siamese-results-dir",
        str(job.siamese_results_dir),
        "--experiment-tag",
        f"{job.ablation}_mae",
    ]

    pretrain_log = args.logs_dir / f"{job.label}_pretrain.log"
    ccs_log = args.logs_dir / f"{job.label}_run_siamese.log"
    marker_path = completion_marker_path(args.markers_dir, job)

    if args.skip_existing_pretrain and pretrain_weights_exist(job):
        print(f"[{job.label}] Skipping pretrain: weights already exist")
    else:
        run_command(pretrain_command, pretrain_log, args.dry_run)

    if args.skip_existing_ccs and marker_path.exists():
        print(f"[{job.label}] Skipping run_siamese: completion marker already exists")
    else:
        run_command(ccs_command, ccs_log, args.dry_run)

    if not args.dry_run:
        write_completion_marker(marker_path, job, args)
    print(f"[{job.label}] Done")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Siamese pretraining and downstream CCS ablations for AlvaDesc and RDKit fingerprints."
    )
    parser.add_argument("--sources", nargs="+", choices=sorted(SOURCES), default=list(SOURCES))
    parser.add_argument("--ablations", nargs="+", choices=sorted(ABLATIONS), default=list(ABLATIONS))
    parser.add_argument("--ccs-config", type=Path, default=Path("configs/ccs_prediction_heads.yaml"))
    parser.add_argument("--folds", type=int, default=5, help="Number of CCS folds to run, from 1 to 5.")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--results-prefix", default="Siamese_ablation")
    parser.add_argument("--configs-dir", type=Path, default=ABLATION_ROOT / "configs")
    parser.add_argument("--logs-dir", type=Path, default=ABLATION_ROOT / "logs")
    parser.add_argument("--markers-dir", type=Path, default=ABLATION_ROOT / "completed")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for subprocesses.")
    parser.add_argument("--max-workers", type=int, default=1, help="Number of ablation pipelines to run in parallel.")
    parser.add_argument("--skip-existing-pretrain", action="store_true")
    parser.add_argument("--skip-existing-ccs", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write generated configs and print commands without running them.")
    args = parser.parse_args()

    if args.max_workers < 1:
        raise ValueError("--max-workers must be >= 1")
    if args.folds < 1 or args.folds > 5:
        raise ValueError("--folds must be between 1 and 5")
    if not args.ccs_config.exists():
        raise FileNotFoundError(f"CCS config not found: {args.ccs_config}")
    return args


def main() -> None:
    args = parse_args()
    jobs = [
        build_job(
            source=source,
            ablation=ablation,
            results_prefix=args.results_prefix,
            configs_dir=args.configs_dir,
            ccs_config_path=args.ccs_config,
        )
        for source in args.sources
        for ablation in args.ablations
    ]

    print(f"Prepared {len(jobs)} ablation pipelines")
    for job in jobs:
        print(f"  - {job.label}: {job.config_path} -> {job.siamese_results_dir}")

    if args.max_workers == 1:
        for job in jobs:
            run_pipeline(job, args)
        return

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(run_pipeline, job, args): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"[{job.label}] Failed: {exc}")
                raise


if __name__ == "__main__":
    main()
