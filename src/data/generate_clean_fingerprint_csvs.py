"""Generate cleaned CCS fingerprint CSV files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.load_ccsbase import (
    ALLOWED_ADDUCTS,
    CCSBASE_DESCRIPTORS_CSV,
    MOLECULAR_FEATURE_COLUMNS,
    RAW_CCSBASE_CSV,
    _clean_ccs as _clean_ccsbase_ccs,
    _ensure_inchi,
    _rename_ccsbase_columns,
    merge_physchem_descriptors,
    resolve_ccs_duplicates,
)
from src.data.load_metlinccs import (
    RAW_METLINCCS_CSV,
    _clean_ccs as _clean_metlinccs_ccs,
    clean_metlinccs_dataframe,
)


FINGERPRINTS_DIR = Path("resources/fingerprints")
CCSBASE_CLEANED_CSV = FINGERPRINTS_DIR / "ccsbase_cleaned.csv"
METLINCCS_CLEANED_CSV = FINGERPRINTS_DIR / "metlinccs_cleaned.csv"


def generate_ccsbase_cleaned(
    input_csv: str | Path = RAW_CCSBASE_CSV,
    descriptor_csv: str | Path = CCSBASE_DESCRIPTORS_CSV,
    output_csv: str | Path = CCSBASE_CLEANED_CSV,
    summary_json: str | Path | None = None,
    add_descriptors: bool | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate the cleaned CCSBase fingerprint CSV and its summary."""

    input_csv = Path(input_csv)
    descriptor_csv = Path(descriptor_csv)
    output_csv = Path(output_csv)
    summary_json = Path(summary_json) if summary_json is not None else _summary_path(output_csv)
    _ensure_can_write(output_csv, overwrite)
    if add_descriptors is None:
        add_descriptors = descriptor_csv.exists()
    if add_descriptors and not descriptor_csv.exists():
        raise FileNotFoundError(f"Descriptor CSV not found: {descriptor_csv}")

    raw = pd.read_csv(input_csv, low_memory=False)
    normalized = _ensure_inchi(raw)
    normalized = _rename_ccsbase_columns(normalized)
    normalized = _clean_ccsbase_ccs(normalized)

    required = ["inchi", "adduct", "ccs"]
    missing_required_rows = int(normalized[required].isna().any(axis=1).sum())
    required_df = normalized.dropna(subset=required).reset_index(drop=True)
    required_df["adduct"] = required_df["adduct"].astype(str).str.strip()

    allowed_mask = required_df["adduct"].isin(ALLOWED_ADDUCTS)
    disallowed_adduct_rows = int((~allowed_mask).sum())
    filtered = required_df.loc[allowed_mask].reset_index(drop=True)

    fingerprint_columns = _fingerprint_columns(filtered)
    pre_duplicate = filtered[["inchi", "adduct", "ccs", *fingerprint_columns]].copy()
    cleaned = resolve_ccs_duplicates(pre_duplicate)
    output_df = cleaned
    if add_descriptors:
        output_df = merge_physchem_descriptors(cleaned, descriptor_csv=descriptor_csv)

    _write_csv(output_df, output_csv)
    summary = {
        "dataset": "ccsbase",
        "input_csv": _path_string(input_csv),
        "descriptor_csv": _path_string(descriptor_csv),
        "output_csv": _path_string(output_csv),
        "add_descriptors": bool(add_descriptors),
        "input_rows": int(len(raw)),
        "input_columns": int(len(raw.columns)),
        "output_rows": int(len(output_df)),
        "output_columns": int(len(output_df.columns)),
        "removed_missing_required_rows": missing_required_rows,
        "removed_disallowed_adduct_rows": disallowed_adduct_rows,
        "removed_duplicate_rows": int(len(pre_duplicate) - len(cleaned)),
        "fingerprint_columns": int(len(fingerprint_columns)),
        "descriptor_columns": list(MOLECULAR_FEATURE_COLUMNS) if add_descriptors else [],
        "allowed_adducts": sorted(ALLOWED_ADDUCTS),
        "adduct_counts": _value_counts(output_df["adduct"]),
    }
    _write_summary(summary, summary_json)
    return summary


def generate_metlinccs_cleaned(
    input_csv: str | Path = RAW_METLINCCS_CSV,
    output_csv: str | Path = METLINCCS_CLEANED_CSV,
    summary_json: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate the cleaned METLIN-CCS fingerprint CSV and its summary."""

    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    summary_json = Path(summary_json) if summary_json is not None else _summary_path(output_csv)
    _ensure_can_write(output_csv, overwrite)

    raw = pd.read_csv(input_csv, low_memory=False)
    normalized = _clean_metlinccs_ccs(raw)

    required = ["inchi", "adduct", "ccs"]
    missing_required_rows = int(normalized[required].isna().any(axis=1).sum())
    required_df = normalized.dropna(subset=required).reset_index(drop=True)

    cleaned = clean_metlinccs_dataframe(raw)
    fingerprint_columns = _fingerprint_columns(cleaned)

    _write_csv(cleaned, output_csv)
    summary = {
        "dataset": "metlinccs",
        "input_csv": _path_string(input_csv),
        "output_csv": _path_string(output_csv),
        "add_descriptors": False,
        "input_rows": int(len(raw)),
        "input_columns": int(len(raw.columns)),
        "output_rows": int(len(cleaned)),
        "output_columns": int(len(cleaned.columns)),
        "removed_missing_required_rows": missing_required_rows,
        "removed_duplicate_rows": int(len(required_df) - len(cleaned)),
        "fingerprint_columns": int(len(fingerprint_columns)),
        "adduct_counts": _value_counts(cleaned["adduct"]),
    }
    _write_summary(summary, summary_json)
    return summary


def _fingerprint_columns(df: pd.DataFrame) -> list[str]:
    columns = [column for column in df.columns if column.startswith("V")]
    if not columns:
        raise ValueError("No fingerprint columns named V* were found.")
    return columns


def _ensure_can_write(output_csv: Path, overwrite: bool) -> None:
    if output_csv.exists() and not overwrite:
        raise FileExistsError(f"{output_csv} already exists. Use --overwrite to replace it.")


def _write_csv(df: pd.DataFrame, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)


def _summary_path(output_csv: Path) -> Path:
    return output_csv.with_name(f"{output_csv.stem}_summary.json")


def _write_summary(summary: dict[str, Any], summary_json: Path) -> None:
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def _value_counts(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts().sort_index().items()}


def _path_string(path: Path) -> str:
    return path.as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate cleaned CCS fingerprint CSV files.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["ccsbase", "metlinccs"],
        default=["ccsbase", "metlinccs"],
        help="Datasets to process. Default: ccsbase metlinccs.",
    )
    parser.add_argument(
        "--ccsbase-descriptors",
        choices=["auto", "always", "never"],
        default="auto",
        help="Merge CCSBase physchem descriptors. Default: auto, only when the descriptor CSV exists.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing cleaned CSV files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    descriptor_mode = {"auto": None, "always": True, "never": False}[args.ccsbase_descriptors]
    if "ccsbase" in args.datasets:
        summary = generate_ccsbase_cleaned(add_descriptors=descriptor_mode, overwrite=args.overwrite)
        print(f"Generated {summary['output_csv']} with {summary['output_rows']:,} rows")
    if "metlinccs" in args.datasets:
        summary = generate_metlinccs_cleaned(overwrite=args.overwrite)
        print(f"Generated {summary['output_csv']} with {summary['output_rows']:,} rows")


if __name__ == "__main__":
    main()
