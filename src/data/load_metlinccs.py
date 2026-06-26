"""Load METLIN-CCS fingerprints and physchem descriptors."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_ccsbase import (
    merge_physchem_descriptors,
    resolve_ccs_duplicates,
)


METLINCCS1_ZIP = Path("data/fingerprints/metlinccs1.zip")
METLINCCS2_ZIP = Path("data/fingerprints/metlinccs2.zip")
RAW_METLINCCS_CSV = Path("resources/fingerprints/metlinccs.csv")
METLINCCS_DESCRIPTORS_CSV = Path("resources/descriptors/metlinccs_physchem.csv")


def build_metlinccs_raw_csv_from_sources(
    metlinccs1_zip: str | Path = METLINCCS1_ZIP,
    metlinccs2_zip: str | Path = METLINCCS2_ZIP,
    output_csv: str | Path = RAW_METLINCCS_CSV,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Build one normalized METLIN-CCS CSV from the two zip sources."""

    output_csv = Path(output_csv)
    if output_csv.exists() and not overwrite:
        return pd.read_csv(output_csv, low_memory=False)

    df1 = _load_metlinccs1_from_zip(Path(metlinccs1_zip))
    df2 = _load_metlinccs2_from_zip(Path(metlinccs2_zip))

    if not df1.empty and not df2.empty:
        merged = pd.concat([df1, df2], ignore_index=True)
    elif not df1.empty:
        merged = df1
    else:
        merged = df2

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    return merged


def load_metlinccs(
    input_csv: str | Path = RAW_METLINCCS_CSV,
    descriptor_csv: str | Path = METLINCCS_DESCRIPTORS_CSV,
    add_descriptors: bool = True,
) -> pd.DataFrame:
    """Load normalized METLIN-CCS CSV, clean/deduplicate, and optionally merge descriptors."""

    input_csv = Path(input_csv)
    descriptor_csv = Path(descriptor_csv)

    if not input_csv.exists():
        raise FileNotFoundError(
            f"METLIN-CCS normalized CSV not found: {input_csv}. "
            "Run build_metlinccs_raw_csv_from_sources() first."
        )

    raw = pd.read_csv(input_csv, low_memory=False)
    cleaned = clean_metlinccs_dataframe(raw)
    output_df = cleaned
    if add_descriptors:
        output_df = merge_physchem_descriptors(cleaned, descriptor_csv=descriptor_csv)

    return output_df


def clean_metlinccs_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean METLIN-CCS rows for CCS training."""

    df = raw.copy()
    df = _clean_ccs(df)
    required = ["inchi", "adduct", "ccs"]
    missing_required = [column for column in required if column not in df.columns]
    if missing_required:
        raise ValueError(f"METLIN-CCS data is missing required columns: {missing_required}")

    df = df.dropna(subset=required).copy()

    df = resolve_ccs_duplicates(df)

    fingerprint_columns = [column for column in df.columns if column.startswith("V")]
    if not fingerprint_columns:
        raise ValueError("METLIN-CCS data must contain fingerprint columns named V*.")
    df = df[["inchi", "adduct", "ccs", *fingerprint_columns]]

    return df


def _load_metlinccs1_from_zip(zip_path: Path) -> pd.DataFrame:
    df = _read_single_csv_from_zip(zip_path)
    if df.empty:
        return pd.DataFrame()

    if "InChI" in df.columns and "inchi" not in df.columns:
        df = df.rename(columns={"InChI": "inchi"})

    df["adduct"] = df.apply(_standardise_metlinccs1_adduct, axis=1)
    df["ccs"] = df.apply(_average_ccs_metlin1, axis=1)
    df = _clean_ccs(df)
    df = df.dropna(subset=["inchi", "ccs", "adduct"]).copy()

    fingerprint_columns = [column for column in df.columns if column.startswith("V")]
    df = df[["inchi", "adduct", "ccs", *fingerprint_columns]].copy()
    df["source_dataset"] = "metlinccs1"
    return df


def _load_metlinccs2_from_zip(zip_path: Path) -> pd.DataFrame:
    df = _read_single_csv_from_zip(zip_path)
    if df.empty:
        return pd.DataFrame()

    adduct_map = {
        "CCS [M+Na]+": "[M+Na]+",
        "CCS [M+H]+": "[M+H]+",
        "CCS [M-H]-": "[M-H]-",
        "CCS [M+NH4]+": "[M+NH4]+",
        "CCS [M+H-H2O]+": "[M+H-H2O]+",
        "CCS [M+Cl]-": "[M+Cl]-",
        "CCS [M-H+FA]-": "[M-H+FA]-",
    }
    id_columns = ["InChI", "Name", "Formula", *[column for column in df.columns if column.startswith("V")]]
    df_long = _expand_metlinccs2(df, adduct_map, id_columns)
    df_long = df_long.rename(columns={"InChI": "inchi"})

    allowed = {"[M+H]+", "[M-H]-", "[M+Na]+"}
    df_long = df_long[df_long["adduct"].isin(allowed)].copy()
    df_long = _clean_ccs(df_long)
    df_long = df_long.dropna(subset=["inchi", "ccs", "adduct"]).copy()
    df_long["source_dataset"] = "metlinccs2"
    return df_long


def _read_single_csv_from_zip(zip_path: Path) -> pd.DataFrame:
    if not zip_path.exists():
        return pd.DataFrame()

    extract_root = Path("/tmp") / zip_path.stem
    extract_root.mkdir(parents=True, exist_ok=True)
    extract_dir = extract_root / zip_path.stem
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    files = sorted(path for path in extract_dir.iterdir() if path.suffix in {".csv", ".tsv"})
    if not files:
        shutil.rmtree(extract_root, ignore_errors=True)
        return pd.DataFrame()
    if len(files) == 1:
        df = pd.read_csv(files[0], low_memory=False)
    else:
        df = _merge_multi_part(files)
    shutil.rmtree(extract_root, ignore_errors=True)
    return df


def _merge_multi_part(files: list[Path]) -> pd.DataFrame:
    dfs = []
    base_columns = None
    for index, file_path in enumerate(files):
        skip = 1 if index > 0 else 0
        part = pd.read_csv(file_path, skiprows=skip, low_memory=False)
        if index == 0:
            base_columns = part.columns.tolist()
        else:
            part.columns = base_columns
        dfs.append(part)
    return pd.concat(dfs, ignore_index=True)


def _standardise_metlinccs1_adduct(row) -> str:
    adduct_raw = str(row.get("Adduct")).strip()
    dimer_val = str(row.get("Dimer.1")).strip().lower()
    mapping = {
        "[M+H]": "[M+H]+",
        "[M-H]": "[M-H]-",
        "[M+Na]": "[M+Na]+",
    }
    std = mapping.get(adduct_raw, adduct_raw)
    if dimer_val == "dimer" and std in {"[M+H]+", "[M-H]-", "[M+Na]+"}:
        std = std.replace("[M+", "[2M+").replace("[M-", "[2M-")
    return std


def _average_ccs_metlin1(row):
    ccs_values = [row.get(column) for column in ["CCS1", "CCS2", "CCS3"] if _notna(row.get(column))]
    return np.mean(ccs_values) if ccs_values else row.get("CCS_AVG")


def _expand_metlinccs2(df, adduct_map: dict[str, str], id_columns: list[str]):
    records = []
    for _, row in df.iterrows():
        for column, adduct in adduct_map.items():
            if column in df.columns and _notna(row[column]):
                record = {key: row[key] for key in id_columns if key in df.columns}
                record["adduct"] = adduct
                record["ccs"] = row[column]
                records.append(record)
    return pd.DataFrame(records)


def _clean_ccs(df: pd.DataFrame) -> pd.DataFrame:
    if "ccs" not in df.columns or df.empty:
        return df
    df = df.copy()
    df["ccs"] = df["ccs"].astype(str).str.strip().str.replace(",", ".", regex=False)
    df["ccs"] = pd.to_numeric(df["ccs"], errors="coerce")
    return df


def _notna(value) -> bool:
    return bool(pd.notna(value))


if __name__ == "__main__":
    build_metlinccs_raw_csv_from_sources()
    df = load_metlinccs()
    print(f"Loaded METLIN-CCS rows: {len(df):,}")
