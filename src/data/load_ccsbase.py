"""Load CCSBase fingerprints and physchem descriptors."""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger


RAW_CCSBASE_CSV = Path("resources/fingerprints/ccsbase.csv")
CCSBASE_DESCRIPTORS_CSV = Path("resources/descriptors/ccsbase_physchem.csv")

ALLOWED_ADDUCTS = frozenset({"[M+H]+", "[M-H]-", "[M+Na]+", "[2M+H]+", "[2M-H]-", "[2M+Na]+"})
MOLECULAR_FEATURE_COLUMNS = [
    "MolWt",
    "HeavyAtomCount",
    "TPSA",
    "MolLogP",
    "NumRotatableBonds",
    "RingCount",
    "FractionCSP3",
    "mol_volume_mean",
]
MOLECULAR_FEATURE_SCALES = {
    "MolWt": 100.0,
    "HeavyAtomCount": 100.0,
    "TPSA": 100.0,
    "MolLogP": 1.0,
    "NumRotatableBonds": 10.0,
    "RingCount": 10.0,
    "FractionCSP3": 1.0,
    "mol_volume_mean": 100.0,
}
MIN_CCS_RELATIVE_DUPLICATE_AGREEMENT = 0.03


def load_ccsbase(
    input_csv: str | Path = RAW_CCSBASE_CSV,
    descriptor_csv: str | Path = CCSBASE_DESCRIPTORS_CSV,
    add_descriptors: bool = True,
) -> pd.DataFrame:
    """Load CCSBase, clean fingerprints/targets, and optionally merge descriptors."""

    input_csv = Path(input_csv)
    descriptor_csv = Path(descriptor_csv)

    raw = pd.read_csv(input_csv, low_memory=False)
    cleaned = clean_ccsbase_dataframe(raw)
    if add_descriptors:
        cleaned = merge_physchem_descriptors(cleaned, descriptor_csv=descriptor_csv)

    return cleaned


def clean_ccsbase_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean CCSBase rows for CCS training."""

    df = raw.copy()

    df = _ensure_inchi(df)
    df = _rename_ccsbase_columns(df)
    df = _clean_ccs(df)

    required = ["inchi", "adduct", "ccs"]
    missing_required = [column for column in required if column not in df.columns]
    if missing_required:
        raise ValueError(f"CCSBase data is missing required columns after normalization: {missing_required}")

    df_clean = df.dropna(subset=required).reset_index(drop=True)
    df_clean["adduct"] = df_clean["adduct"].astype(str).str.strip()

    df_clean = df_clean[df_clean["adduct"].isin(ALLOWED_ADDUCTS)].reset_index(drop=True)

    fingerprint_columns = [column for column in df_clean.columns if column.startswith("V")]
    if not fingerprint_columns:
        raise ValueError("CCSBase data must contain fingerprint columns named V*.")

    keep_columns = ["inchi", "adduct", "ccs", *fingerprint_columns]
    df_clean = df_clean[keep_columns].copy()
    df_clean = resolve_ccs_duplicates(df_clean)

    return df_clean


def merge_physchem_descriptors(
    df: pd.DataFrame,
    descriptor_csv: str | Path = CCSBASE_DESCRIPTORS_CSV,
) -> pd.DataFrame:
    """Merge cached physchem descriptors by InChI."""

    descriptor_csv = Path(descriptor_csv)
    if not descriptor_csv.exists():
        raise FileNotFoundError(f"Descriptor CSV not found: {descriptor_csv}")
    if "inchi" not in df.columns:
        raise ValueError("Cannot merge descriptors: 'inchi' is missing from dataframe.")

    descriptor_df = pd.read_csv(
        descriptor_csv,
        usecols=["inchi", *MOLECULAR_FEATURE_COLUMNS],
        low_memory=False,
    )
    descriptor_df = descriptor_df.drop_duplicates(subset="inchi", keep="last")
    return df.merge(descriptor_df, on="inchi", how="left")


def resolve_ccs_duplicates(
    df: pd.DataFrame,
    ccs_threshold: float = MIN_CCS_RELATIVE_DUPLICATE_AGREEMENT,
) -> pd.DataFrame:
    """Average agreeing duplicate CCS measurements and drop conflicting ones."""

    if df is None or df.empty:
        return df

    work_df = df.reset_index(drop=True).copy()
    keep_rows = []

    for _, group in work_df.groupby(["inchi", "adduct"], sort=False):
        if len(group) == 1:
            keep_rows.append(group.iloc[0])
            continue

        ccs_values = pd.to_numeric(group["ccs"], errors="coerce").values.astype(float)
        mean_ccs = ccs_values.mean()
        if mean_ccs == 0:
            continue
        if all(abs(ccs - mean_ccs) / mean_ccs <= ccs_threshold for ccs in ccs_values):
            row = group.iloc[0].copy()
            row["ccs"] = mean_ccs
            keep_rows.append(row)

    if not keep_rows:
        return work_df.iloc[0:0].copy()
    return pd.DataFrame(keep_rows).reset_index(drop=True)


def _ensure_inchi(df: pd.DataFrame) -> pd.DataFrame:
    if "inchi" in df.columns and not df["inchi"].isna().all():
        return df
    smiles_col = next((column for column in ["smi", "SMI", "SMILES", "smiles", "Smiles"] if column in df.columns), None)
    if smiles_col is None:
        raise ValueError("Neither 'inchi' nor SMILES/SMI column found in CCSBase data.")

    RDLogger.DisableLog("rdApp.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="rdkit")

    def smiles_to_inchi(smiles):
        if pd.isna(smiles) or not isinstance(smiles, str) or not str(smiles).strip():
            return None
        try:
            mol = Chem.MolFromSmiles(str(smiles).strip())
            return Chem.MolToInchi(mol) if mol is not None else None
        except Exception:
            return None

    df = df.copy()
    df["inchi"] = df[smiles_col].apply(smiles_to_inchi)
    RDLogger.EnableLog("rdApp.*")
    return df


def _rename_ccsbase_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {"Adduct": "adduct", "CCS": "ccs", "InChI": "inchi", "Inchi": "inchi", "SMI": "smi"}
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
    return df


def _clean_ccs(df: pd.DataFrame) -> pd.DataFrame:
    if "ccs" not in df.columns or df.empty:
        return df
    df = df.copy()
    df["ccs"] = df["ccs"].astype(str).str.strip().str.replace(",", ".", regex=False)
    df["ccs"] = pd.to_numeric(df["ccs"], errors="coerce")
    return df


if __name__ == "__main__":
    df = load_ccsbase()
    print(f"Loaded CCSBase rows: {len(df):,}")
