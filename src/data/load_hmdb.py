"""Load HMDB fingerprints and chemical classifications."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


RAW_HMDB_CSV = Path("resources/fingerprints/hmdb.csv")
HMDB_CLASSIFICATIONS_TSV = Path("resources/classifications/hmdb_classifications.tsv")
MIN_CLASS_COUNT = 100


def load_hmdb(
    input_csv: str | Path = RAW_HMDB_CSV,
    classifications_tsv: str | Path = HMDB_CLASSIFICATIONS_TSV,
) -> pd.DataFrame:
    """Load HMDB fingerprints and chemical classifications.

    Returns one row per InChI with columns ``inchi`` and the fingerprint columns
    from the source CSV. Fingerprint values are numeric ``float32`` values
    clipped to the binary range ``[0, 1]``. The returned dataframe also includes
    ``classification`` for grouped fingerprint training.
    """

    input_csv = Path(input_csv)
    classifications_tsv = Path(classifications_tsv)

    columns = _read_columns(input_csv)
    _validate_columns(columns, input_csv)

    fingerprint_columns = _fingerprint_columns(columns)
    usecols = ["inchi", *fingerprint_columns]

    raw = pd.read_csv(input_csv, usecols=usecols, low_memory=False)
    processed = _clean_fingerprints(raw, fingerprint_columns)
    processed = _add_classifications(processed, classifications_tsv)

    return processed


def _read_columns(csv_path: Path) -> list[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"HMDB input CSV not found: {csv_path}")
    return list(pd.read_csv(csv_path, nrows=0).columns)


def _validate_columns(columns: list[str], csv_path: Path) -> None:
    if "inchi" not in columns:
        raise ValueError(f"HMDB CSV must contain an 'inchi' column: {csv_path}")
    if not _fingerprint_columns(columns):
        raise ValueError(f"HMDB CSV must contain fingerprint columns named V*: {csv_path}")


def _fingerprint_columns(columns: list[str]) -> list[str]:
    return [column for column in columns if column.startswith("V")]


def _clean_fingerprints(raw: pd.DataFrame, fingerprint_columns: list[str]) -> pd.DataFrame:
    df = raw.copy()

    df["inchi"] = df["inchi"].map(_clean_text)
    missing_inchi_mask = df["inchi"].isna()
    df = df.loc[~missing_inchi_mask].reset_index(drop=True)

    numeric_fingerprints = df[fingerprint_columns].apply(pd.to_numeric, errors="coerce")
    invalid_fingerprint_mask = numeric_fingerprints.isna().any(axis=1)
    df = df.loc[~invalid_fingerprint_mask].reset_index(drop=True)
    numeric_fingerprints = numeric_fingerprints.loc[~invalid_fingerprint_mask].reset_index(drop=True)

    numeric_fingerprints = numeric_fingerprints.astype("float32").clip(lower=0.0, upper=1.0)
    empty_fingerprint_mask = numeric_fingerprints.sum(axis=1) == 0
    df = df.loc[~empty_fingerprint_mask].reset_index(drop=True)
    numeric_fingerprints = numeric_fingerprints.loc[~empty_fingerprint_mask].reset_index(drop=True)

    duplicate_inchi_mask = df.duplicated(subset=["inchi"])
    df = df.loc[~duplicate_inchi_mask].reset_index(drop=True)
    numeric_fingerprints = numeric_fingerprints.loc[~duplicate_inchi_mask].reset_index(drop=True)

    df = pd.concat(
        [
            df[["inchi"]].reset_index(drop=True),
            numeric_fingerprints[fingerprint_columns].reset_index(drop=True),
        ],
        axis=1,
    ).copy()

    return df


def _add_classifications(df: pd.DataFrame, classifications_tsv: Path) -> pd.DataFrame:
    if not classifications_tsv.exists():
        raise FileNotFoundError(f"HMDB classifications TSV not found: {classifications_tsv}")

    class_df = pd.read_csv(classifications_tsv, sep="\t", usecols=["inchi", "superclass"])
    class_df["inchi"] = class_df["inchi"].map(_clean_text)
    class_df["superclass"] = class_df["superclass"].map(_clean_text)
    class_df = class_df.dropna(subset=["inchi"]).drop_duplicates(subset=["inchi"])

    merged = df.merge(class_df, on="inchi", how="left")
    merged["classification"] = merged["superclass"].fillna("Unknown")

    class_counts = merged["classification"].value_counts()
    scarce_classes = class_counts[class_counts < MIN_CLASS_COUNT].index
    merged["classification"] = merged["classification"].replace(scarce_classes, "Scarce")
    return merged.drop(columns=["superclass"])


def _clean_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


if __name__ == "__main__":
    df = load_hmdb()
    print(f"Loaded HMDB fingerprints: {len(df):,}")
