"""Load HMDB RDKit fingerprints and chemical classifications."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.load_hmdb import HMDB_CLASSIFICATIONS_TSV, load_hmdb


RAW_HMDB_RDKIT_CSV = Path("resources/fingerprints/hmdb_rdkit.csv")


def load_hmdb_rdkit(
    input_csv: str | Path = RAW_HMDB_RDKIT_CSV,
    classifications_tsv: str | Path = HMDB_CLASSIFICATIONS_TSV,
) -> pd.DataFrame:
    """Load HMDB RDKit fingerprints and chemical classifications."""

    return load_hmdb(
        input_csv=input_csv,
        classifications_tsv=classifications_tsv,
    )
