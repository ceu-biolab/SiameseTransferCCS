"""Generate RDKit fingerprint CSVs.

Fingerprint definition:
Morgan(radius=2, 1024 bits) + RDKit topological(1024 bits) + MACCS(166 bits).
The script reads the active CSVs and writes sibling ``*_rdkit.csv`` files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, MACCSkeys
from tqdm import tqdm


MORGAN_RADIUS = 2
MORGAN_BITS = 1024
RDKIT_TOPOLOGICAL_BITS = 1024
MACCS_BITS = 166
TOTAL_BITS = MORGAN_BITS + RDKIT_TOPOLOGICAL_BITS + MACCS_BITS

FINGERPRINTS_DIR = Path("resources/fingerprints")
DEFAULT_INPUTS = {
    "hmdb": FINGERPRINTS_DIR / "hmdb.csv",
    "ccsbase": FINGERPRINTS_DIR / "ccsbase_cleaned.csv",
    "metlinccs": FINGERPRINTS_DIR / "metlinccs_cleaned.csv",
}


def default_output_for(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_rdkit.csv")


def find_identifier_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    inchi_col = next((col for col in ("inchi", "InChI", "Inchi") if col in df.columns), None)
    smiles_col = next((col for col in ("smi", "SMI", "smiles", "SMILES", "Smiles") if col in df.columns), None)
    return inchi_col, smiles_col


def build_mol(inchi_value: Any, smiles_value: Any):
    if pd.notna(inchi_value) and str(inchi_value).strip():
        try:
            mol = Chem.MolFromInchi(str(inchi_value).strip())
            if mol is not None:
                return mol, "inchi"
        except Exception:
            pass

    if pd.notna(smiles_value) and str(smiles_value).strip():
        try:
            mol = Chem.MolFromSmiles(str(smiles_value).strip())
            if mol is not None:
                return mol, "smiles"
        except Exception:
            pass

    return None, None


def compute_fingerprint_array(
    df: pd.DataFrame,
    inchi_col: str | None,
    smiles_col: str | None,
    show_progress: bool = True,
) -> tuple[np.ndarray, dict[str, int]]:
    fps = np.zeros((len(df), TOTAL_BITS), dtype=np.uint8)
    stats = {"from_inchi": 0, "from_smiles": 0, "failed": 0}

    iterator = df.itertuples(index=False, name=None)
    if show_progress:
        iterator = tqdm(iterator, total=len(df), desc="    Computing RDKit fingerprints")

    inchi_idx = df.columns.get_loc(inchi_col) if inchi_col else None
    smiles_idx = df.columns.get_loc(smiles_col) if smiles_col else None

    for row_idx, row in enumerate(iterator):
        inchi_value = row[inchi_idx] if inchi_idx is not None else None
        smiles_value = row[smiles_idx] if smiles_idx is not None else None
        mol, source = build_mol(inchi_value, smiles_value)

        if mol is None:
            stats["failed"] += 1
            continue

        morgan_fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=MORGAN_RADIUS,
            nBits=MORGAN_BITS,
        )
        rdkit_fp = Chem.RDKFingerprint(mol, fpSize=RDKIT_TOPOLOGICAL_BITS)
        maccs_fp = MACCSkeys.GenMACCSKeys(mol)

        morgan_arr = np.zeros((MORGAN_BITS,), dtype=np.uint8)
        rdkit_arr = np.zeros((RDKIT_TOPOLOGICAL_BITS,), dtype=np.uint8)
        maccs_arr_full = np.zeros((MACCS_BITS + 1,), dtype=np.uint8)

        DataStructs.ConvertToNumpyArray(morgan_fp, morgan_arr)
        DataStructs.ConvertToNumpyArray(rdkit_fp, rdkit_arr)
        DataStructs.ConvertToNumpyArray(maccs_fp, maccs_arr_full)

        fps[row_idx] = np.concatenate([morgan_arr, rdkit_arr, maccs_arr_full[1:]])
        stats[f"from_{source}"] += 1

    return fps, stats


def build_output_dataframe(df: pd.DataFrame, fps: np.ndarray) -> pd.DataFrame:
    preferred_metadata = [
        "inchi", "InChI", "Inchi",
        "smi", "SMI", "smiles", "SMILES", "Smiles",
        "adduct", "Adduct",
        "ccs", "CCS",
    ]
    metadata_cols = [column for column in preferred_metadata if column in df.columns]
    fp_cols = [f"V{i + 1}" for i in range(TOTAL_BITS)]
    fp_df = pd.DataFrame(fps, columns=fp_cols, index=df.index, dtype=np.uint8)
    return pd.concat([df[metadata_cols].reset_index(drop=True), fp_df.reset_index(drop=True)], axis=1)


def process_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    overwrite: bool = False,
    show_progress: bool = True,
) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path is not None else default_output_for(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if output_path.exists() and not overwrite:
        print(f"Skipping existing RDKit fingerprint file: {output_path}")
        return

    print(f"\nProcessing {input_path} -> {output_path}")
    df = pd.read_csv(input_path, low_memory=False)
    inchi_col, smiles_col = find_identifier_columns(df)
    if inchi_col is None and smiles_col is None:
        raise ValueError(
            f"No InChI or SMILES column found in {input_path}. "
            "Expected one of: inchi/InChI/Inchi or smi/smiles/SMILES/Smiles."
        )

    print(f"  Rows: {len(df):,}")
    print(f"  Identifier columns: inchi={inchi_col or '-'} | smiles={smiles_col or '-'}")
    print(
        "  Fingerprint config: "
        f"Morgan(radius={MORGAN_RADIUS}, bits={MORGAN_BITS}) + "
        f"RDKitTopological(bits={RDKIT_TOPOLOGICAL_BITS}) + "
        f"MACCS(bits={MACCS_BITS}) = total {TOTAL_BITS} bits"
    )

    fps, stats = compute_fingerprint_array(df, inchi_col, smiles_col, show_progress=show_progress)
    output_df = build_output_dataframe(df, fps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    print(
        "  Done: "
        f"from InChI={stats['from_inchi']:,}, "
        f"from SMILES={stats['from_smiles']:,}, "
        f"failed={stats['failed']:,}"
    )
    print(f"  Saved: {output_path}")


def selected_inputs(names: list[str] | None) -> dict[str, Path]:
    if not names:
        return dict(DEFAULT_INPUTS)
    unknown = sorted(set(names) - set(DEFAULT_INPUTS))
    if unknown:
        raise ValueError(f"Unknown dataset(s): {unknown}. Expected one of: {sorted(DEFAULT_INPUTS)}")
    return {name: DEFAULT_INPUTS[name] for name in names}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RDKit fingerprint CSV files.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DEFAULT_INPUTS),
        help="Datasets to process. Default: hmdb ccsbase metlinccs.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Regenerate output files if they already exist.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.*")

    for input_path in selected_inputs(args.datasets).values():
        process_file(
            input_path,
            overwrite=args.overwrite,
            show_progress=not args.no_progress,
        )


if __name__ == "__main__":
    main()
