"""Generate cached physchem descriptors for CCS datasets in resources."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions


DEFAULT_DATASETS = {
    "ccsbase": {
        "input_csv": "resources/fingerprints/ccsbase_cleaned.csv",
        "output_csv": "resources/descriptors/ccsbase_physchem.csv",
    },
    "metlinccs": {
        "input_csv": "resources/fingerprints/metlinccs_cleaned.csv",
        "output_csv": "resources/descriptors/metlinccs_physchem.csv",
    },
}

OUTPUT_COLUMNS = [
    "inchi",
    "MolWt",
    "HeavyAtomCount",
    "TPSA",
    "MolLogP",
    "NumRotatableBonds",
    "RingCount",
    "FractionCSP3",
    "mol_volume_mean",
]


def generate_for_dataset(
    dataset_name: str,
    input_csv: str | Path,
    output_csv: str | Path,
    include_volume_3d: bool,
    max_isomers: int,
    save_every: int,
    force: bool,
) -> None:
    """Generate or resume descriptor cache for one CCS dataset."""

    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    source_df = _load_unique_inchi(input_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if force and output_csv.exists():
        output_csv.unlink()

    if output_csv.exists():
        out_df = pd.read_csv(output_csv, low_memory=False)
        for col in OUTPUT_COLUMNS:
            if col not in out_df.columns:
                out_df[col] = np.nan
        out_df = out_df[OUTPUT_COLUMNS]
        processed = set(out_df["inchi"].dropna().astype(str))
        print(f"{dataset_name}: resuming from cache with {len(processed):,} molecules already processed")
    else:
        out_df = pd.DataFrame(columns=OUTPUT_COLUMNS)
        processed = set()

    todo = [inchi for inchi in source_df["inchi"].astype(str).tolist() if inchi not in processed]
    print(f"{dataset_name}: processing {len(todo):,} molecules from {input_csv}")

    RDLogger.DisableLog("rdApp.*")
    new_rows = []
    for idx, inchi in enumerate(todo, start=1):
        new_rows.append(
            compute_physchem_descriptors(
                inchi,
                include_volume_3d=include_volume_3d,
                max_isomers=max_isomers,
            )
        )
        if idx % save_every == 0:
            out_df = _flush_rows(out_df, new_rows, output_csv)
            print(f"  -> {dataset_name}: saved progress {idx:,}/{len(todo):,}")
            new_rows = []

    if new_rows:
        out_df = _flush_rows(out_df, new_rows, output_csv)

    out_df.to_csv(output_csv, index=False)
    valid_2d = int(out_df["MolWt"].notna().sum())
    valid_volume = int(out_df["mol_volume_mean"].notna().sum())
    print(
        f"{dataset_name}: finished {len(out_df):,} molecules | "
        f"valid 2D={valid_2d:,} | valid volume={valid_volume:,} -> {output_csv}"
    )


def compute_physchem_descriptors(
    inchi: str,
    include_volume_3d: bool = True,
    max_isomers: int = 3,
) -> dict[str, Any]:
    """Compute the descriptor row used by CCS models."""

    mol = _mol_from_inchi(inchi)
    if mol is None:
        return _empty_descriptor_row(inchi)

    row = _empty_descriptor_row(inchi)
    try:
        row["MolWt"] = float(Descriptors.MolWt(mol))
        row["HeavyAtomCount"] = float(mol.GetNumHeavyAtoms())
        row["TPSA"] = float(rdMolDescriptors.CalcTPSA(mol))
        row["MolLogP"] = float(Descriptors.MolLogP(mol))
        row["NumRotatableBonds"] = float(rdMolDescriptors.CalcNumRotatableBonds(mol))
        row["RingCount"] = float(rdMolDescriptors.CalcNumRings(mol))
        row["FractionCSP3"] = float(rdMolDescriptors.CalcFractionCSP3(mol))
    except Exception:
        return _empty_descriptor_row(inchi)

    if include_volume_3d:
        row["mol_volume_mean"] = _compute_volume(mol, max_isomers=max_isomers)
    else:
        row["mol_volume_mean"] = np.nan

    return row


def _load_unique_inchi(input_csv: Path):
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    df = pd.read_csv(input_csv, usecols=["inchi"], low_memory=False)
    return df.dropna(subset=["inchi"]).drop_duplicates(subset="inchi").reset_index(drop=True)


def _mol_from_inchi(inchi: str):
    try:
        return Chem.MolFromInchi(str(inchi).strip())
    except Exception:
        return None


def _empty_descriptor_row(inchi: str) -> dict[str, Any]:
    return {
        "inchi": inchi,
        "MolWt": np.nan,
        "HeavyAtomCount": np.nan,
        "TPSA": np.nan,
        "MolLogP": np.nan,
        "NumRotatableBonds": np.nan,
        "RingCount": np.nan,
        "FractionCSP3": np.nan,
        "mol_volume_mean": np.nan,
    }


def _enumerate_stereoisomers(mol, max_isomers: int):
    try:
        opts = StereoEnumerationOptions(tryEmbedding=False, unique=True, maxIsomers=max_isomers)
        variants = [Chem.Mol(x) for x in EnumerateStereoisomers(mol, options=opts)]
    except Exception:
        variants = []
    if not variants:
        variants = [Chem.Mol(mol)]
    return variants[:max_isomers]


def _compute_single_volume(mol, random_seed: int) -> float | None:
    work = Chem.AddHs(Chem.Mol(mol))
    params = AllChem.ETKDGv3()
    params.randomSeed = int(random_seed)
    try:
        status = AllChem.EmbedMolecule(work, params)
        if status != 0:
            return None
        if AllChem.MMFFHasAllMoleculeParams(work):
            AllChem.MMFFOptimizeMolecule(work, maxIters=200)
        else:
            AllChem.UFFOptimizeMolecule(work, maxIters=200)
        return float(AllChem.ComputeMolVolume(work))
    except Exception:
        return None


def _compute_volume(mol, max_isomers: int) -> float:
    volumes = []
    for idx, variant in enumerate(_enumerate_stereoisomers(mol, max_isomers=max_isomers)):
        volume = _compute_single_volume(variant, random_seed=13 + idx)
        if volume is not None and np.isfinite(volume):
            volumes.append(volume)

    if not volumes:
        return np.nan
    return float(np.mean(volumes))


def _flush_rows(out_df, rows: list[dict[str, Any]], output_csv: Path):
    merged = pd.concat([out_df, pd.DataFrame(rows)], ignore_index=True)
    merged = merged.drop_duplicates(subset="inchi", keep="last")
    merged = merged[OUTPUT_COLUMNS]
    merged.to_csv(output_csv, index=False)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate cached physchem descriptors for CCS datasets.")
    parser.add_argument("--datasets", nargs="+", default=["ccsbase", "metlinccs"], choices=sorted(DEFAULT_DATASETS))
    parser.add_argument("--no-volume-3d", action="store_true", help="Generate only fast 2D descriptors.")
    parser.add_argument("--max-isomers", type=int, default=3)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--force", action="store_true", help="Overwrite existing descriptor CSVs.")
    args = parser.parse_args()

    for dataset_name in args.datasets:
        paths = DEFAULT_DATASETS[dataset_name]
        generate_for_dataset(
            dataset_name=dataset_name,
            input_csv=paths["input_csv"],
            output_csv=paths["output_csv"],
            include_volume_3d=not args.no_volume_3d,
            max_isomers=args.max_isomers,
            save_every=args.save_every,
            force=args.force,
        )


if __name__ == "__main__":
    main()
