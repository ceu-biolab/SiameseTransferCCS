import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import multiprocessing as mp
import os
import queue
import signal

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions


class MoleculeTimeoutError(TimeoutError):
    pass


DESCRIPTOR_COLUMNS = [
    "inchi",
    "logp",
    "mol_volume_mean",
]


def _raise_timeout(signum, frame):
    raise MoleculeTimeoutError("Molecular volume calculation timed out")


def _load_hmdb_source(source_csv: str) -> pd.DataFrame:
    df = pd.read_csv(source_csv, usecols=["inchi"])
    df = df.dropna(subset=["inchi"]).drop_duplicates(subset="inchi").reset_index(drop=True)
    return df


def _mol_from_inchi(inchi: str):
    try:
        return Chem.MolFromInchi(inchi.strip())
    except Exception:
        return None


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


def _base_descriptor_row(inchi: str, mol):
    try:
        logp = float(Descriptors.MolLogP(mol))
    except Exception:
        logp = np.nan

    return {
        "inchi": inchi,
        "logp": logp,
        "mol_volume_mean": np.nan,
    }


def _empty_descriptor_row(inchi: str):
    return {
        "inchi": inchi,
        "logp": np.nan,
        "mol_volume_mean": np.nan,
    }


def _compute_physchem_descriptors_no_timeout(inchi: str, max_isomers: int = 3, result_queue=None):
    mol = _mol_from_inchi(inchi)
    if mol is None:
        return _empty_descriptor_row(inchi)

    base_row = _base_descriptor_row(inchi, mol)
    if result_queue is not None:
        result_queue.put(("base", base_row))

    variants = _enumerate_stereoisomers(mol, max_isomers=max_isomers)
    volumes = []
    for idx, variant in enumerate(variants):
        volume = _compute_single_volume(variant, random_seed=13 + idx)
        if volume is not None and np.isfinite(volume):
            volumes.append(volume)

    row = dict(base_row)
    if volumes:
        row["mol_volume_mean"] = float(np.mean(volumes))
    return row


def compute_physchem_descriptors(inchi: str, max_isomers: int = 3, timeout_seconds: int | None = 60):
    mol = _mol_from_inchi(inchi)
    if mol is None:
        return _empty_descriptor_row(inchi)

    base_row = _base_descriptor_row(inchi, mol)

    previous_handler = None
    timeout_active = timeout_seconds is not None and timeout_seconds > 0
    if timeout_active:
        previous_handler = signal.signal(signal.SIGALRM, _raise_timeout)
        signal.alarm(int(timeout_seconds))

    try:
        variants = _enumerate_stereoisomers(mol, max_isomers=max_isomers)
        volumes = []
        for idx, variant in enumerate(variants):
            volume = _compute_single_volume(variant, random_seed=13 + idx)
            if volume is not None and np.isfinite(volume):
                volumes.append(volume)
    except MoleculeTimeoutError:
        return base_row
    finally:
        if timeout_active:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)

    row = dict(base_row)
    if volumes:
        row["mol_volume_mean"] = float(np.mean(volumes))
    return row


def _descriptor_subprocess_entry(inchi: str, max_isomers: int, result_queue):
    try:
        row = _compute_physchem_descriptors_no_timeout(
            inchi,
            max_isomers=max_isomers,
            result_queue=result_queue,
        )
        result_queue.put(("result", row))
    except Exception as exc:
        result_queue.put(("error", repr(exc)))


def _drain_queue(result_queue):
    messages = []
    while True:
        try:
            messages.append(result_queue.get_nowait())
        except queue.Empty:
            break
        except Exception:
            break
    return messages


def _compute_with_subprocess(inchi: str, max_isomers: int = 3, timeout_seconds: int | None = 60):
    ctx_name = "fork" if "fork" in mp.get_all_start_methods() else None
    ctx = mp.get_context(ctx_name) if ctx_name else mp.get_context()
    result_queue = ctx.Queue(maxsize=4)
    process = ctx.Process(
        target=_descriptor_subprocess_entry,
        args=(inchi, max_isomers, result_queue),
    )
    process.start()
    process.join(None if timeout_seconds is None or timeout_seconds <= 0 else int(timeout_seconds))

    timed_out = process.is_alive()
    if timed_out:
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join()

    messages = _drain_queue(result_queue)
    result_queue.close()
    result_queue.join_thread()

    result = None
    base_row = None
    error = None
    for kind, payload in messages:
        if kind == "base":
            base_row = payload
        elif kind == "result":
            result = payload
        elif kind == "error":
            error = payload

    if result is not None:
        return result
    if timed_out:
        return dict(base_row) if base_row is not None else _empty_descriptor_row(inchi)
    if error is not None:
        return _empty_descriptor_row(inchi)
    return _empty_descriptor_row(inchi)


def _flush_progress(output_csv: str, out_df: pd.DataFrame, new_rows: list[dict]) -> pd.DataFrame:
    if not new_rows:
        return out_df
    new_df = pd.DataFrame(new_rows, columns=DESCRIPTOR_COLUMNS)
    if out_df.empty:
        merged = new_df
    else:
        merged = pd.concat([out_df, new_df], ignore_index=True)
    merged = merged.drop_duplicates(subset="inchi", keep="last")
    merged = merged[DESCRIPTOR_COLUMNS]
    merged.to_csv(output_csv, index=False)
    return merged


def _process_todo(todo: list[str], args, out_df: pd.DataFrame) -> pd.DataFrame:
    timeout_seconds = None if args.timeout_seconds <= 0 else args.timeout_seconds
    new_rows = []
    completed = 0

    if args.worker_mode == "inline":
        for inchi in todo:
            new_rows.append(
                compute_physchem_descriptors(
                    inchi,
                    max_isomers=args.max_isomers,
                    timeout_seconds=timeout_seconds,
                )
            )
            completed += 1
            if completed % args.save_every == 0:
                out_df = _flush_progress(args.output_csv, out_df, new_rows)
                print(f"  -> Saved progress: {completed:,}/{len(todo):,}")
                new_rows = []
        return _flush_progress(args.output_csv, out_df, new_rows)

    max_workers = max(1, int(args.num_workers))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending = set()
        todo_iter = iter(todo)

        def submit_next():
            try:
                next_inchi = next(todo_iter)
            except StopIteration:
                return False
            pending.add(
                executor.submit(
                    _compute_with_subprocess,
                    next_inchi,
                    args.max_isomers,
                    timeout_seconds,
                )
            )
            return True

        for _ in range(max_workers):
            if not submit_next():
                break

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                new_rows.append(future.result())
                completed += 1
                submit_next()

                if completed % args.save_every == 0:
                    out_df = _flush_progress(args.output_csv, out_df, new_rows)
                    print(f"  -> Saved progress: {completed:,}/{len(todo):,}")
                    new_rows = []

    return _flush_progress(args.output_csv, out_df, new_rows)


def main():
    parser = argparse.ArgumentParser(description="Generate cached HMDB physchem descriptors (logP + molecular volume).")
    parser.add_argument("--source-csv", default="resources/fingerprints/hmdb.csv")
    parser.add_argument("--output-csv", default="resources/descriptors/hmdb_physchem.csv")
    parser.add_argument("--max-isomers", type=int, default=3)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=60,
        help="Maximum seconds allowed per molecule for volume calculation. Use 0 to disable.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=6,
        help="Number of molecules to process concurrently in subprocess mode.",
    )
    parser.add_argument(
        "--worker-mode",
        choices=["subprocess", "inline"],
        default="subprocess",
        help="Use subprocess for per-molecule timeouts, or inline for serial execution.",
    )
    args = parser.parse_args()

    source_df = _load_hmdb_source(args.source_csv)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    if os.path.exists(args.output_csv):
        out_df = pd.read_csv(args.output_csv)
        for column in DESCRIPTOR_COLUMNS:
            if column not in out_df.columns:
                out_df[column] = np.nan
        out_df = out_df[DESCRIPTOR_COLUMNS]
        processed = set(out_df["inchi"].dropna().astype(str))
        print(f"Resuming from cache: {len(processed):,} molecules already processed")
    else:
        out_df = pd.DataFrame(columns=DESCRIPTOR_COLUMNS)
        processed = set()

    todo = [inchi for inchi in source_df["inchi"].astype(str).tolist() if inchi not in processed]
    print(f"Processing {len(todo):,} HMDB molecules from {args.source_csv}")
    print(
        f"Worker mode: {args.worker_mode}, num_workers={args.num_workers}, "
        f"timeout_seconds={args.timeout_seconds}"
    )

    out_df = _process_todo(todo, args, out_df)
    out_df.to_csv(args.output_csv, index=False)
    valid_volume = int(out_df["mol_volume_mean"].notna().sum())
    print(f"Finished. Cached {len(out_df):,} molecules, {valid_volume:,} with valid molecular volume.")


if __name__ == "__main__":
    main()
