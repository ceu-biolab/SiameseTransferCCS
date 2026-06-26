# METLIN Enchanter

Code for pretraining Siamese models on molecular fingerprints and using them for CCS prediction. The main workflow uses RDKit fingerprints; AlvaDesc is kept as an explicit alternative.

## Structure

```text
configs/
  ccs_prediction_heads.yaml                 # CCS heads and training settings
  pretrain_siamese.yaml                     # RDKit pretraining
  pretrain_siamese_alvadesc.yaml            # AlvaDesc pretraining
  pretrain_siamese_deep_only*.yaml          # architecture ablations
  pretrain_siamese_wide_only*.yaml

resources/
  classifications/
    hmdb_classifications.tsv                # original input
  fingerprints/
    hmdb.csv                                # original AlvaDesc input
    ccsbase.csv                             # original AlvaDesc input
    metlinccs.csv                           # original AlvaDesc input
    *_cleaned.csv                           # generated cleaned datasets
    *_rdkit.csv                             # generated RDKit fingerprints
  descriptors/
    *_physchem.csv                          # generated physicochemical descriptors

src/data/
  load_*.py                                 # in-memory loading and cleaning
  generate_clean_fingerprint_csvs.py        # materializes cleaned CCSBase/METLIN files
  generate_*_physchem_descriptors.py        # physicochemical descriptors
  generate_rdkit_fingerprints.py            # RDKit fingerprints

src/models/
  pretrain_siamese.py                       # RDKit Siamese model
  pretrain_siamese_alvadesc.py              # AlvaDesc Siamese model
  run_baseline.py                           # direct CCS models on fingerprints
  run_siamese.py                            # CCS models using a Siamese encoder
  ablations/                                # deep-only, wide-only, and task ablations

results/                                    # current outputs
main.py                                     # full RDKit pipeline
```

This working tree may also contain `resources_old/` and `results_old/`. They are snapshots from previous runs used for reproducibility checks, not part of the normal workflow.

## Environment

The Conda/Micromamba environment is defined in `environment.yml`.

```bash
conda activate 'pytorch+_env'
```

or:

```bash
micromamba activate 'pytorch+_env'
```

To create it from scratch:

```bash
micromamba env create -f environment.yml
```

The scripts set `KERAS_BACKEND=torch` when the variable is not already defined.

## Data

The project assumes these files are the original inputs:

```text
resources/fingerprints/hmdb.csv
resources/classifications/hmdb_classifications.tsv
resources/fingerprints/ccsbase.csv
resources/fingerprints/metlinccs.csv
```

All other files under `resources/fingerprints` and `resources/descriptors` can be regenerated.

## Data Preparation

Starting from only the four original input files, run:

```bash
python -m src.data.generate_clean_fingerprint_csvs \
  --ccsbase-descriptors never \
  --overwrite
```

```bash
python -m src.data.generate_hmdb_physchem_descriptors \
  --num-workers 6
```

```bash
python -m src.data.generate_ccs_physchem_descriptors
```

After `ccsbase_physchem.csv` exists, materialize the cleaned CCSBase file again with descriptors:

```bash
python -m src.data.generate_clean_fingerprint_csvs \
  --datasets ccsbase \
  --ccsbase-descriptors always \
  --overwrite
```

Then generate RDKit fingerprints:

```bash
python -m src.data.generate_rdkit_fingerprints --overwrite
```

Notes:

- `generate_hmdb_physchem_descriptors` supports `--num-workers`.
- `generate_ccs_physchem_descriptors` resumes from existing descriptor CSVs; use `--force` only when rebuilding from scratch.
- Cleaned CSV generation also writes `*_summary.json`.

## Main Pipeline

The full RDKit pipeline is:

```bash
python main.py
```

It runs, in order:

1. HMDB RDKit Siamese pretraining.
2. Direct CCS baselines with RDKit fingerprints.
3. CCS training with the pretrained Siamese encoder.

## Siamese Pretraining

RDKit:

```bash
python -m src.models.pretrain_siamese \
  --config configs/pretrain_siamese.yaml
```

AlvaDesc:

```bash
python -m src.models.pretrain_siamese_alvadesc \
  --config configs/pretrain_siamese_alvadesc.yaml
```

Default outputs:

```text
results/Siamese_physchem
results/Siamese_physchem_alvadesc
```

The training loss is MAE. The Siamese model keeps the Tanimoto similarity, LogP, and molecular volume tasks when they are enabled in the YAML file.

## Direct CCS Models

Baselines with fingerprints, adduct, and auxiliary features:

```bash
python -m src.models.run_baseline \
  --config configs/ccs_prediction_heads.yaml \
  --fingerprint-source rdkit \
  --folds 5
```

For AlvaDesc:

```bash
python -m src.models.run_baseline \
  --config configs/ccs_prediction_heads.yaml \
  --fingerprint-source alvadesc \
  --folds 5
```

Each run covers:

- CCSBase -> CCSBase
- CCSBase -> METLINCCS
- METLINCCS -> CCSBase
- METLINCCS -> METLINCCS

Models:

- `linear_regression_fingerprints_only`
- `linear_regression`
- `gated_residual_mlp`

## CCS With Siamese Encoder

RDKit:

```bash
python -m src.models.run_siamese \
  --config configs/ccs_prediction_heads.yaml \
  --fingerprint-source rdkit \
  --folds 5 \
  --siamese-results-dir results/Siamese_physchem
```

AlvaDesc:

```bash
python -m src.models.run_siamese \
  --config configs/ccs_prediction_heads.yaml \
  --fingerprint-source alvadesc \
  --folds 5 \
  --siamese-results-dir results/Siamese_physchem_alvadesc
```

Models:

- `linear_regression`: frozen encoder.
- `linear_regression_ft`: fine-tuned encoder.
- `gated_residual_mlp`: fine-tuned encoder with auxiliary features.

## Ablations

Ablation code lives in `src/models/ablations`.

Deep-only or wide-only pretraining:

```bash
python -m src.models.ablations.pretrain_siamese_deep_only \
  --config configs/pretrain_siamese_deep_only.yaml
```

```bash
python -m src.models.ablations.pretrain_siamese_wide_only \
  --config configs/pretrain_siamese_wide_only.yaml
```

Their CCS runners are:

```bash
python -m src.models.ablations.run_siamese_deep_only
python -m src.models.ablations.run_siamese_wide_only
```

For pretraining task sweeps:

```bash
python -m src.models.ablations.run_siamese_ablations \
  --sources rdkit alvadesc \
  --folds 5
```

## CCS Configuration

`configs/ccs_prediction_heads.yaml` controls:

- `mass.enabled`: molecular mass as an auxiliary feature.
- `molecular_features.enabled`: cached physicochemical descriptors.
- `training.*`: batch size, learning rates, epochs, and callbacks.

Descriptor imputation uses the training median within each fold. Scaling is also fitted per fold to avoid data leakage.

## Results

Outputs are written to:

```text
results/Siamese_physchem*/          # Siamese pretraining
results/CCSTrainer/                 # CCS models, metrics, splits, and metadata
results/ablations/                  # ablation sweeps
```

The `*_results.csv` files contain a `Total` row with fold mean and standard deviation. Weights, scalers, encoders, histories, and plots are stored inside each fold directory.
