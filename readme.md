# SiameseTransferCCS

Code for pretraining Siamese models on molecular fingerprints and using them for CCS prediction. The main workflow uses RDKit fingerprints; AlvaDesc is kept as an explicit alternative.

## Structure

```text
configs/
  ccs_prediction_heads.yaml                 # CCS heads and training settings
  pretrain_siamese.yaml                     # RDKit pretraining
  pretrain_siamese_alvadesc.yaml            # AlvaDesc pretraining
  pretrain_siamese_deep_only.yaml           # RDKit architecture ablations
  pretrain_siamese_wide_only.yaml

resources/
  classifications/
    hmdb_classifications.zip                # original input
  descriptors/
    physchem_descriptors.zip                # descriptor caches used in the article
  fingerprints/
    hmdb.zip                                # original AlvaDesc input
    ccsbase.zip                             # original AlvaDesc input
    metlinccs.zip                           # original AlvaDesc input
    *_cleaned.zip                           # generated cleaned datasets
    *_rdkit.zip                             # generated RDKit fingerprints

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
The runtime versions in `requirements.txt` match those in `environment.yml`
and the environment used to reproduce the article results.

## Data

To reproduce the article results, we recommend using the prepared datasets,
fingerprints, and physicochemical descriptor caches distributed as ZIP archives
in this repository and in the associated
[Zenodo deposit](https://zenodo.org/uploads/23163180). Extract the archives as
described below before running the training pipeline. Alternatively, the prepared
files can be regenerated following the instructions in [Data Preparation](#data-preparation).

The data preparation scripts use the following source files:

```text
resources/fingerprints/hmdb.csv
resources/classifications/hmdb_classifications.tsv
resources/fingerprints/ccsbase.csv
resources/fingerprints/metlinccs.csv
```
From the project root, extract the bundled classification and fingerprint
archives into their respective directories:

```bash
for archive in resources/classifications/*.zip resources/fingerprints/*.zip; do
  unzip -n "$archive" -d "$(dirname "$archive")"
done
```

The archive `resources/descriptors/physchem_descriptors.zip` contains the exact
descriptor caches used in the article: `hmdb_physchem.csv`,
`ccsbase_physchem.csv`, and `metlinccs_physchem.csv`. Extract it into
`resources/descriptors` before pretraining or CCS model training:

```bash
unzip -n resources/descriptors/physchem_descriptors.zip -d resources/descriptors
```

These extraction commands preserve any existing files.

## Data Preparation

This optional workflow regenerates the prepared data from the four source files
listed above. It is not required when using the prepared files distributed with
the project. Note that some data preparation steps, especially molecular-volume
calculation, are computationally intensive and can take up to several days,
depending on the available hardware. We recommend using the prepared files
to reproduce the article results.

To regenerate the prepared data, run:

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

Pretraining uses the HMDB molecule set after a common filter that excludes
molecules without LogP, including when the LogP task is disabled. Within each
fingerprint source, every objective ablation therefore uses the same molecule
set and reproducible 90/10 training/validation partition, stratified by molecular classification. The
persisted validation partition is used only for early stopping, learning-rate
scheduling, and checkpoint selection; it is not a test partition. The default
configuration draws 100,000 training pairs and 10,000 validation pairs per
epoch.

The training loss is MAE. The Siamese model keeps the Tanimoto similarity,
LogP, and molecular volume tasks when they are enabled in the YAML file.

## Baseline Models

Run the baselines with fingerprints, adduct, and auxiliary features:

```bash
python -m src.models.run_baseline \
  --config configs/ccs_prediction_heads.yaml \
  --fingerprint-source rdkit \
  --folds 5
```

For AlvaDesc fingerprints execute:

```bash
python -m src.models.run_baseline \
  --config configs/ccs_prediction_heads.yaml \
  --fingerprint-source alvadesc \
  --folds 5
```

Each run covers four evaluation scenarios: two within-database settings, where
training and testing use separate subsets of the same database, and two
cross-database settings, where the model is trained on one database and tested
on the other. The arrows indicate the training source and test destination:

- CCSBase -> CCSBase
- CCSBase -> METLINCCS
- METLINCCS -> CCSBase
- METLINCCS -> METLINCCS

Each run evaluates three baseline models using molecular fingerprints and adduct
information. They differ in predictor complexity and the inclusion of
physicochemical descriptors:

- `linear_regression_fingerprints_only`: linear regression without physicochemical descriptors.
- `linear_regression`: linear regression augmented with physicochemical descriptors.
- `gated_residual_mlp`: a descriptor-gated residual multilayer perceptron (DGR-MLP).

For details, see “Baseline and ablation experiments” in the Materials and Methods
of the accompanying article, *Siamese Molecular Pretraining Improves Collision
Cross Section Prediction Across Experimental Databases*.

## CCS Prediction Using a Pretrained Siamese Encoder

These commands use a pretrained Siamese encoder to transform molecular
fingerprints into embeddings for CCS prediction. Each command evaluates all
three prediction strategies in the four within- and cross-database scenarios
described above, using five folds. Run the corresponding pretraining command
first, or provide an existing compatible pretraining results directory.

For RDKit fingerprints, load the encoder from `results/Siamese_physchem`:
```bash
python -m src.models.run_siamese \
  --config configs/ccs_prediction_heads.yaml \
  --fingerprint-source rdkit \
  --folds 5 \
  --siamese-results-dir results/Siamese_physchem
```

For AlvaDesc fingerprints, use the corresponding encoder from
`results/Siamese_physchem_alvadesc`. This runs the same evaluation with AlvaDesc
fingerprints and their matching pretrained representation:
```bash
python -m src.models.run_siamese \
  --config configs/ccs_prediction_heads.yaml \
  --fingerprint-source alvadesc \
  --folds 5 \
  --siamese-results-dir results/Siamese_physchem_alvadesc
```

The three strategies differ in whether the encoder is updated during CCS
training and in the regression head used:

- `linear_regression`: a linear predictor on frozen embeddings and adduct information, without physicochemical descriptors.
- `linear_regression_ft`: a linear predictor with physicochemical descriptors and a fine-tuned encoder.
- `gated_residual_mlp`: a descriptor-gated residual MLP with a fine-tuned encoder.

Use `--models` to select a subset of these strategies. For methodological details,
see the Materials and Methods of the accompanying article.

## Ablations

Ablation code is in `src/models/ablations`.

The ablations use RDKit fingerprints and the DGR-MLP CCS head.
The complete suite runs sequentially and contains random initialization, six
pretraining-objective ablations, wide-only, and deep-only. The all-task
Wide+Deep reference is validated from the existing standard CCS results and is
not trained again.

First, the reference CCS results must exist for the four routes under the tag
`hmdb90_10_v1`. They can be generated with:

```bash
python -m src.models.run_siamese \
  --config configs/ccs_prediction_heads.yaml \
  --fingerprint-source rdkit \
  --folds 5 \
  --random-seed 42 \
  --siamese-results-dir results/Siamese_physchem \
  --experiment-tag hmdb90_10_v1 \
  --models gated_residual_mlp
```

Then run all nine ablations in one invocation:

```bash
python -m src.models.ablations.run_siamese_ablations \
  --fingerprint-source rdkit \
  --folds 5 \
  --random-seed 42 \
  --experiment-tag dgr_hmdb_ablations
```

Use `--dry-run` to validate the reference and
print the eight pretraining commands and nine downstream commands without
creating files.

After a complete run, you can generate the figure and statistical tables from
the experiment manifest:

```bash
python -m src.analysis.generate_dgr_ablation_figures \
  --manifest results/ablations/dgr_hmdb/dgr_hmdb_ablations/experiment_manifest.json
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

`MSPE(%)` is `100 * mean(((prediction - target) / target) ** 2)`: a 10% relative
error contributes 1.0 to this metric. Older outputs stored the unconverted ratio
under the same column name; existing result files are not automatically migrated.



## License

The source code of this project is distributed under the GNU General Public
License version 3 only (`GPL-3.0-only`). See [LICENSE](LICENSE) for details.

