# Anomaly Detection for CNP Transactions

Supervised Research (Law, Economics, and Data Science), 851-0763-00L HS2025
Anna Kravchenko | akravchenko@student.ethz.ch

Anomaly detection for card-not-present (CNP) fraud on the IEEE-CIS / Vesta dataset, framed as
top-K alert ranking under a fixed daily review budget rather than a leaderboard classification
task. Full motivation, literature review, and methodology are in the two proposal PDFs in
`docs/` (`Topic and Group` and `Outline`).

## Data

Place the four IEEE-CIS competition files in `Data/`:

```
Data/train_transaction.csv
Data/train_identity.csv
Data/test_transaction.csv      (present on disk, but never read by any notebook)
Data/test_identity.csv         (present on disk, but never read by any notebook)
```

Only the labeled training files are used anywhere in this project. The unlabeled competition
test files are excluded from all analyses, per the outline, and are not required to run any
notebook.

## Notebooks, in run order

| # | Notebook | Reads | Produces |
|---|---|---|---|
| 1 | `Part 1 - EDA.ipynb` | `Data/train_*.csv` | `artifacts/` (schema snapshot, missingness summary, target balance, chronological split boundaries) |
| 2 | `Part 2 - Feature Audit.ipynb` | `Data/train_*.csv`, `artifacts/split_config.json` | `artifacts/` (rare-category keep-lists, V-block groupings, audited/selected feature lists, PSI drift tables) |
| 3 | `Part 3 - Preprocessing.ipynb` | `Data/train_*.csv`, `artifacts/*` from Parts 1-2 | `processed/train_processed_gbm.parquet`, `processed/train_processed_ifae.parquet`, `processed/preprocessing_meta.json` |
| 4 | `Part 4 - Modelling.ipynb` | `processed/*` from Part 3 | `predictions/` (model comparison, Precision@K/Recall@K tables and curves, SHAP outputs, alert-budget recommendation, figures) |

Run with **Kernel → Restart & Run All**, in order 1 → 2 → 3 → 4. Each notebook reads only the
`artifacts/`/`processed/` files written by the ones before it plus the raw data -- no manual
copying between folders is required.

**Before running, set `PROJECT_DIR` at the top of each notebook** (the first code cell after the
imports) to the full local path of this project folder -- the one containing this README and
`Data/`. For example:

```python
PROJECT_DIR = Path("/Users/yourname/path/to/project")
```

Each notebook only needs this one line changed; everything else (`Data/`, `artifacts/`,
`processed/`, `predictions/`) is located relative to it automatically.

## Method summary

- **Split.** Strictly chronological `train` (60%) / `val` (20%) / `holdout` (20%) windows by
  `TransactionDT`, shared identically across all four notebooks via `artifacts/split_config.json`.
  `holdout` is the outline's "test" window, renamed to make explicit that its labels are never
  used for any fitting or selection decision.
- **Preprocessing (Part 3).** Every fitted statistic (rare-category vocab, frequency encodings,
  UID aggregates, V-block PCA, one-hot categories, imputation median, scaler mean/std) is fit on
  `train`-split rows only, then applied to `val`/`holdout`. Two feature branches are produced: a
  label-encoded one for gradient-boosted trees, and a one-hot + frequency-encoded + standardized
  one for distance/reconstruction-based models.
- **Modeling (Part 4).** Isolation Forest and a small tabular Autoencoder are the primary,
  unsupervised detectors (`isFraud` never used for fitting). A single isotonic-calibrated
  LightGBM model is included as a labeled supervised reference. Evaluation is operational:
  Precision@K/Recall@K per day, alerts-per-1000-transactions, estimated analyst-hours, PR-AUC,
  weekly temporal-stability/calibration checks, SHAP explainability, and a final alert-budget
  recommendation for two staffing scenarios.

## Environment

See `requirements.txt` for exact package versions this project was run and verified with
(Python 3.13). Install with:

```
pip install -r requirements.txt
```
