# Data

This project uses the **Home Credit Default Risk** competition data from Kaggle:
https://www.kaggle.com/competitions/home-credit-default-risk

The data is **not stored in this repository**: competition rules do not allow
redistribution, and the files are ~2.5 GB. `python -m src.pipeline` downloads it
automatically through the Kaggle API.

## One-time setup

1. Sign in to Kaggle and **accept the competition rules**
   (https://www.kaggle.com/competitions/home-credit-default-risk/rules).
   Without this step the API answers `403 Forbidden`.
2. Go to **Settings → API → Create New Token**. A `kaggle.json` file is downloaded.
3. Move it to:
   - Windows: `C:\Users\<your-user>\.kaggle\kaggle.json`
   - Linux / macOS: `~/.kaggle/kaggle.json` (then `chmod 600 ~/.kaggle/kaggle.json`)

   Alternative: set the environment variables `KAGGLE_USERNAME` and `KAGGLE_KEY`.
4. `kaggle.json` is listed in `.gitignore`. **Never commit it.**

## Folder layout after running the pipeline

```
data/
├── raw/      # Kaggle CSVs (downloaded)
├── bronze/   # Parquet copies of the CSVs + load metadata
├── silver/   # typed and cleaned tables
└── gold/     # analytical tables: fct_bureau_monthly_status, fct_transitions,
              # dim_credit_performance, vintage_curves, client_features, test_scores
```

All of these folders are ignored by Git. Only aggregated, non-identifying
extracts are versioned, in `reports/data_studio/data/`.

## Files used

| File | Bronze table |
|---|---|
| application_train.csv | application |
| bureau.csv | bureau |
| bureau_balance.csv | bureau_balance |
| previous_application.csv | previous_application |
| installments_payments.csv | installments |
| POS_CASH_balance.csv | pos_cash |
| credit_card_balance.csv | credit_card |

`application_test.csv` has no `TARGET` and is not used. Column definitions are in
`HomeCredit_columns_description.csv`, included in the Kaggle download.
