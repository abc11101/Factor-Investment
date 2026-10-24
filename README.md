# Factor Investment

Learn weights for the 24 factors in `features.py` using **Ridge and Lasso regression**.
The models predict each stock's next-month total return. Ridge shrinks the learned
weights; Lasso can set weights to zero, providing a feature-selection report.
Neither model is forced to follow the economic signs used by the existing
equal-weight composite.

## Run in VS Code

Open this repository folder, then select the `.venv` Python interpreter. The local
environment is already installed on this laptop. In VS Code's terminal:

```bash
source .venv/bin/activate
python linear_models.py --demo --output outputs/demo
```

The demo uses **synthetic data with a deliberately planted relationship**. It checks
the implementation; its selected features and prediction scores are not empirical
investment findings. No real monthly market/fundamental dataset is bundled.

On a different machine, create the environment first (Python 3.10 or newer):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run real data with:

```bash
python linear_models.py --input data/monthly_panel.csv --output outputs/linear
```

For a rolling rather than expanding training history, add
`--train-window-months 60`. The default minimum is 36 labeled feature months;
inner validation uses three consecutive six-month blocks. For a shorter research
sample, explicitly choose smaller values, for example
`--min-train-months 24 --cv-splits 3 --validation-months 4`.

## Input data

Supply one row per stock per calendar month, using decimal total returns
(`0.02` means 2%). Dates are normalized to calendar month end. Each ticker's
observed history must be consecutive because the existing momentum and annual
growth calculations use row lags. Different ticker histories may start/end at
different months. Missing interior months are rejected rather than interpreted
as zero returns. Duplicate stock/month observations are rejected too.

Required columns match the input contract in `features.py`:

| Group | Columns |
| --- | --- |
| Identifiers | `date`, `ticker` |
| Prices and trading | `ret`, `prc`, `shs_out`, `vol_sh`, `mkt_ret` |
| Balance sheet | `at`, `equity`, `debt`, `cash` |
| Income statement | `rev_ttm`, `gp_ttm`, `ebit_ttm`, `da_ttm`, `ni_ttm` |
| Cash flow | `cfo_ttm`, `capex_ttm`, `d_nwc_ttm` |
| Payouts | `div_ttm`, `buyback_ttm`, `issuance_ttm` |
| Optional | `sector`, `pref`, `minority`, `tax_rate` |

Use a consistent currency and units across prices, shares, and fundamentals.
`mkt_ret` must be the same for every stock on a date. Returns must be finite and
at least -100%; other missing factor inputs are handled during preprocessing.
Fundamentals and sector labels must reflect what was **actually available at that
date**. The model cannot reconstruct filing dates from accounting period dates.
Include delisted stocks and their final total returns where available; a missing
next-month return is left unknown and excluded from scoring rather than set to zero.

The default uses `standardize()` from `features.py`: monthly 1st/99th percentile
winsorization, cross-sectional z-scores, and sector demeaning when `sector` is
present. Pass `--no-sector-neutral` to omit sector demeaning. These operations use
only the contemporaneous universe. Stocks need at least 12 observed factors by
default (`--min-features` changes this). Preserve the full pre-evaluation history
so rolling factors can warm up.

## Training and time order

At the close of month **t**, features from t predict the return in **t+1**.
Targets are matched by stock and exact next calendar month, rather than by an
unrestricted next-row shift. Forecasts assume execution after the features become
available; implement execution delays and costs when building portfolios later.

For each forecast date:

1. Keep historical feature rows whose next-month return is already observed by t.
2. Tune each model's alpha on chronological validation blocks, keeping all stocks
   for a month together. Each fold's training labels are known by its validation
   start date. No random stock-row splits are used.
3. Fit median imputation and scaling separately within each training fold, inside
   a scikit-learn pipeline. Entirely missing training features are retained with
   a zero imputation, preserving all 24 coefficient names.
4. Refit the chosen model on the available history and predict stocks at t.
5. Advance a month and repeat. The final month can be forecast even though its
   realized return is not available yet.

The alpha grids differ because scikit-learn's Ridge and Lasso normalize their
objectives differently. They are configurable through `ModelConfig` in Python.
`alpha_at_grid_edge` flags a choice at an endpoint; extend the grid using training
validation if this happens frequently. A candidate that fails or does not converge
in any validation fold is excluded, with its status recorded in `cv_results.csv`.
The lowest-MSE candidate whose final refit converges is selected. If no candidate
works, the run stops rather than reporting unreliable coefficients. Default tuning minimizes validation
MSE; it does not optimize portfolio Sharpe or the number of selected features.

## Outputs and interpretation

| Output | Use |
| --- | --- |
| `predictions.csv` | Monthly stock forecasts, realized next-month targets where known, and baseline values |
| `coefficients.csv` | All 24 learned coefficients for both models at each forecast date, selection flags, and training coverage |
| `lasso_selection.csv` | Feature selection frequency, mean signed/absolute coefficients, and latest selected status |
| `fit_summary.csv` | Chosen alpha, training size, cutoff, intercept, and number of nonzero weights |
| `cv_results.csv` | Every candidate alpha and its validation MSE |
| `validation_splits.csv` | Training and validation cutoffs for auditing time order |
| `monthly_metrics.csv` | Monthly MSE and Spearman rank information coefficient (IC) |
| `metrics.csv` | Equally weighted monthly MSE, RMSE, mean IC, and MSE skill against the historical mean |
| `latest_models.joblib` | Latest fitted preprocessing pipelines for Ridge/Lasso, feature order, cutoff, and configuration |
| `run_metadata.json` | Configuration, feature names, library versions, and whether the run was synthetic |

`coefficient` multiplies the imputed, cross-sectionally standardized factor.
Together with the reported intercept, it reconstructs the prediction. The
`standardized_coefficient` instead describes a one-training-standard-deviation
change after the pipeline's scaler. Coefficients are return sensitivities, **not
portfolio allocations**, and need not sum to one.

Positive coefficients predict higher returns as a factor increases; negative
coefficients predict lower returns, conditional on the other factors. Inspect
Lasso's selection frequency across months as well as its latest selection.
Correlated features can substitute for one another, so a nonzero coefficient
does not establish causality or statistical significance, and a zero does not
prove that a factor is economically irrelevant.

The historical-mean return prediction is the MSE baseline. The existing signed
equal-weight composite is compared through rank IC only because its score is not
in return units. Latest unknown targets are forecast but excluded from evaluation.
Portfolio construction, trading costs, factor attribution, and performance
significance are subsequent project steps. Preserve a final untouched time period
before repeatedly revising the research design using these results.

Only load model files you created or trust. To reuse the saved pipelines, first
build/standardize new raw observations with sufficient history using the same
`prepare_panel()` configuration, then call `bundle['models']['ridge'].predict()`
on the 24 feature columns in `bundle['features']` order.

## Checks

```bash
python -m unittest discover -s tests -v
```

The checks cover target alignment, chronological folds, future-data isolation,
missing monthly observations, and Lasso recovering a known sparse synthetic signal.

Implementation references: [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html),
[Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html),
and [preprocessing leakage and pipelines](https://scikit-learn.org/stable/common_pitfalls.html).
