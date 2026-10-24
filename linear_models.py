"""Learn weights for features.FACTORS using monthly walk-forward Ridge and Lasso.

Run `python linear_models.py --demo` or see README.md for the real-data contract.
No economic sign constraints are imposed on either model.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import FACTORS, build_factors, composite_score, standardize

FEATURES = list(FACTORS)
REQUIRED = {
    "date", "ticker", "ret", "prc", "shs_out", "vol_sh", "mkt_ret",
    "at", "equity", "debt", "cash", "rev_ttm", "gp_ttm", "ebit_ttm",
    "da_ttm", "ni_ttm", "cfo_ttm", "capex_ttm", "d_nwc_ttm",
    "div_ttm", "buyback_ttm", "issuance_ttm",
}


@dataclass(frozen=True)
class ModelConfig:
    min_train_months: int = 36
    cv_splits: int = 3
    validation_months: int = 6
    train_window_months: int | None = None
    min_features: int = 12
    sector_neutral: bool = True
    ridge_alphas: tuple[float, ...] = tuple(np.logspace(-2, 6, 9))
    lasso_alphas: tuple[float, ...] = tuple(np.logspace(-5, -1, 9))

    def validate(self):
        if self.cv_splits < 2 or self.validation_months < 1:
            raise ValueError("Use at least 2 CV splits and 1 month per validation block.")
        if self.min_train_months <= self.cv_splits * self.validation_months:
            raise ValueError("min_train_months must exceed cv_splits * validation_months.")
        if self.train_window_months is not None and self.train_window_months < self.min_train_months:
            raise ValueError("The rolling window must be at least min_train_months.")
        if not 1 <= self.min_features <= len(FEATURES):
            raise ValueError(f"min_features must be between 1 and {len(FEATURES)}.")
        for alphas in (self.ridge_alphas, self.lasso_alphas):
            if not alphas or not np.isfinite(alphas).all() or min(alphas) <= 0:
                raise ValueError("Alpha grids must contain positive finite values.")


def validate_input(raw: pd.DataFrame) -> pd.DataFrame:
    """Require consecutive monthly observations because existing factors use row lags."""
    missing = REQUIRED.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing input columns: {', '.join(sorted(missing))}")
    raw = raw.copy()
    if raw.empty or raw[["date", "ticker"]].isna().any().any():
        raise ValueError("Supply nonempty data with nonmissing dates and tickers.")
    raw["date"] = pd.to_datetime(raw["date"], errors="raise").dt.to_period("M").dt.to_timestamp("M")
    raw["ticker"] = raw["ticker"].astype(str)
    if raw.duplicated(["date", "ticker"]).any():
        raise ValueError("Input must contain one row per ticker and calendar month.")
    numeric = REQUIRED - {"date", "ticker"}
    for col in numeric | ({"tax_rate", "pref", "minority"} & set(raw.columns)):
        raw[col] = pd.to_numeric(raw[col], errors="raise")
    if not np.isfinite(raw["ret"]).all() or (raw["ret"] < -1).any():
        raise ValueError("ret must contain finite decimal total returns, at least -1.")
    if "sector" in raw and raw["sector"].isna().any():
        raise ValueError("Fill missing sector labels or omit the sector column.")
    raw = raw.sort_values(["ticker", "date"]).reset_index(drop=True)
    month_number = raw["date"].dt.year * 12 + raw["date"].dt.month
    gaps = month_number.groupby(raw["ticker"]).diff().dropna()
    if (gaps != 1).any():
        raise ValueError("Gaps within a ticker's monthly history invalidate row-based factor lags. "
                         "Repair the input calendar upstream; do not invent zero returns.")
    if raw.groupby("date")["mkt_ret"].nunique(dropna=False).gt(1).any():
        raise ValueError("mkt_ret must be the same for all stocks within a month.")
    return raw


def prepare_panel(raw: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    """Use information at month t to predict the total return during month t+1.

    Cross-sectional normalization uses only stocks observed in the same month.
    Historical labels are matched by calendar month, never by a blind next-row shift.
    Upstream fundamentals must already be lagged to their actual availability dates.
    """
    config.validate()
    raw = validate_input(raw)
    panel = standardize(build_factors(raw), sector_neutral=config.sector_neutral)
    panel["feature_count"] = panel[FEATURES].notna().sum(axis=1)
    panel["equal_weight_score"] = composite_score(panel)
    panel["target_date"] = panel["date"] + pd.offsets.MonthEnd(1)
    labels = raw[["date", "ticker", "ret"]].rename(columns={"date": "target_date", "ret": "target"})
    panel = panel.merge(labels, on=["target_date", "ticker"], how="left", validate="one_to_one")
    panel = panel.loc[panel["feature_count"] >= config.min_features]
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


def monthly_cv_splits(train: pd.DataFrame, config: ModelConfig):
    """Expanding folds grouped by month, with every training label known at cutoff."""
    months = np.sort(train["date"].unique())
    start = len(months) - config.cv_splits * config.validation_months
    if start < 1:
        raise ValueError("Insufficient training months for chronological validation.")
    splits = []
    for offset in range(start, len(months), config.validation_months):
        val_months = months[offset:offset + config.validation_months]
        cutoff = pd.Timestamp(val_months[0])
        fit = np.flatnonzero(((train["date"] < cutoff) & (train["target_date"] <= cutoff)).to_numpy())
        val = np.flatnonzero(train["date"].isin(val_months).to_numpy())
        if not len(fit) or not len(val):
            raise ValueError("An inner validation fold has no training or validation observations.")
        splits.append((fit, val))
    return splits


def model_pipeline(name: str) -> Pipeline:
    regressor = Ridge(solver="svd") if name == "ridge" else Lasso(
        max_iter=50_000, tol=1e-4, selection="random", random_state=42, precompute=True)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("regressor", regressor),
    ])


def tune_model(name, alphas, train, splits):
    """Reject unconverged candidates; publish their status rather than their weights."""
    search = GridSearchCV(model_pipeline(name), {"regressor__alpha": list(alphas)},
                          cv=splits, scoring="neg_mean_squared_error", refit=False,
                          n_jobs=1, error_score=np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        # Failures remain explicit in the returned audit table.
        warnings.simplefilter("ignore", FitFailedWarning)
        warnings.filterwarnings("ignore", message="One or more of the test scores are non-finite")
        search.fit(train[FEATURES], train["target"])
    candidates = []
    for params, score in zip(search.cv_results_["params"], search.cv_results_["mean_test_score"]):
        candidates.append({"alpha": float(params["regressor__alpha"]), "cv_mse": -float(score),
                           "status": "valid" if np.isfinite(score) else "cv_fit_failed"})
    order = sorted((i for i, row in enumerate(candidates) if row["status"] == "valid"),
                   key=lambda i: candidates[i]["cv_mse"])
    for rank, index in enumerate(order, 1):
        candidates[index]["rank"] = rank
    for index in order:
        chosen = candidates[index]
        model = model_pipeline(name).set_params(regressor__alpha=chosen["alpha"])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(train[FEATURES], train["target"])
        except ConvergenceWarning:
            chosen["status"] = "refit_not_converged"
            continue
        chosen["status"] = "selected"
        return model, chosen, candidates
    raise ValueError(f"No converged {name} candidate. Increase regularization or inspect redundant factors.")


def rank_ic(actual: pd.Series, predicted: pd.Series) -> float:
    valid = actual.notna() & predicted.notna()
    a, p = actual[valid], predicted[valid]
    if len(a) < 3 or a.nunique() < 2 or p.nunique() < 2:
        return float("nan")
    return float(a.rank().corr(p.rank()))


def prediction_metrics(predictions: pd.DataFrame):
    records = []
    for (date, model), group in predictions.groupby(["date", "model"], sort=True):
        actual = group["target"]
        valid = actual.notna()
        if not valid.any():
            continue
        for name, values in [(model, group["prediction"])]:
            records.append({"date": date, "model": name, "n_stocks": int(valid.sum()),
                            "mse": float(((actual[valid] - values[valid]) ** 2).mean()),
                            "rank_ic": rank_ic(actual, values)})
        # Baselines are identical for both fitted models; add them once per month.
        if model == "ridge":
            records.append({"date": date, "model": "historical_mean", "n_stocks": int(valid.sum()),
                            "mse": float(((actual[valid] - group.loc[valid, "historical_mean"]) ** 2).mean()),
                            "rank_ic": np.nan})
            records.append({"date": date, "model": "equal_weight", "n_stocks": int(valid.sum()),
                            "mse": np.nan, "rank_ic": rank_ic(actual, group["equal_weight_score"])})
    monthly = pd.DataFrame(records, columns=["date", "model", "n_stocks", "mse", "rank_ic"])
    summary = []
    mean_baseline = monthly.loc[monthly["model"] == "historical_mean", "mse"].mean()
    for name, group in monthly.groupby("model"):
        mse = group["mse"].mean()  # equal weight to each evaluation month
        summary.append({"model": name, "months": len(group), "mean_monthly_mse": mse,
                        "rmse": np.sqrt(mse), "mean_rank_ic": group["rank_ic"].mean(),
                        "mse_skill_vs_historical_mean": 1 - mse / mean_baseline if mean_baseline > 0 else np.nan})
    return monthly, pd.DataFrame(summary)


def selection_summary(coefficients: pd.DataFrame) -> pd.DataFrame:
    lasso = coefficients.loc[coefficients["model"] == "lasso"].sort_values("date")
    summary = lasso.groupby("feature").agg(
        fitted_months=("date", "size"), selection_frequency=("selected", "mean"),
        mean_coefficient=("coefficient", "mean"),
        mean_abs_coefficient=("coefficient", lambda x: x.abs().mean()),
        latest_coefficient=("coefficient", "last"), latest_selected=("selected", "last"),
    ).reset_index()
    return summary.sort_values(["selection_frequency", "mean_abs_coefficient"], ascending=False)


def walk_forward(panel: pd.DataFrame, config: ModelConfig) -> dict:
    """Retune and refit both models at each eligible month-end, including live forecasts."""
    config.validate()
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    predictions, coefficients, fits, tuning, fold_records = [], [], [], [], []
    latest_models = {}
    for date in sorted(panel["date"].unique()):
        date = pd.Timestamp(date)
        train = panel.loc[(panel["date"] < date) & (panel["target_date"] <= date) & panel["target"].notna()]
        if config.train_window_months is not None:
            train = train.loc[train["date"] >= date - pd.DateOffset(months=config.train_window_months)]
        if train["date"].nunique() < config.min_train_months:
            continue
        train = train.reset_index(drop=True)
        current = panel.loc[panel["date"] == date]
        splits = monthly_cv_splits(train, config)
        for fold, (fit_idx, val_idx) in enumerate(splits, 1):
            fold_records.append({"prediction_date": date, "fold": fold,
                                 "train_feature_end": train.iloc[fit_idx]["date"].max(),
                                 "train_label_end": train.iloc[fit_idx]["target_date"].max(),
                                 "validation_start": train.iloc[val_idx]["date"].min(),
                                 "validation_end": train.iloc[val_idx]["date"].max(),
                                 "validation_label_end": train.iloc[val_idx]["target_date"].max()})
        for name, alphas in [("ridge", config.ridge_alphas), ("lasso", config.lasso_alphas)]:
            model, chosen, candidates = tune_model(name, alphas, train, splits)
            latest_models[name] = model
            forecast = current[["date", "target_date", "ticker", "target", "equal_weight_score"]].copy()
            forecast["model"] = name
            forecast["prediction"] = model.predict(current[FEATURES])
            forecast["historical_mean"] = train["target"].mean()
            predictions.append(forecast)
            scaler = model.named_steps["scaler"]
            regressor = model.named_steps["regressor"]
            coef = regressor.coef_ / scaler.scale_
            intercept = float(regressor.intercept_ - np.dot(coef, scaler.mean_))
            for feature, weight, scaled in zip(FEATURES, coef, regressor.coef_):
                coefficients.append({"date": date, "model": name, "feature": feature,
                                     "coefficient": float(weight), "standardized_coefficient": float(scaled),
                                     "selected": bool(abs(scaled) > 1e-10),
                                     "train_nonmissing_fraction": float(train[feature].notna().mean())})
            best_alpha = chosen["alpha"]
            fits.append({"date": date, "model": name, "alpha": best_alpha,
                         "alpha_at_grid_edge": best_alpha in (min(alphas), max(alphas)),
                         "train_months": train["date"].nunique(), "train_rows": len(train),
                         "train_label_end": train["target_date"].max(), "intercept": intercept,
                         "nonzero_coefficients": int((abs(regressor.coef_) > 1e-10).sum()),
                         "best_cv_mse": chosen["cv_mse"],
                         "failed_candidates": sum(row["status"] in ("cv_fit_failed", "refit_not_converged")
                                                  for row in candidates)})
            for row in candidates:
                tuning.append({"date": date, "model": name, **row})
    if not predictions:
        raise ValueError("No eligible forecast months. Supply more history or reduce min_train_months "
                         "and the validation block sizes; check factor coverage too.")
    forecasts = pd.concat(predictions, ignore_index=True)
    weights = pd.DataFrame(coefficients)
    monthly, metrics = prediction_metrics(forecasts)
    return {"predictions": forecasts, "coefficients": weights, "fit_summary": pd.DataFrame(fits),
            "cv_results": pd.DataFrame(tuning), "validation_splits": pd.DataFrame(fold_records),
            "lasso_selection": selection_summary(weights), "monthly_metrics": monthly,
            "metrics": metrics, "models": latest_models}


def synthetic_panel(months=78, stocks=60, seed=42) -> pd.DataFrame:
    """Artificial raw monthly data for plumbing checks, never investment evidence."""
    rng = np.random.default_rng(seed)
    sectors = np.array([f"S{i % 5}" for i in range(stocks)])
    base_price = rng.uniform(20, 100, stocks)
    shares = rng.uniform(1e7, 1e8, stocks)
    size = base_price * shares
    equity = size * rng.uniform(.2, .8, stocks)
    previous_return = np.zeros(stocks)
    records = []
    for date in pd.date_range("2015-01-31", periods=months, freq="ME"):
        market = rng.normal(.005, .03)
        # A planted price-to-book signal plus reversal; demonstration only.
        value = np.log(equity / (base_price * shares))
        value = (value - value.mean()) / value.std()
        returns = np.clip(market + .012 * value - .15 * previous_return + rng.normal(0, .04, stocks), -.8, .8)
        base_price = base_price * (1 + returns)
        revenue = size * (1 + .002 * len(records)) * rng.lognormal(0, .08, stocks)
        income = revenue * rng.uniform(.04, .15, stocks)
        records.append(pd.DataFrame({
            "date": date, "ticker": [f"T{i:03}" for i in range(stocks)], "sector": sectors,
            "ret": returns, "mkt_ret": market, "prc": base_price, "shs_out": shares,
            "vol_sh": shares * rng.uniform(.01, .2, stocks), "equity": equity,
            "at": equity * 2, "debt": equity * .5, "cash": equity * .1,
            "rev_ttm": revenue, "gp_ttm": revenue * .3, "ebit_ttm": income * 1.2,
            "da_ttm": revenue * .03, "ni_ttm": income, "cfo_ttm": income * 1.3,
            "capex_ttm": revenue * .02, "d_nwc_ttm": revenue * .01,
            "div_ttm": income * .2, "buyback_ttm": income * .1, "issuance_ttm": income * .05,
        }))
        previous_return = returns
    return pd.concat(records, ignore_index=True)


def save_results(results: dict, output: Path, config: ModelConfig, demo: bool):
    output.mkdir(parents=True, exist_ok=True)
    for name, value in results.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(output / f"{name}.csv", index=False)
    latest_date = results["predictions"]["date"].max()
    joblib.dump({"models": results["models"], "features": FEATURES, "as_of": latest_date,
                 "config": asdict(config)}, output / "latest_models.joblib")
    metadata = {"synthetic_demo": demo, "features": FEATURES, "config": asdict(config),
                "prediction_start": str(results["predictions"]["date"].min().date()),
                "prediction_end": str(latest_date.date()), "python_library_versions": {
                    "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
                "target": "next calendar month's decimal total return",
                "timing": "Forecast after month-end information is available; train_label_end <= prediction_date."}
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Monthly raw panel CSV matching features.py.")
    source.add_argument("--demo", action="store_true", help="Use explicitly synthetic demonstration data.")
    parser.add_argument("--output", type=Path, default=Path("outputs/linear"))
    parser.add_argument("--min-train-months", type=int, default=36)
    parser.add_argument("--cv-splits", type=int, default=3)
    parser.add_argument("--validation-months", type=int, default=6)
    parser.add_argument("--train-window-months", type=int)
    parser.add_argument("--min-features", type=int, default=12)
    parser.add_argument("--no-sector-neutral", action="store_true")
    args = parser.parse_args()
    config = ModelConfig(min_train_months=args.min_train_months, cv_splits=args.cv_splits,
                         validation_months=args.validation_months, train_window_months=args.train_window_months,
                         min_features=args.min_features, sector_neutral=not args.no_sector_neutral)
    raw = synthetic_panel() if args.demo else pd.read_csv(args.input, dtype={"ticker": str})
    panel = prepare_panel(raw, config)
    print(f"{'SYNTHETIC DEMO' if args.demo else 'REAL INPUT'}: {len(FEATURES)} factors, {len(panel):,} eligible rows.")
    results = walk_forward(panel, config)
    save_results(results, args.output, config, args.demo)
    print(results["metrics"].round(5).to_string(index=False))
    print("\nLasso selection frequency (top 10; nonzero is not proof of significance):")
    print(results["lasso_selection"].head(10).round(5).to_string(index=False))
    print(f"\nSaved predictions, coefficients, selection frequencies and fitted models to {args.output.resolve()}")


if __name__ == "__main__":
    main()
