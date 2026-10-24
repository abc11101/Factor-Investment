"""Checks for timing and selection behavior, using only synthetic data."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from linear_models import (FEATURES, ModelConfig, model_pipeline, monthly_cv_splits,
                           prepare_panel, synthetic_panel, tune_model, validate_input, walk_forward)


def small_factor_panel():
    rng = np.random.default_rng(19)
    dates = pd.date_range("2020-01-31", periods=18, freq="ME")
    panel = pd.MultiIndex.from_product([dates, [f"T{i}" for i in range(20)]],
                                       names=["date", "ticker"]).to_frame(index=False)
    x = rng.normal(size=(len(panel), len(FEATURES)))
    panel[FEATURES] = x
    panel["target_date"] = panel["date"] + pd.offsets.MonthEnd(1)
    panel["target"] = .05 * x[:, 0] - .02 * x[:, 1] + rng.normal(0, .005, len(panel))
    panel.loc[panel["date"] == dates[-1], "target"] = np.nan
    panel["equal_weight_score"] = panel[FEATURES].mean(axis=1)
    return panel


class TimingTests(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig(min_train_months=12, cv_splits=2, validation_months=3,
                                  ridge_alphas=(.1, 10.), lasso_alphas=(.001, .01))

    def test_target_matches_next_calendar_month_after_shuffle(self):
        raw = synthetic_panel(months=15, stocks=10).sample(frac=1, random_state=4)
        panel = prepare_panel(raw, self.config)
        observed = raw.set_index(["date", "ticker"])["ret"]
        for row in panel.itertuples():
            expected = observed.get((row.target_date, row.ticker), np.nan)
            if np.isnan(expected):
                self.assertTrue(np.isnan(row.target))
            else:
                self.assertAlmostEqual(row.target, expected)

    def test_month_gap_and_duplicate_are_rejected(self):
        raw = synthetic_panel(months=15, stocks=10)
        with self.assertRaisesRegex(ValueError, "Gaps"):
            validate_input(raw.drop(raw.index[20]))
        with self.assertRaisesRegex(ValueError, "one row"):
            validate_input(pd.concat([raw, raw.iloc[:1]], ignore_index=True))

    def test_invalid_return_and_config_are_rejected(self):
        raw = synthetic_panel(months=15, stocks=10)
        raw.loc[0, "ret"] = -1.5
        with self.assertRaisesRegex(ValueError, "decimal total returns"):
            validate_input(raw)
        with self.assertRaisesRegex(ValueError, "exceed"):
            replace(self.config, min_train_months=6).validate()

    def test_folds_keep_months_together_and_respect_label_availability(self):
        panel = small_factor_panel().iloc[:12 * 20].copy()
        for fit, val in monthly_cv_splits(panel, self.config):
            train, validation = panel.iloc[fit], panel.iloc[val]
            self.assertLess(train.date.max(), validation.date.min())
            self.assertLessEqual(train.target_date.max(), validation.date.min())
            for date in validation.date.unique():
                self.assertEqual((validation.date == date).sum(), (panel.date == date).sum())

    def test_future_changes_cannot_change_earlier_forecasts_or_weights(self):
        panel = small_factor_panel()
        result = walk_forward(panel, self.config)
        first_date = result["predictions"].date.min()
        changed = panel.copy()
        # These labels are unknown at first_date, including its next-month target.
        changed.loc[changed.target_date > first_date, "target"] = 100.
        changed.loc[changed.date > first_date, FEATURES] *= 50.
        other = walk_forward(changed, self.config)
        for key in ["predictions", "coefficients"]:
            columns = [c for c in result[key].columns if c != "target"]
            before = result[key].loc[result[key].date == first_date, columns].reset_index(drop=True)
            after = other[key].loc[other[key].date == first_date, columns].reset_index(drop=True)
            assert_frame_equal(before, after)
        folds = result["validation_splits"]
        self.assertTrue((folds.train_label_end <= folds.validation_start).all())
        self.assertTrue((folds.validation_label_end <= folds.prediction_date).all())
        live = result["predictions"].loc[result["predictions"].date == panel.date.max()]
        self.assertTrue(live.target.isna().all())
        self.assertTrue(np.isfinite(live.prediction).all())
        # Exported factor weights and intercept must reconstruct the saved model.
        latest = panel.loc[panel.date == panel.date.max()]
        for name, pipeline in result["models"].items():
            weights = (result["coefficients"].loc[(result["coefficients"].model == name) &
                                                 (result["coefficients"].date == latest.date.max())]
                       .set_index("feature").loc[FEATURES, "coefficient"])
            intercept = result["fit_summary"].loc[(result["fit_summary"].model == name) &
                                                   (result["fit_summary"].date == latest.date.max()), "intercept"].iloc[0]
            imputed = pipeline.named_steps["imputer"].transform(latest[FEATURES])
            np.testing.assert_allclose(imputed @ weights.to_numpy() + intercept,
                                       pipeline.predict(latest[FEATURES]), atol=1e-12)

    def test_expanding_and_rolling_windows(self):
        result = walk_forward(small_factor_panel(), replace(self.config, train_window_months=12))
        self.assertEqual(set(result["fit_summary"].train_months), {12})


class SelectionTests(unittest.TestCase):
    def test_unconverged_candidate_is_excluded_and_reported(self):
        train = small_factor_panel().iloc[:12 * 20]
        config = ModelConfig(min_train_months=12, cv_splits=2, validation_months=3)
        factory = model_pipeline

        def limited_pipeline(name):
            return factory(name).set_params(regressor__max_iter=1, regressor__tol=1e-12)

        with patch("linear_models.model_pipeline", side_effect=limited_pipeline):
            _, chosen, candidates = tune_model("lasso", (1e-12, 1.), train, monthly_cv_splits(train, config))
        self.assertEqual(chosen["alpha"], 1.)
        self.assertEqual(candidates[0]["status"], "cv_fit_failed")

    def test_lasso_recovers_sparse_signal_and_keeps_empty_column(self):
        rng = np.random.default_rng(5)
        x = pd.DataFrame(rng.normal(size=(2000, len(FEATURES))), columns=FEATURES)
        y = .20 * x[FEATURES[0]] - .10 * x[FEATURES[6]] + rng.normal(0, .005, len(x))
        x[FEATURES[-1]] = np.nan
        model = model_pipeline("lasso").set_params(regressor__alpha=.005).fit(x, y)
        coef = model.named_steps["regressor"].coef_
        self.assertEqual(len(coef), len(FEATURES))
        self.assertEqual(set(np.flatnonzero(abs(coef) > 1e-10)), {0, 6})
        self.assertGreater(coef[0], 0)
        self.assertLess(coef[6], 0)
        self.assertTrue(np.isfinite(model.predict(x.iloc[:3])).all())


if __name__ == "__main__":
    unittest.main()
