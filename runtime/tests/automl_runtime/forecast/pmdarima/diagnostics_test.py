#
# Copyright (C) 2021 Databricks, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import unittest

import pandas as pd
from pmdarima.arima import auto_arima, StepwiseContext

from databricks.automl_runtime.forecast.pmdarima.diagnostics import generate_cutoffs, \
    cross_validation, single_cutoff_forecast


class TestDiagnostics(unittest.TestCase):

    def setUp(self) -> None:
        num_rows = 15
        self.X = pd.concat([
            pd.to_datetime(pd.Series(range(num_rows), name="ds").apply(lambda i: f"2020-07-{i + 1}")),
            pd.Series(range(num_rows), name="y")
        ], axis=1)

    def test_generate_cutoffs_success(self):
        cutoffs = generate_cutoffs(self.X, horizon=3, unit="d", num_folds=3)
        self.assertEqual(cutoffs, [pd.Timestamp("2020-07-10 12:00:00"),
                                   pd.Timestamp("2020-07-12 00:00:00")])

    def test_generate_cutoffs_success_large_num_folds(self):
        cutoffs = generate_cutoffs(self.X, horizon=2, unit="d", num_folds=20)
        self.assertEqual(cutoffs, [pd.Timestamp("2020-07-07 00:00:00"),
                                   pd.Timestamp("2020-07-08 00:00:00"),
                                   pd.Timestamp("2020-07-09 00:00:00"),
                                   pd.Timestamp("2020-07-10 00:00:00"),
                                   pd.Timestamp("2020-07-11 00:00:00"),
                                   pd.Timestamp("2020-07-12 00:00:00"),
                                   pd.Timestamp('2020-07-13 00:00:00')])

    def test_generate_cutoffs_failure_horizon_too_large(self):
        with self.assertRaisesRegex(ValueError, "Less data than horizon after initial window. "
                                                "Make horizon shorter."):
            generate_cutoffs(self.X, horizon=9, unit="d", num_folds=3)

    def test_cross_validation_success(self):
        cutoffs = generate_cutoffs(self.X, horizon=3, unit="d", num_folds=3)
        y_train = self.X[self.X["ds"] <= cutoffs[0]].set_index("ds")
        with StepwiseContext(max_steps=1):
            model = auto_arima(y=y_train, m=1)

        expected_ds = self.X[self.X["ds"] > cutoffs[0]]["ds"]
        expected_cols = ["ds", "y", "cutoff", "yhat", "yhat_lower", "yhat_upper"]
        df_cv = cross_validation(model, self.X, cutoffs)
        self.assertEqual(df_cv["ds"].tolist(), expected_ds.tolist())
        self.assertEqual(set(df_cv.columns), set(expected_cols))

    def test_single_cutoff_forecast_success(self):
        cutoff_zero = self.X["ds"].min()
        cutoff_one = pd.Timestamp("2020-07-10 12:00:00")
        cutoff_two = pd.Timestamp("2020-07-12 00:00:00")
        y_train = self.X[self.X["ds"] <= cutoff_one].set_index("ds")
        test_df = self.X[self.X['ds'] > cutoff_one].copy()
        test_df["cutoff"] = [cutoff_one] * 2 + [cutoff_two] * 3
        with StepwiseContext(max_steps=1):
            model = auto_arima(y=y_train, m=1)

        expected_ds = test_df["ds"][:2]
        expected_cols = ["ds", "y", "cutoff", "yhat", "yhat_lower", "yhat_upper"]
        forecast_df = single_cutoff_forecast(model, test_df, cutoff_zero, cutoff_one)
        self.assertEqual(forecast_df["ds"].tolist(), expected_ds.tolist())
        self.assertEqual(set(forecast_df.columns), set(expected_cols))
