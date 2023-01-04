#
# Copyright (C) 2022 Databricks, Inc.
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

from databricks.automl_runtime.forecast.utils import generate_cutoffs
from databricks.automl_runtime.forecast.pmdarima.diagnostics import cross_validation, single_cutoff_forecast


class TestDiagnostics(unittest.TestCase):

    def setUp(self) -> None:
        num_rows = 15
        self.df = pd.concat([
            pd.to_datetime(pd.Series(range(num_rows), name="ds").apply(lambda i: f"2020-07-{i + 1}")),
            pd.Series(range(num_rows), name="y")
        ], axis=1)
        self.df_with_exogenous = pd.concat([
            pd.to_datetime(pd.Series(range(num_rows), name="ds").apply(lambda i: f"2020-07-{i + 1}")),
            pd.Series(range(num_rows), name="y"),
            pd.Series(range(num_rows), name="x1"),
            pd.Series(range(num_rows), name="x2")
        ], axis=1)

    def test_cross_validation_success(self):
        for df in [self.df, self.df_with_exogenous]:
            cutoffs = generate_cutoffs(df, horizon=3, unit="D", seasonal_period=1, seasonal_unit="D", num_folds=3)
            train_df = df[df["ds"] <= cutoffs[0]].set_index("ds")
            y_train = train_df[["y"]]
            X_train = train_df.drop(["y"], axis=1)
            with StepwiseContext(max_steps=1):
                model = auto_arima(
                    y=y_train,
                    X=X_train if not X_train.empty else None,
                    m=1)

            expected_ds = df[df["ds"] > cutoffs[0]]["ds"]
            added_cols = ["cutoff", "yhat", "yhat_lower", "yhat_upper"]
            df_cv = cross_validation(model, df, cutoffs)
            self.assertEqual(df_cv["ds"].tolist(), expected_ds.tolist())
            self.assertEqual(set(df_cv.columns), set(df.columns.tolist() + added_cols))

    def test_single_cutoff_forecast_success(self):
        for df in [self.df, self.df_with_exogenous]:
            cutoff_zero = df["ds"].min()
            cutoff_one = pd.Timestamp("2020-07-10 12:00:00")
            cutoff_two = pd.Timestamp("2020-07-12 00:00:00")
            train_df = df[df["ds"] <= cutoff_one].set_index("ds")
            y_train = train_df[["y"]]
            X_train = train_df.drop(["y"], axis=1)
            test_df = df[df['ds'] > cutoff_one].copy()
            test_df["cutoff"] = [cutoff_one] * 2 + [cutoff_two] * 3
            with StepwiseContext(max_steps=1):
                model = auto_arima(
                    y=y_train,
                    X=X_train if not X_train.empty else None,
                    m=1)

            expected_ds = test_df["ds"][:2]
            added_cols = ["cutoff", "yhat", "yhat_lower", "yhat_upper"]
            forecast_df = single_cutoff_forecast(model, test_df, cutoff_zero, cutoff_one)
            self.assertEqual(forecast_df["ds"].tolist(), expected_ds.tolist())
            self.assertEqual(set(forecast_df.columns), set(df.columns.tolist() + added_cols))
