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

from prophet import Prophet

from databricks.automl_runtime.forecast.utils import generate_cutoffs
from databricks.automl_runtime.forecast.prophet.diagnostics import cross_validation


class TestDiagnostics(unittest.TestCase):

    def setUp(self) -> None:
        num_rows = 15
        self.X = pd.concat([
            pd.to_datetime(pd.Series(range(num_rows), name="ds").apply(lambda i: f"{2020+i//12:04d}-{i%12+1:02d}-15")),
            pd.Series(range(num_rows), name="y")
        ], axis=1)

    def test_cross_validation_success(self):
        cutoffs = generate_cutoffs(self.X, horizon=3, unit="MS", seasonal_period=1, seasonal_unit="D", num_folds=3)
        model = Prophet()
        model.fit(self.X)

        horizon = pd.DateOffset(months=3)
        expected_ds = pd.concat([self.X[(self.X["ds"] > cutoff) & \
                (self.X["ds"] <= cutoff + horizon)]["ds"]
                for cutoff in cutoffs])
        expected_cols = ["ds", "y", "cutoff", "yhat", "yhat_lower", "yhat_upper"]
        df_cv = cross_validation(model, horizon=horizon, cutoffs=cutoffs)
        self.assertEqual(df_cv["ds"].tolist(), expected_ds.tolist())
        self.assertEqual(set(df_cv.columns), set(expected_cols))
