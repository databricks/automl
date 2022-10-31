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
import json
import datetime
from unittest.mock import patch

import pandas as pd
from hyperopt import hp

from databricks.automl_runtime.forecast.prophet.forecast import ProphetHyperoptEstimator


class TestProphetHyperoptEstimator(unittest.TestCase):
    def setUp(self) -> None:
        self.num_rows = 12
        y_series = pd.Series(range(self.num_rows), name="y")
        self.df = pd.concat(
            [
                pd.to_datetime(
                    pd.Series(range(self.num_rows), name="ds").apply(
                        lambda i: f"2020-07-{i + 1}"
                    )
                ),
                y_series,
            ],
            axis=1,
        )
        self.df_datetime_date = pd.concat(
            [
                pd.Series(range(self.num_rows), name="ds").apply(
                    lambda i: datetime.date(2020, 7, i + 1)
                ),
                y_series,
            ],
            axis=1,
        )
        self.df_string_time = pd.concat(
            [
                pd.Series(range(self.num_rows), name="ds").apply(
                    lambda i: f"2020-07-{i + 1}"
                ),
                y_series,
            ],
            axis=1,
        )
        self.df_horizon_truncation = pd.concat(
            [
                pd.to_datetime(
                    pd.Series(range(21), name="ds").apply(lambda i: f"2020-07-{i + 1}")
                ),
                y_series,
            ],
            axis=1,
        )
        self.search_space = {
            "changepoint_prior_scale": hp.loguniform(
                "changepoint_prior_scale", -2.3, -0.7
            )
        }

    def test_sequential_training(self):
        hyperopt_estim = ProphetHyperoptEstimator(
            horizon=1,
            frequency_unit="d",
            metric="smape",
            interval_width=0.8,
            country_holidays="US",
            search_space=self.search_space,
            num_folds=2,
            trial_timeout=1000,
            random_state=0,
            is_parallel=False,
        )

        for df in [self.df, self.df_datetime_date, self.df_string_time]:
            results = hyperopt_estim.fit(df)
            self.assertAlmostEqual(results["mse"][0], 0)
            self.assertAlmostEqual(results["rmse"][0], 0, delta=1e-6)
            self.assertAlmostEqual(results["mae"][0], 0, delta=1e-7)
            self.assertAlmostEqual(results["mape"][0], 0)
            self.assertAlmostEqual(results["mdape"][0], 0)
            self.assertAlmostEqual(results["smape"][0], 0)
            self.assertAlmostEqual(results["coverage"][0], 1)
            # check the best result parameter is inside the search space
            model_json = json.loads(results["model_json"][0])
            self.assertGreaterEqual(model_json["changepoint_prior_scale"], 0.1)
            self.assertLessEqual(model_json["changepoint_prior_scale"], 0.5)

    @patch("databricks.automl_runtime.forecast.prophet.forecast.fmin")
    @patch("databricks.automl_runtime.forecast.prophet.forecast.Trials")
    @patch("databricks.automl_runtime.forecast.prophet.forecast.partial")
    def test_horizon_truncation(self, mock_partial, mock_trials, mock_fmin):
        hyperopt_estim = ProphetHyperoptEstimator(
            horizon=100,
            frequency_unit="d",
            metric="smape",
            interval_width=0.8,
            country_holidays="US",
            search_space=self.search_space,
            num_folds=2,
            trial_timeout=1000,
            random_state=0,
            is_parallel=False,
        )

        results = hyperopt_estim.fit(self.df_horizon_truncation)
        # the dataframe has 21 timestamps, which means the timedelta is 20. So validation horizon is at most 20/4=5
        kwargs = mock_partial.call_args_list[0].kwargs
        self.assertEqual(kwargs["horizon"], 5)
        self.assertEqual(len(kwargs["cutoffs"]), 1)

    @patch("databricks.automl_runtime.forecast.prophet.forecast.fmin")
    @patch("databricks.automl_runtime.forecast.prophet.forecast.Trials")
    @patch("databricks.automl_runtime.forecast.prophet.forecast.partial")
    def test_no_horizon_truncation(self, mock_partial, mock_trials, mock_fmin):
        horizon = 4
        num_folds = 2
        hyperopt_estim = ProphetHyperoptEstimator(
            horizon=horizon,
            frequency_unit="d",
            metric="smape",
            interval_width=0.8,
            country_holidays="US",
            search_space=self.search_space,
            num_folds=num_folds,
            trial_timeout=1000,
            random_state=0,
            is_parallel=False,
        )

        results = hyperopt_estim.fit(self.df_horizon_truncation)
        # the dataframe has 21 timestamps, which means the timedelta is 20. So validation horizon is at most 20/4=5
        kwargs = mock_partial.call_args_list[0].kwargs
        self.assertEqual(kwargs["horizon"], horizon)
        self.assertEqual(len(kwargs["cutoffs"]), num_folds)
