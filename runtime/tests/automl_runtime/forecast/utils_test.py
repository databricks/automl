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

from databricks.automl_runtime.forecast.utils import generate_cutoffs


class TestGenerateCutoffs(unittest.TestCase):

    def setUp(self) -> None:
        self.X = pd.DataFrame(
            pd.date_range(start="2020-07-01", end="2020-08-30", freq='d'), columns=["ds"]
        ).rename_axis("y").reset_index()

    def test_generate_cutoffs_success(self):
        cutoffs = generate_cutoffs(self.X, horizon=7, unit="D", num_folds=3, seasonal_period=7)
        self.assertEqual([pd.Timestamp('2020-08-19 12:00:00'), pd.Timestamp('2020-08-23 00:00:00')], cutoffs)

    def test_generate_cutoffs_success_large_num_folds(self):
        cutoffs = generate_cutoffs(self.X, horizon=7, unit="D", num_folds=20, seasonal_period=1)
        self.assertEqual([pd.Timestamp('2020-07-22 12:00:00'),
                          pd.Timestamp('2020-07-26 00:00:00'),
                          pd.Timestamp('2020-07-29 12:00:00'),
                          pd.Timestamp('2020-08-02 00:00:00'),
                          pd.Timestamp('2020-08-05 12:00:00'),
                          pd.Timestamp('2020-08-09 00:00:00'),
                          pd.Timestamp('2020-08-12 12:00:00'),
                          pd.Timestamp('2020-08-16 00:00:00'),
                          pd.Timestamp('2020-08-19 12:00:00'),
                          pd.Timestamp('2020-08-23 00:00:00')], cutoffs)

    def test_generate_cutoffs_success_with_gaps(self):
        df = pd.DataFrame(
            pd.date_range(start="2020-07-01", periods=30, freq='3d'), columns=["ds"]
        ).rename_axis("y").reset_index()
        cutoffs = generate_cutoffs(df, horizon=1, unit="D", num_folds=5, seasonal_period=1)
        self.assertEqual([pd.Timestamp('2020-09-16 00:00:00'),
                          pd.Timestamp('2020-09-19 00:00:00'),
                          pd.Timestamp('2020-09-22 00:00:00'),
                          pd.Timestamp('2020-09-25 00:00:00')], cutoffs)

    def test_generate_cutoffs_success_hourly(self):
        df = pd.DataFrame(
            pd.date_range(start="2020-07-01", periods=168, freq='h'), columns=["ds"]
        ).rename_axis("y").reset_index()
        expected_cutoffs = [pd.Timestamp('2020-07-07 08:00:00'),
                            pd.Timestamp('2020-07-07 11:00:00'),
                            pd.Timestamp('2020-07-07 14:00:00'),
                            pd.Timestamp('2020-07-07 17:00:00')]
        cutoffs = generate_cutoffs(df, horizon=6, unit="H", num_folds=5, seasonal_period=24)
        self.assertEqual(expected_cutoffs, cutoffs)

        cutoffs_different_seasonal_unit = generate_cutoffs(df, horizon=6, unit="H", num_folds=5,
                                                           seasonal_period=1, seasonal_unit="D")
        self.assertEqual(expected_cutoffs, cutoffs_different_seasonal_unit)

    def test_generate_cutoffs_success_weekly(self):
        df = pd.DataFrame(
            pd.date_range(start="2020-07-01", periods=52, freq='W'), columns=["ds"]
        ).rename_axis("y").reset_index()
        cutoffs = generate_cutoffs(df, horizon=4, unit="W", num_folds=3, seasonal_period=1)
        self.assertEqual([pd.Timestamp('2021-05-16 00:00:00'), pd.Timestamp('2021-05-30 00:00:00')], cutoffs)

    def test_generate_cutoffs_failure_horizon_too_large(self):
        with self.assertRaisesRegex(ValueError, "Less data than horizon after initial window. "
                                                "Make horizon shorter."):
            generate_cutoffs(self.X, horizon=20, unit="D", num_folds=3, seasonal_period=1)

    def test_generate_cutoffs_less_data(self):
        with self.assertRaisesRegex(ValueError, "Less data than horizon."):
            generate_cutoffs(self.X, horizon=100, unit="D", num_folds=3, seasonal_period=1)
