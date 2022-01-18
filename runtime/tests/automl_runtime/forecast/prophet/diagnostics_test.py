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
from prophet import Prophet

from databricks.automl_runtime.forecast.prophet.diagnostics import generate_cutoffs


class TestGenerateCutoffs(unittest.TestCase):

    def setUp(self) -> None:
        num_rows = 9
        self.X = pd.concat([
            pd.to_datetime(pd.Series(range(num_rows), name="ds")
                           .apply(lambda i: f"2020-07-{3*i+1}")),
            pd.Series(range(num_rows), name="y")
        ], axis=1)
        self.model = Prophet()
        self.model.fit(self.X)

    def test_generate_cutoffs(self):
        horizon_timedelta = pd.to_timedelta(2, unit="days")
        cutoffs = generate_cutoffs(self.model, horizon=horizon_timedelta, num_folds=3)
        self.assertEqual(cutoffs, [pd.Timestamp("2020-07-20 00:00:00"),
                                   pd.Timestamp("2020-07-23 00:00:00")])

    def test_generate_cutoffs_large_num_folds(self):
        horizon_timedelta = pd.to_timedelta(2, unit="days")
        cutoffs = generate_cutoffs(self.model, horizon=horizon_timedelta, num_folds=20)
        self.assertEqual(cutoffs, [pd.Timestamp("2020-07-08 00:00:00"),
                                   pd.Timestamp("2020-07-11 00:00:00"),
                                   pd.Timestamp("2020-07-14 00:00:00"),
                                   pd.Timestamp("2020-07-17 00:00:00"),
                                   pd.Timestamp('2020-07-20 00:00:00'),
                                   pd.Timestamp('2020-07-23 00:00:00')])

    def test_generate_cutoffs_large_horizon(self):
        horizon_timedelta = pd.to_timedelta(9, unit="days")
        with self.assertRaisesRegex(ValueError, "Less data than horizon after initial window. "
                                                "Make horizon or initial shorter."):
            generate_cutoffs(self.model, horizon=horizon_timedelta, num_folds=3)

    def test_generate_cutoffs_less_data(self):
        horizon_timedelta = pd.to_timedelta(30, unit="days")
        with self.assertRaisesRegex(ValueError, "Less data than horizon."):
            generate_cutoffs(self.model, horizon=horizon_timedelta, num_folds=3)
