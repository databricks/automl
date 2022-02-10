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

import pandas as pd
from pandas.testing import assert_frame_equal
import unittest

from databricks.automl_runtime.sklearn.datetime_imputer import DatetimeImputer

class TestDatetimeImputer(unittest.TestCase):
    def get_test_df(self, fill_value_1=None, fill_value_2=None):
        return pd.DataFrame({
            "f1": ["2021-01-01", fill_value_1, "2021-01-03", "2021-01-05", "2021-01-07", "2021-01-07"],
            "f2": [f"2022-01-01 00:0{i}" for i in [1,3,5,7,7]] + [fill_value_2]})

    def to_datetime(self, X):
        return pd.DataFrame({
            col_name: pd.to_datetime(col_value)
            for col_name, col_value in X.iteritems()})

    def test_imputer_median(self):
        assert_frame_equal(
                DatetimeImputer().fit_transform(self.get_test_df()),
                self.to_datetime(self.get_test_df("2021-01-05", "2022-01-01 00:05")))
        assert_frame_equal(
                DatetimeImputer(strategy="median").fit_transform(self.get_test_df()),
                self.to_datetime(self.get_test_df("2021-01-05", "2022-01-01 00:05")))

    def test_imputer_mean(self):
        assert_frame_equal(
                DatetimeImputer(strategy="mean").fit_transform(self.get_test_df()),
                self.to_datetime(self.get_test_df("2021-01-04 14:24", "2022-01-01 00:04:36")))

    def test_imputer_most_frequent(self):
        assert_frame_equal(
                DatetimeImputer(strategy="most_frequent").fit_transform(self.get_test_df()),
                self.to_datetime(self.get_test_df("2021-01-07", "2022-01-01 00:07")))

    def test_imputer_constant(self):
        assert_frame_equal(
                DatetimeImputer(strategy="constant", fill_value="1970-01-01").fit_transform(self.get_test_df()),
                self.to_datetime(self.get_test_df("1970-01-01", "1970-01-01 00:00")))

    def test_validate_input(self):
        with self.assertRaises(ValueError):
            DatetimeImputer(strategy='foo')
        with self.assertRaises(ValueError):
            DatetimeImputer(strategy='constant', fill_value=None)
