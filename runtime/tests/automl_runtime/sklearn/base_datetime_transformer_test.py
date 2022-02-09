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
import numpy as np

from numpy.testing import assert_array_almost_equal

from databricks.automl_runtime.sklearn.base_datetime_transformer import BaseDatetimeTransformer
from databricks.automl_runtime.sklearn.date_transformer import DateTransformer
from databricks.automl_runtime.sklearn.timestamp_transformer import TimestampTransformer


class TestBaseDatetimeTransformer(unittest.TestCase):
    PRECISION = 5

    def setUp(self) -> None:
        num_rows = 4
        self.X = pd.concat([
            pd.Series(range(num_rows), name='timestamp1').apply(lambda i: pd.Timestamp(2017, 2, 1, i + 5)),
        ], axis=1)

    def test_cyclic_transform(self):
        cyclic_transformed = pd.concat(BaseDatetimeTransformer._cyclic_transform(self.X['timestamp1'].dt.hour, 24),
                                       axis=1)
        cyclic_expected = np.array([[0.9659258262890683, 0.25881904510252074],
                                    [1.0, 6.123233995736766e-17],
                                    [0.9659258262890683, -0.25881904510252063],
                                    [0.8660254037844387, -0.4999999999999998]])
        np.testing.assert_array_almost_equal(cyclic_transformed, cyclic_expected, decimal=self.PRECISION,
                                             err_msg=f"Actual: {cyclic_transformed}\nExpected: {cyclic_expected}\n"
                                                     f"Equality: {cyclic_transformed == cyclic_expected}")

    def test_generate_datetime_features(self):
        feature_generated = BaseDatetimeTransformer._generate_datetime_features(self.X[['timestamp1']]).to_numpy()
        feature_expected = np.array([
            [1485925200, False, 0.9165622558699762, -0.39989202431974097,
             0.2424681428783799, 0.97015936819118, 0.5251794996758523,
             0.8509914765261879, 0.9659258262890683, 0.25881904510252074, 5, False, False],
            [1485928800, False, 0.9009688679024191, -0.43388373911755806,
             0.2506525322587205, 0.9680771188662043, 0.5257880785139989,
             0.8506155985476382, 1.0, 6.123233995736766e-17, 6, False, False],
            [1485932400, False, 0.8841153935046098, -0.46726862827306204,
             0.25881904510252074, 0.9659258262890683, 0.5263963883313831,
             0.8502392853495278, 0.9659258262890683, -0.25881904510252063, 7, False,
             False],
            [1485936000, False, 0.8660254037844387, -0.4999999999999998,
             0.26696709897415166, 0.9637056438899408, 0.5270044288167619,
             0.8498625371243979, 0.8660254037844387, -0.4999999999999998, 8, False, False],
        ])
        np.testing.assert_array_almost_equal(feature_generated, feature_expected, decimal=self.PRECISION,
                                             err_msg=f"Actual: {feature_generated}\nExpected: {feature_expected}\n"
                                                     f"Equality: {feature_generated == feature_expected}")

    def test_impute(self):
        def get_df(fill_value=None):
            return pd.DataFrame({
                "X": ["2021-01-01", fill_value, "2021-01-03", "2021-01-05", "2021-01-07", "2021-01-07"]})

        assert_array_almost_equal(
                DateTransformer("median").fit_transform(get_df()),
                DateTransformer().transform(get_df("2021-01-05")),
                err_msg="impute with median yield unexpected results")

        assert_array_almost_equal(
                DateTransformer("most_frequent").fit_transform(get_df()),
                DateTransformer().transform(get_df("2021-01-07")),
                err_msg="impute with most_frequent yield unexpected results")

        assert_array_almost_equal(
                DateTransformer().fit_transform(get_df()),
                DateTransformer().transform(get_df("1970-01-01")),
                err_msg="impute with default value yield unexpected results")

        assert_array_almost_equal(
                TimestampTransformer("mean").fit_transform(get_df()),
                TimestampTransformer().transform(get_df("2021-01-04 14:24")),
                err_msg="impute with mean yield unexpected results")

        assert_array_almost_equal(
                TimestampTransformer("2000-01-23").fit_transform(get_df()),
                TimestampTransformer().transform(get_df("2000-01-23")),
                err_msg="impute with custom value yield unexpected results")
