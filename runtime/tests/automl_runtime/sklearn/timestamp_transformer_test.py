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
from sklearn.pipeline import Pipeline

from databricks.automl_runtime.sklearn import TimestampTransformer


class TestTimestampTransformer(unittest.TestCase):

    def setUp(self) -> None:
        num_rows = 4
        self.X = pd.concat([
            pd.Series(range(num_rows), name="int1"),
            pd.Series(range(num_rows), name="timestamp1").apply(
                lambda i: pd.Timestamp(2017, 2, 1, i + 5) if i < 3 else pd.Timestamp(pd.NaT)),
            pd.Series(range(num_rows), name="timestamp_str").apply(lambda i: f"2020-07-0{i+1} 01:23:45"),
        ], axis=1)
        self.timestamp_expected = np.array([
            [1485925200, False, 0.9165622558699762, -0.39989202431974097,
             0.2424681428783799, 0.97015936819118, 0.5251794996758523,
             0.8509914765261879, 0.9659258262890683, 0.25881904510252074, 5, False, False],
            [1485928800, False, 0.9009688679024191, -0.43388373911755806,
             0.2506525322587205, 0.9680771188662043, 0.5257880785139989,
             0.8506155985476382, 1.0, 6.123233995736766e-17, 6, False, False],
            [1485932400, False, 0.8841153935046098,
             -0.46726862827306204, 0.25881904510252074, 0.9659258262890683,
             0.5263963883313831, 0.8502392853495278, 0.9659258262890683,
             -0.25881904510252063, 7, False, False],
            [0, False, 0.43388373911755823, -0.900968867902419,
             0.20129852008866006, 0.9795299412524945, 0.01716632975470737,
             0.9998526477050269, 0.0, 1.0, 0, True, True],
        ])
        self.timestamp_str_expected = np.array([
            [1593566625, False, 0.9659258262890683, -0.25881904510252063,
             0.20956351245274862, 0.9777950369318034, -0.0007152988127704499,
             -0.9999997441737715, 0.25881904510252074, 0.9659258262890683, 1.0, False, False],
            [1593653025, False, 0.39989202431974136, -0.916562255869976,
             0.4021024289259159, 0.9155946901614702, -0.01788151877495909,
             -0.9998401128612018, 0.25881904510252074, 0.9659258262890683, 1.0, False, False],
            [1593739425, False, -0.4672686282730616, -0.88411539350461,
             0.578179224713827, 0.8159097891981183, -0.0350424689714891,
             -0.9993858240781597, 0.25881904510252074, 0.9659258262890683, 1.0, True, False],
            [1593825825, True, -0.9825664732332883, -0.18591160716291416,
             0.7305852951087796, 0.6828214455996658, -0.05219309199157016,
             -0.9986370117056345, 0.25881904510252074, 0.9659258262890683, 1.0, True, False],
        ])
        self.transformer = TimestampTransformer()

    def test_transform(self):
        timestamp_transformed = self.transformer.transform(self.X[["timestamp1"]])
        np.testing.assert_array_almost_equal(timestamp_transformed.to_numpy(), self.timestamp_expected, decimal=5,
                                             err_msg=f"Actual: {timestamp_transformed}\n"
                                                     f"Expected: {self.timestamp_expected}\n"
                                                     f"Equality: {timestamp_transformed == self.timestamp_expected}")

        timestamp_str_transformed = self.transformer.transform(self.X[["timestamp_str"]])
        np.testing.assert_array_almost_equal(timestamp_str_transformed.to_numpy(), self.timestamp_str_expected, decimal=5,
                                             err_msg=f"Actual: {timestamp_str_transformed}\n"
                                                     f"Expected: {self.timestamp_str_expected}\n"
                                                     f"Equality: {timestamp_str_transformed == self.timestamp_str_expected}")

    def test_with_pipeline(self):
        pipeline = Pipeline([("ts_transformer", self.transformer)])
        timestamp_transformed = pipeline.fit_transform(self.X[["timestamp1"]])
        np.testing.assert_array_almost_equal(timestamp_transformed.to_numpy(), self.timestamp_expected, decimal=5,
                                             err_msg=f"Actual: {timestamp_transformed}\n"
                                                     f"Expected: {self.timestamp_expected}\n"
                                                     f"Equality: {timestamp_transformed == self.timestamp_expected}")
