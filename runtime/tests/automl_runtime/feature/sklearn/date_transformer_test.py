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

from databricks.automl_runtime.feature.sklearn.date_transformer import DateTransformer


class TestDateTransformer(unittest.TestCase):

    def setUp(self) -> None:
        num_rows = 4
        self.X = pd.concat([
            pd.Series(range(num_rows), name='int1'),
            pd.Series(range(num_rows), name='date1').apply(lambda i: "2020-07-0{}".format(i + 1))
        ], axis=1)
        self.date_expected = np.array([
            [1593561600, False, 0.9749279121818236, -0.22252093395631434,
             0.20129852008866006, 0.9795299412524945, 1.2246467991473532e-16, -1.0,
             False, False],
            [1593648000, False, 0.43388373911755823, -0.900968867902419,
             0.39435585511331855, 0.9189578116202306, -0.017166329754707124,
             -0.9998526477050269, False, False],
            [1593734400, False, -0.433883739117558, -0.9009688679024191,
             0.5712682150947923, 0.8207634412072763, -0.03432760051324357,
             -0.9994106342455052, True, False],
            [1593820800, True, -0.9749279121818236, -0.2225209339563146,
             0.7247927872291199, 0.6889669190756866, -0.05147875477034649,
             -0.9986740898848305, True, False],
        ])
        self.transformer = DateTransformer()

    def test_transform(self):
        date_transformed = self.transformer.transform(self.X[['date1']])
        np.testing.assert_array_almost_equal(date_transformed.to_numpy(), self.date_expected, decimal=5,
                                             err_msg=f"Actual: {date_transformed}\nExpected: {self.date_expected}\n"
                                                     f"Equality: {date_transformed == self.date_expected}")

    def test_with_pipeline(self):
        pipeline = Pipeline([("date_transformer", self.transformer)])
        date_transformed = pipeline.fit_transform(self.X[['date1']])
        np.testing.assert_array_almost_equal(date_transformed.to_numpy(), self.date_expected, decimal=5,
                                             err_msg=f"Actual: {date_transformed}\nExpected: {self.date_expected}\n"
                                                     f"Equality: {date_transformed == self.date_expected}")
