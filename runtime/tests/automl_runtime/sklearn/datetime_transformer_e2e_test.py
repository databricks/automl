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

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from databricks.automl_runtime.sklearn import DateTransformer
from databricks.automl_runtime.sklearn import TimestampTransformer


class TestDatetimeTransformer(unittest.TestCase):
    PRECISION = 5

    def setUp(self) -> None:
        num_rows = 4
        self.X = pd.concat([
            pd.Series(range(num_rows), name='int1'),
            pd.Series(range(num_rows), name='date1').apply(lambda i: "2020-07-0{}".format(i + 1)),
            pd.Series(range(num_rows), name='timestamp1').apply(
                lambda i: pd.Timestamp(2017, 2, 1, i + 5) if i < 3 else pd.Timestamp(pd.NaT)),
        ], axis=1)

    def test_timestamp_and_date_transformers_success(self):
        # This test set up is similar to how datatime transformers will be used in AutoML notebooks
        timestamp_transformer = TimestampTransformer()
        ohe_transformer = ColumnTransformer(
            [("ohe", OneHotEncoder(sparse=False), [timestamp_transformer.HOUR_COLUMN_INDEX])], remainder="passthrough")
        timestamp_preprocessor = Pipeline([
            ("extractor", timestamp_transformer),
            ("onehot_encoder", ohe_transformer)
        ])
        transformers = [("timestamp_timestamp1", timestamp_preprocessor, ["timestamp1"]),
                        ("date_date1", DateTransformer(), ["date1"])]
        preprocessor = ColumnTransformer(transformers, remainder="passthrough")

        X_transformed = preprocessor.fit_transform(self.X)
        X_expected = np.array([
            [0.0, 1.0, 0.0, 0.0, 1485925200, False, 0.9165622558699762, -0.39989202431974097,
             0.2424681428783799, 0.97015936819118, 0.5251794996758523,
             0.8509914765261879, 0.9659258262890683, 0.25881904510252074, False, False,
             1593561600, False, 0.9749279121818236, -0.22252093395631434,
             0.20129852008866006, 0.9795299412524945, 1.2246467991473532e-16, -1.0,
             False, False, 0],
            [0.0, 0.0, 1.0, 0.0, 1485928800, False, 0.9009688679024191,
             -0.43388373911755806, 0.2506525322587205, 0.9680771188662043,
             0.5257880785139989, 0.8506155985476382, 1.0, 6.123233995736766e-17, False,
             False, 1593648000, False, 0.43388373911755823, -0.900968867902419,
             0.39435585511331855, 0.9189578116202306, -0.017166329754707124,
             -0.9998526477050269, False, False, 1],
            [0.0, 0.0, 0.0, 1.0, 1485932400, False, 0.8841153935046098,
             -0.46726862827306204, 0.25881904510252074, 0.9659258262890683,
             0.5263963883313831, 0.8502392853495278, 0.9659258262890683, -0.25881904510252063,
             False, False, 1593734400, False, -0.433883739117558, -0.9009688679024191,
             0.5712682150947923, 0.8207634412072763, -0.03432760051324357,
             -0.9994106342455052, True, False, 2],
            [1.0, 0.0, 0.0, 0.0, 0, False, 0.43388373911755823, -0.900968867902419,
             0.20129852008866006, 0.9795299412524945, 0.01716632975470737,
             0.9998526477050269, 0.0, 1.0, True, True, 1593820800, True, -0.9749279121818236,
             -0.2225209339563146, 0.7247927872291199, 0.6889669190756866, -0.05147875477034649,
             -0.9986740898848305, True, False, 3],
        ])
        np.testing.assert_array_almost_equal(X_transformed, X_expected, decimal=self.PRECISION,
                                             err_msg=f"Actual: {X_transformed}\nExpected: {X_expected}\n"
                                                     f"Equality: {X_transformed == X_expected}")
