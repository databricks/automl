#
# Copyright (C) 2023 Databricks, Inc.
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
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from databricks.automl_runtime.sklearn import PandasTransformerWrapper


class TestPandasTransformerWrapper(unittest.TestCase):
    
    def setUp(self) -> None:
        missing_values = np.nan
        feature_names = np.array(["a", "b", "c", "d"], dtype=object)
        self.X = pd.DataFrame(
            [
                [missing_values, missing_values, 1, missing_values],
                [4, 1, 2, 10],
            ],
            columns=feature_names,
        )
        self.expected_output_X = [[4, 1, 1, 10], [4, 1, 2, 10]]
        self.expected_columns = {"a", "b", "c", "d"}

    def test_fit_and_transform(self):
        imputer = PandasTransformerWrapper(SimpleImputer()).fit(self.X)
        output_df = imputer.transform(self.X)
        self.assertTrue(isinstance(output_df, pd.DataFrame))
        self.assertCountEqual(self.expected_columns, set(output_df.columns))
        np.testing.assert_almost_equal(output_df.to_numpy(), self.expected_output_X)

    def test_pipeline(self):
        output_df = Pipeline([("imputer", PandasTransformerWrapper(SimpleImputer()))]).fit_transform(self.X)
        self.assertTrue(isinstance(output_df, pd.DataFrame))
        self.assertCountEqual(self.expected_columns, set(output_df.columns))
        np.testing.assert_almost_equal(output_df.to_numpy(), self.expected_output_X)
