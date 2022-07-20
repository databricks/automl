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

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from databricks.automl_runtime.sklearn.column_selector import ColumnSelector


class TestColumnSelector(unittest.TestCase):

    def test_select_columns(self):
        X_in = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                            columns=["a", "b", "c"])
        # select multiple columns
        selected_cols = ["a", "b"]
        X_out_expected = X_in[selected_cols]
        col_selector = ColumnSelector(selected_cols)
        X_out = col_selector.transform(X_in)
        np.testing.assert_array_almost_equal(X_out, X_out_expected,
                                             err_msg=f"Actual: {X_out}\nExpected: {X_out_expected}\n"
                                                     f"Equality: {X_out == X_out_expected}")
        # select single column
        selected_cols = "a"
        X_out_expected = X_in[[selected_cols]]
        col_selector = ColumnSelector(selected_cols)
        X_out = col_selector.transform(X_in)
        np.testing.assert_array_almost_equal(X_out, X_out_expected,
                                             err_msg=f"Actual: {X_out}\nExpected: {X_out_expected}\n"
                                                     f"Equality: {X_out == X_out_expected}")

    def test_select_column_in_pipeline(self):
        X_in = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                            columns=["a", "b", "c"])
        y = pd.DataFrame(np.array([[1], [0], [1]]),
                         columns=["label"])
        X_out_expected = np.array([1, 0, 1])

        standardizer = StandardScaler()
        selected_cols = ["a", "b"]
        col_selector = ColumnSelector(selected_cols)
        preprocessor = ColumnTransformer([("standardizer", standardizer, selected_cols)], remainder="drop")

        model = Pipeline([
            ("column_selector", col_selector),
            ("preprocessor", preprocessor),
            ("decision_tree", DecisionTreeClassifier())
        ])
        model.fit(X=X_in, y=y)
        # Add one column so that the dataframe for prediction is different with the data for training
        X_in["useless"] = 1
        X_out = model.predict(X_in)
        np.testing.assert_array_almost_equal(X_out, X_out_expected,
                                             err_msg=f"Actual: {X_out}\nExpected: {X_out_expected}\n"
                                                     f"Equality: {X_out == X_out_expected}")
