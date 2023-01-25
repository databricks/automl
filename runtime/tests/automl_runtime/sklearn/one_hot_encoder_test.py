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
import numpy as np
from scipy import sparse
from databricks.automl_runtime.sklearn import OneHotEncoder
from sklearn.pipeline import Pipeline


class TestOneHotEncoder(unittest.TestCase):

    def setUp(self) -> None:
        self.numeric_X = [[1, 2, 3, 4]]
        self.expected_numeric_x = [[1, 2, 3, 4]]

        self.categorical_X = [['Male', 1], ['Female', 3], ['Female', 2]]
        self.expected_categorical_X_value = [[1, 0, 1], [0, 1, 3], [0, 1, 2]]
        self.expected_categorical_X_indicator = [[1, 0, 0, 1], [0, 1, 0, 3],
                                                 [0, 1, 0, 2]]

        self.categorical_X_with_none = [['Male', 1], [np.nan, 2], [None, 3],
                                        [pd.NA, 1]]
        self.expected_categorical_X_with_none = [[1, 0, 0, 1], [0, 1, 0, 2],
                                                 [0, 1, 0, 3], [0, 0, 1, 1]]

    def test_fit_and_transform(self):
        numerical_output = OneHotEncoder().fit_transform(self.numeric_X)
        self.assertTrue(isinstance(numerical_output, sparse.csr_matrix))
        np.testing.assert_almost_equal(numerical_output.toarray(),
                                       self.expected_numeric_x)

        categorical_output = OneHotEncoder().fit_transform(self.categorical_X)
        self.assertTrue(isinstance(categorical_output, sparse.csr_matrix))
        np.testing.assert_almost_equal(
            categorical_output.toarray(),
            self.expected_categorical_X_value)  # default is value
        categorical_output = OneHotEncoder(
            handle_unknown="indicator").fit_transform(self.categorical_X)
        np.testing.assert_almost_equal(categorical_output.toarray(),
                                       self.expected_categorical_X_indicator)

        categorical_output_with_none = OneHotEncoder(
            handle_unknown="indicator").fit_transform(
                self.categorical_X_with_none)
        self.assertTrue(
            isinstance(categorical_output_with_none, sparse.csr_matrix))
        np.testing.assert_almost_equal(categorical_output_with_none.toarray(),
                                       self.expected_categorical_X_with_none)

    def test_sparsity_equals_false(self):
        # similar to the previous test, but the sparse flag is set to the opposite
        transformer = OneHotEncoder(sparse=False)
        transformer.fit(self.numeric_X)
        numerical_output = transformer.transform(self.numeric_X)
        self.assertFalse(isinstance(numerical_output, sparse.csr_matrix))
        np.testing.assert_almost_equal(numerical_output,
                                       self.expected_numeric_x)

        transformer = OneHotEncoder(sparse=False)
        transformer.fit(self.categorical_X)
        categorical_output = transformer.transform(self.categorical_X)
        self.assertFalse(isinstance(categorical_output, sparse.csr_matrix))
        np.testing.assert_almost_equal(
            categorical_output,
            self.expected_categorical_X_value)  # default is value

        transformer = OneHotEncoder(sparse=False, handle_unknown="indicator")
        transformer.fit(self.categorical_X)
        categorical_output = transformer.transform(self.categorical_X)
        self.assertFalse(isinstance(categorical_output, sparse.csr_matrix))
        np.testing.assert_almost_equal(categorical_output,
                                       self.expected_categorical_X_indicator)

        transformer = OneHotEncoder(sparse=False, handle_unknown="indicator")
        transformer.fit(self.categorical_X_with_none)
        categorical_output_with_none = transformer.transform(
            self.categorical_X_with_none)
        self.assertFalse(
            isinstance(categorical_output_with_none, sparse.csr_matrix))
        np.testing.assert_almost_equal(categorical_output_with_none,
                                       self.expected_categorical_X_with_none)

    def test_pipeline(self):
        numerical_output = Pipeline([("onehotencoder", OneHotEncoder())
                                     ]).fit_transform(self.numeric_X)
        self.assertTrue(isinstance(numerical_output, sparse.csr_matrix))
        np.testing.assert_almost_equal(numerical_output.toarray(),
                                       self.expected_numeric_x)

        categorical_output = Pipeline([("onehotencoder", OneHotEncoder())
                                       ]).fit_transform(self.categorical_X)
        self.assertTrue(isinstance(categorical_output, sparse.csr_matrix))
        np.testing.assert_almost_equal(categorical_output.toarray(),
                                       self.expected_categorical_X_value)

        categorical_output_with_none = Pipeline([
            ("onehotencoder", OneHotEncoder(handle_unknown="indicator"))
        ]).fit_transform(self.categorical_X_with_none)
        self.assertTrue(
            isinstance(categorical_output_with_none, sparse.csr_matrix))
        np.testing.assert_almost_equal(categorical_output_with_none.toarray(),
                                       self.expected_categorical_X_with_none)

        # testing sparse=False as well (should I make a separate test)
        categorical_output_with_none = Pipeline([
            ("onehotencoder",
             OneHotEncoder(sparse=False, handle_unknown="indicator"))
        ]).fit_transform(self.categorical_X_with_none)
        self.assertFalse(
            isinstance(categorical_output_with_none, sparse.csr_matrix))
        np.testing.assert_almost_equal(categorical_output_with_none,
                                       self.expected_categorical_X_with_none)
