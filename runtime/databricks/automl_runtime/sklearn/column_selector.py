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

from abc import ABC
from typing import List, Union

import pandas as pd

from sklearn.base import TransformerMixin, BaseEstimator


class ColumnSelector(ABC, TransformerMixin, BaseEstimator):
    """
    Transformer to select specific columns from a dataset

    Parameters
    ----------
    cols: A list specifying the feature column names to be selected.
    """

    def __init__(self, cols: Union[List[str], str]) -> None:
        if not isinstance(cols, list):
            self.cols = [cols]
        else:
            self.cols = cols

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Do nothing and return the estimator unchanged.
        This method is just there to implement the usual API and hence work in pipelines.
        """
        return self

    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """
        Select the chosen columns
        :param X: pd.DataFrame of shape (n_samples, n_features)
            n_features is the number of feature columns.
        :return: pd.DataFrame of shape (n_samples, k_features)
            Subset of the feature space where k_features <= n_features.
        """
        X = X.copy()
        return X[self.cols]

    def get_feature_names_out(self, input_features=None):
        return self.cols
