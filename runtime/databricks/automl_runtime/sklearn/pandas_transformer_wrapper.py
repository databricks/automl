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
from __future__ import annotations
from typing import List

import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

class PandasTransformerWrapper(TransformerMixin, BaseEstimator):
    """
    A wrapper of `SimpleImputer` with the support for pandas dataframe output.
    """

    def __init__(self, estimator: BaseEstimator) -> None:
        self.estimator = estimator

    def fit(self, X: pd.DataFrame, y: pd.DataFrame=None) -> PandasTransformerWrapper:
        """Fits the imputer on X

        Parameters
        ----------

        X : pd.DataFrame of shape = [n_samples, n_features]
            Training dataframe, where n_samples is the number of samples
            and n_features is the number of features.
        y : Not used, present here for API consistency by convention.
        """
        self.columns = X.columns
        self.estimator = self.estimator.fit(X=X, y=y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute all missing values in `X`.

        Parameters
        ----------

        X : pd.DataFrame of shape = [n_samples, n_features]
            Training dataframe, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------

        pd.DataFrame of shape (n_samples, n_features_encoded)
            Transformed features.
        """
        return pd.DataFrame(self.estimator.transform(X), columns=self.columns)

    def fit_transform(self, X: pd.DataFrame, y: pd.DataFrame=None):
        """Learn a list of feature name -> indices mappings and transform X.
        """
        self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self, input_features: List[str]=None):
        """Get output feature names for transformation.
        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Not used, present here for API consistency by convention.
        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        return self.columns.to_list()
