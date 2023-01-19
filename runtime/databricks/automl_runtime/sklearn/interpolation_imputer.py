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
from typing import Any, Dict, List

import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

DEFAULT_PARAMS = dict(
        method="linear",
        limit_direction="both",
        axis=0)


class InterpolationImputer(TransformerMixin, BaseEstimator):
    """Null value Imputer by interpolation
    """

    def __init__(self,
                 *,
                 impute_params: Dict[str, Any]=None,
                 impute_all: bool=True,
        ):
        """Create a `DatetimeImputer`.

        Parameters
        ----------
        impute_params: `dict` or None, default None
        Params to pass to the imputation algorithm.
        See `pandas.DataFrame.interpolate` for their respective options.

        impute_all: boolean, default True
        Indicator to decide whether to impute all null values. If True, if the custom
        imutation method cannot fill in all null values, use the default method if fill
        in the nulls.
        """
        self._impute_params = impute_params or DEFAULT_PARAMS
        self._impute_all = impute_all

    def fit(self, X: pd.DataFrame, y: pd.DataFrame=None) -> InterpolationImputer:
        self.columns = X.columns
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute the missing values with `pandas.DataFrame.interpolate`.

        Parameters
        ----------
        X: a pandas DataFrame whose values are date or timestamp.
        """
        assert isinstance(X, pd.DataFrame)
        self.columns = X.columns

        X_imputed = X.interpolate(**self._impute_params)
        if self._impute_all:
            X_imputed.interpolate(inplace=True, **DEFAULT_PARAMS)

        return X_imputed

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
