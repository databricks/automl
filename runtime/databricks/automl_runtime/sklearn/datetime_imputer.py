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

import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

class DatetimeImputer(TransformerMixin, BaseEstimator):
    """Imputer for date and timestamp data."""

    def __init__(self, strategy='mean', fill_value=None):
        """Create a `DatetimeImputer`.

        Parameters
        ----------
        strategy: imputation strategy, one of 'mean', 'median', 'most_frequent' or 'constant'

        fill_value: the value used when `strategy` is 'constant'
        """
        if strategy not in ('mean', 'median', 'most_frequent', 'constant'):
            raise ValueError(f'Unknown strategy: {strategy}')
        if strategy == 'constant' and not fill_value:
            raise ValueError('A `fill_value` need to be provided for `constant` strategy.')
        self.strategy = strategy
        self.fill_value = fill_value

    def fit(self, X, y=None):
        """Find necessary values (e.g. mean, median, or most_frequent) of the input data.

        Parameters
        ----------
        X: a pandas DataFrame whose values are date or timestamp.

        y: not used
        """
        self.fill_values = {}
        for col_name, col_value in X.iteritems():
            col_value = pd.to_datetime(col_value, errors="coerce")
            if self.strategy == 'mean':
                self.fill_values[col_name] = col_value.mean(skipna=True)
            elif self.strategy == 'median':
                self.fill_values[col_name] = col_value.median(skipna=True)
            elif self.strategy == 'most_frequent':
                self.fill_values[col_name] = col_value.mode(dropna=True)[0]
            elif self.strategy == 'constant':
                self.fill_values[col_name] = pd.to_datetime(self.fill_value)
            else:
                raise ValueError(f'Unknown strategy: {self.strategy}')  # pragma: no cover
        return self
            

    def transform(self, X):
        """Convert the input to datetime object and then impute the missing values.

        Parameters
        ----------
        X: a pandas DataFrame whose values are date or timestamp.
        """
        return X.apply(pd.to_datetime, errors="coerce").fillna(self.fill_values)

