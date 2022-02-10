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
    def __init__(self, strategy='median', fill_value=None):
        if strategy not in ('mean', 'median', 'most_frequent', 'constant'):
            raise ValueError(f'Unknown strategy: {strategy}')
        if strategy == 'constant' and not fill_value:
            raise ValueError('A `fill_value` need to be provided for `constant` strategy.')
        self.strategy = strategy
        self.fill_value = fill_value

    def fit(self, X, y=None):
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
                raise ValueError(f'Unknown strategy: {strategy}')  # pragma: no cover
        return self
            

    def transform(self, X):
        return pd.DataFrame({
            col_name: pd.to_datetime(col_value, errors="coerce").fillna(
                self.fill_values[col_name])
            for col_name, col_value in X.iteritems()
        })


