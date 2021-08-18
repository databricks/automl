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

import pandas as pd

from databricks.automl_runtime.feature.sklearn.base_datetime_transformer import BaseDateTimeTransformer


class DateTransformer(BaseDateTimeTransformer):
    """
    Generate features from date column.
    """

    def transform(self, X):
        """
        Transform date data to datetime features.
        :param X: A pandas dataframe of shape (n_samples, 1), where the only column is a date column
        :return: A pandas dataframe with transformed features
        """
        X.iloc[:, 0] = X.iloc[:, 0].apply(pd.to_datetime, errors="coerce")
        X = X.fillna(pd.Timestamp(self.EPOCH))  # Fill NaT with the Unix epoch

        return self._generate_datetime_features(X, include_timestamp=False)
