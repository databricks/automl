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

from databricks.automl_runtime.sklearn.base_datetime_transformer import BaseDatetimeTransformer


class TimestampTransformer(BaseDatetimeTransformer):
    """
    Generate features from timestamp or datetime-like string column.
    """

    HOUR_COLUMN_INDEX = 10

    def transform(self, X):
        """
        Transform timestamp data to datetime features.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, 1)
            The only column is either a timestamp column or string column with
            datetime values encoded in ISO 8601 format.

        Returns
        -------
        X_tr : pd.DataFrame of shape (n_samples, 13)
            Transformed features.
        """
        # Convert column to datetime if data type is string and standardize to UTC
        X.iloc[:, 0] = X.iloc[:, 0].apply(pd.to_datetime, errors="coerce", utc=True).dt.tz_localize(None)
        X = X.fillna(pd.Timestamp(self.EPOCH))  # Fill NaT with the Unix epoch

        return self._generate_datetime_features(X, include_timestamp=True)
