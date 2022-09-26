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

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import holidays

from sklearn.base import TransformerMixin, BaseEstimator


class BaseDatetimeTransformer(ABC, TransformerMixin, BaseEstimator):
    """
    Abstract transformer for datetime features.
    Implements common functions to transform date and timestamp.
    """

    EPOCH = "1970-01-01"
    HOURS_IN_DAY = 24
    DAYS_IN_WEEK = 7
    DAYS_IN_MONTH = 31
    DAYS_IN_YEAR = 366  # Account for leap years
    WEEKEND_START = 5

    def fit(self, X, y=None):
        """
        Do nothing and return the estimator unchanged.
        This method is just there to implement the usual API and hence work in pipelines.
        """
        return self

    @abstractmethod
    def transform(self, X):
        pass  # pragma: no cover

    @staticmethod
    def _cyclic_transform(unit, period):
        """
        Encode cyclic features with a sine/cosine transform.

        Parameters
        ----------
        unit : array-like
            cyclic features to transform.

        period : int
            period for the sine/cosine transform.

        Returns
        -------
        cyclic_transformed : list of two array-like objects
            The sine and cosine transformed cyclic features.
        """
        return [np.sin(2 * np.pi * unit / period), np.cos(2 * np.pi * unit / period)]

    @classmethod
    def _generate_datetime_features(cls, X, include_timestamp=True):
        """
        Extract relevant features from the datetime column.

        For each datetime column, extract relevant information from the date:
        - Unix timestamp
        - whether the date is a weekend
        - whether the date is a holiday
        - cyclic features

        For cyclic features, plot the values along a unit circle to encode temporal proximity:
        - hour of the day
        - hours since the beginning of the week
        - hours since the beginning of the month
        - hours since the beginning of the year

        Additionally, extract extra information from columns with timestamps:
        - hour of the day

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, 1)
            The only column is a datetime column.

        include_timestamp : boolean, default=True
            Indicates if the input column includes timestamp information.

        Returns
        -------
        X_features : pd.DataFrame
            The generated features.
        """
        col = X.columns[0]
        dt = X[col].dt
        unix_seconds = (X[col] - pd.Timestamp(cls.EPOCH)) // pd.Timedelta("1s")
        features = [
            unix_seconds,
            dt.dayofweek >= cls.WEEKEND_START,
            *cls._cyclic_transform(dt.dayofweek * cls.HOURS_IN_DAY + dt.hour, cls.DAYS_IN_WEEK * cls.HOURS_IN_DAY),
            *cls._cyclic_transform(dt.day * cls.HOURS_IN_DAY + dt.hour, cls.DAYS_IN_MONTH * cls.HOURS_IN_DAY),
            *cls._cyclic_transform(dt.dayofyear * cls.HOURS_IN_DAY + dt.hour, cls.DAYS_IN_YEAR * cls.HOURS_IN_DAY),
        ]

        # Extract additional features for columns with timestamps
        if include_timestamp:
            features.extend([*cls._cyclic_transform(dt.hour, cls.HOURS_IN_DAY), dt.hour])

        for holiday_calendar in [holidays.UnitedStates(), holidays.EuropeanCentralBank()]:
            features.append(X[col].apply(lambda datetime: datetime in holiday_calendar))
        ans = pd.concat(features, axis=1)
        # Give non-string column names to avoid duplicated feature names, sklearn 1.0 does not work on duplicated
        # feature nanmes.
        ans.columns = range(len(ans.columns))
        return ans
