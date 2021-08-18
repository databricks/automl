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

import numpy as np
import pandas as pd
import holidays

from abc import ABC, abstractmethod

from sklearn.base import TransformerMixin, BaseEstimator


class BaseDateTimeTransformer(ABC, TransformerMixin, BaseEstimator):
    """
    Abstract transformer for datetime feature.
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
        Do nothing and return the estimator unchanged. This method is just there to implement
        the usual API and hence work in sklearn pipelines.
        """
        return self

    @abstractmethod
    def transform(self, X):
        pass

    @staticmethod
    def _cyclic_transform(unit, period):
        """
        Encode cyclic features with a sine/cosine transform.
        """
        return [np.sin(2 * np.pi * unit / period), np.cos(2 * np.pi * unit / period)]

    @classmethod
    def _generate_datetime_features(cls, X, include_timestamp=True):
        """
        Extract relevant features from the datetime column.
        :param X: A pandas dataframe of shape (n_samples, 1), where the only column is a datetime column
        :param include_timestamp: A boolean indicates if the input column include timestamp information
        :return: A pandas dataframe with generated features
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

        return pd.concat(features, axis=1)
