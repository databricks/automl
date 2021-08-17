import numpy as np
import pandas as pd
import holidays

from abc import ABC, abstractmethod

from sklearn.base import TransformerMixin, BaseEstimator


class BaseDateTimeTransformer(ABC, TransformerMixin, BaseEstimator):
    EPOCH = "1970-01-01"
    HOURS_IN_DAY = 24
    DAYS_IN_WEEK = 7
    DAYS_IN_MONTH = 31
    DAYS_IN_YEAR = 366  # Account for leap years
    WEEKEND_START = 5

    def fit(self, X, y=None):
        """
        Do nothing and return the estimator unchanged. This method is just there to implement
        the usual API and hence work in pipelines.
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
    def _generate_features(cls, X, include_timestamp=True):
        """
        Extract relevant features from the datetime column.
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
