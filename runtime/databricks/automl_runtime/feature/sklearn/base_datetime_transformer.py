import numpy as np
import pandas as pd
import holidays

from sklearn.base import TransformerMixin, BaseEstimator


class BaseDateTimeTransformer(TransformerMixin, BaseEstimator):
    epoch = "1970-01-01"
    hours_in_day = 24
    days_in_week = 7
    days_in_month = 31
    days_in_year = 366  # Account for leap years
    weekend_start = 5

    def fit(self, X, y=None):
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError

    @classmethod
    def _cyclic_transform(cls, unit, period):
        return [np.sin(2 * np.pi * unit / period), np.cos(2 * np.pi * unit / period)]

    @classmethod
    def _generate_features(cls, X, include_timestamp=True):
        col = X.columns[0]
        dt = X[col].dt
        unix_seconds = (X[col] - pd.Timestamp(cls.epoch)) // pd.Timedelta("1s")
        features = [
            unix_seconds,
            dt.dayofweek >= cls.weekend_start,
            *cls._cyclic_transform(dt.dayofweek * cls.hours_in_day + dt.hour, cls.days_in_week * cls.hours_in_day),
            *cls._cyclic_transform(dt.day * cls.hours_in_day + dt.hour, cls.days_in_month * cls.hours_in_day),
            *cls._cyclic_transform(dt.dayofyear * cls.hours_in_day + dt.hour, cls.days_in_year * cls.hours_in_day),
        ]

        # Extract additional features for columns with timestamps
        if include_timestamp:
            features.extend([*cls._cyclic_transform(dt.hour, cls.hours_in_day), dt.hour])

        for holiday_calendar in [holidays.UnitedStates(), holidays.EuropeanCentralBank()]:
            features.append(X[col].apply(lambda datetime: datetime in holiday_calendar))

        return pd.concat(features, axis=1)
