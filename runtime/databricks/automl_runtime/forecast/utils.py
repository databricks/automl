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

from typing import List

import pandas as pd


def generate_cutoffs(df: pd.DataFrame, horizon: int, unit: str,
                     seasonal_period: int, num_folds: int) -> List[pd.Timestamp]:
    """
    Generate cutoff times for cross validation with the control of number of folds.
    :param df: pd.DataFrame of the historical data
    :param horizon: int number of time into the future for forecasting.
    :param unit: frequency of the timeseries, which must be a pandas offset alias.
    :param seasonal_period: length of the seasonality period in days
    :param num_folds: int number of cutoffs for cross validation.
    :return: list of pd.Timestamp cutoffs for corss-validation.
    """
    period = max(0.5 * horizon, 1)  # avoid empty cutoff buckets
    period_timedelta = pd.to_timedelta(period, unit=unit)
    horizon_timedelta = pd.to_timedelta(horizon, unit=unit)

    seasonality_timedelta = pd.to_timedelta(seasonal_period, unit=unit)

    initial = max(3 * horizon_timedelta, seasonality_timedelta)

    # Last cutoff is "latest date in data - horizon_timedelta" date
    cutoff = df["ds"].max() - horizon_timedelta
    if cutoff < df["ds"].min():
        raise ValueError("Less data than horizon.")
    result = [cutoff]
    while result[-1] >= min(df["ds"]) + initial and len(result) < num_folds:
        cutoff -= period_timedelta
        # If data does not exist in data range (cutoff, cutoff + horizon_timedelta]
        if not (((df["ds"] > cutoff) & (df["ds"] <= cutoff + horizon_timedelta)).any()):
            # Next cutoff point is "last date before cutoff in data - horizon_timedelta"
            if cutoff > df["ds"].min():
                closest_date = df[df["ds"] <= cutoff].max()["ds"]
                cutoff = closest_date - horizon_timedelta
        # else no data left, leave cutoff as is, it will be dropped.
        result.append(cutoff)
    result = result[:-1]
    if len(result) == 0:
        raise ValueError(
            "Less data than horizon after initial window. Make horizon shorter."
        )
    return list(reversed(result))
