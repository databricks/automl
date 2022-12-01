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
import logging
from typing import List, Optional, Tuple, Union
from databricks.automl_runtime.forecast import DATE_OFFSET_KEYWORD_MAP,\
    QUATERLY_OFFSET_ALIAS, NON_DAILY_OFFSET_ALIAS, OFFSET_ALIAS_MAP, PERIOD_ALIAS_MAP

import pandas as pd

_logger = logging.getLogger(__name__)

def get_validation_horizon(df: pd.DataFrame, horizon: int, unit: str) -> int:
    """
    Return validation_horizon, which is the lesser of `horizon` and one quarter of the dataframe's timedelta
    Since the seasonality period is never more than half of the dataframe's timedelta,
    there is no case where seasonality would affect the validation horizon. (This is prophet's default seasonality
    behavior, and we enforce it for ARIMA.)
    :param df: pd.DataFrame of the historical data
    :param horizon: int number of time into the future for forecasting
    :param unit: frequency unit of the time series, which must be a pandas offset alias
    :return: horizon used for validation, in terms of the input `unit`
    """
    MIN_HORIZONS = 4 # minimum number of horizons in the datafram
    horizon_dateoffset = pd.DateOffset(**DATE_OFFSET_KEYWORD_MAP[unit])* horizon

    if MIN_HORIZONS * horizon_dateoffset + df["ds"].min() <= df["ds"].max():
        return horizon
    else:
        # In order to calculate the validation horizon, we incrementally add offset
        # to the start time to the quater of total timedelta. We did this since
        # pd.DateOffset does not support divide by operation.
        unit_dateoffset = pd.DateOffset(**DATE_OFFSET_KEYWORD_MAP[unit])
        max_horizon = 0
        cur_timestamp = df["ds"].min()
        while cur_timestamp + unit_dateoffset <= df["ds"].max():
            cur_timestamp += unit_dateoffset
            max_horizon += 1
        _logger.info(f"Horizon {horizon_dateoffset} too long relative to dataframe's "
        f"timedelta. Validation horizon will be reduced to {max_horizon//MIN_HORIZONS*unit_dateoffset}.")
        return max_horizon // MIN_HORIZONS

def generate_cutoffs(df: pd.DataFrame, horizon: int, unit: str,
                     num_folds: int, seasonal_period: int = 0, 
                     seasonal_unit: Optional[str] = None) -> List[pd.Timestamp]:
    """
    Generate cutoff times for cross validation with the control of number of folds.
    :param df: pd.DataFrame of the historical data.
    :param horizon: int number of time into the future for forecasting.
    :param unit: frequency unit of the time series, which must be a pandas offset alias.
    :param num_folds: int number of cutoffs for cross validation.
    :param seasonal_period: length of the seasonality period.
    :param seasonal_unit: Optional frequency unit for the seasonal period. If not specified, the function will use
                          the same frequency unit as the time series.
    :return: list of pd.Timestamp cutoffs for cross-validation.
    """
    period = max(0.5 * horizon, 1)  # avoid empty cutoff buckets

    # avoid non-integer months, quaters ands years.
    if unit in NON_DAILY_OFFSET_ALIAS:
        period = int(period)
        period_dateoffset = pd.DateOffset(**DATE_OFFSET_KEYWORD_MAP[unit])*period
    else:
        offset_kwarg = {list(DATE_OFFSET_KEYWORD_MAP[unit])[0]: period}
        period_dateoffset = pd.DateOffset(**offset_kwarg)

    horizon_dateoffset = pd.DateOffset(**DATE_OFFSET_KEYWORD_MAP[unit])*horizon

    if not seasonal_unit:
        seasonal_unit = unit

    seasonality_dateoffset = pd.DateOffset(**DATE_OFFSET_KEYWORD_MAP[unit])*seasonal_period

    # We can not compare DateOffset directly, so we add to start time and compare.
    initial = seasonality_dateoffset
    if df["ds"].min() + 3 * horizon_dateoffset > df["ds"].min() + seasonality_dateoffset:
        initial = 3 * horizon_dateoffset

    # Last cutoff is "latest date in data - horizon_dateoffset" date
    cutoff = df["ds"].max() - horizon_dateoffset
    if cutoff < df["ds"].min():
        raise ValueError("Less data than horizon.")
    result = [cutoff]
    while result[-1] >= min(df["ds"]) + initial and len(result) <= num_folds:
        cutoff -= period_dateoffset
        # If data does not exist in data range (cutoff, cutoff + horizon_dateoffset]
        if not (((df["ds"] > cutoff) & (df["ds"] <= cutoff + horizon_dateoffset)).any()):
            # Next cutoff point is "last date before cutoff in data - horizon_dateoffset"
            if cutoff > df["ds"].min():
                closest_date = df[df["ds"] <= cutoff].max()["ds"]
                cutoff = closest_date - horizon_dateoffset
        # else no data left, leave cutoff as is, it will be dropped.
        result.append(cutoff)
    result = result[:-1]
    if len(result) == 0:
        raise ValueError(
            "Less data than horizon after initial window. Make horizon shorter."
        )
    return list(reversed(result))

def is_quaterly_alias(freq: str):
    return freq in QUATERLY_OFFSET_ALIAS

def is_frequency_consistency(
                start_time: Union[pd.Series, pd.Timestamp],
                end_time: Union[pd.Series, pd.Timestamp], 
                freq:str) -> bool:
    """
    Validate the periods given a start time, end time is consistent with given frequency.
    :param start_time: A pd series convertable to datetime
    :param end_time: A pd series convertable to datetime, must be in same size
                as start_time.
    :param freq: A string that is accepted by OFFSET_ALIAS_MAP, e.g. 'day',
                'month' etc.
    :return: A boolean indicate whether the time interval is
             evenly divisible by the period.
    """
    periods = calculate_periods(start_time, end_time, freq)
    diff = pd.to_datetime(end_time) -  pd.DateOffset(
                **DATE_OFFSET_KEYWORD_MAP[OFFSET_ALIAS_MAP[freq]]
            ) * periods == pd.to_datetime(start_time)
    if type(diff) is bool:
        return diff
    return diff.all()


def calculate_periods(
                start_time: Union[pd.Series, pd.Timestamp],
                end_time: Union[pd.Series, pd.Timestamp], 
                freq:str) -> pd.Series:
    """
    Calculate the periods given a start time, end time and period frequency.
    :param start_time: A pd series convertable to datetime
    :param end_time: A pd series convertable to datetime, must be in same size
                as start_time.
    :param freq: A string that is accepted by OFFSET_ALIAS_MAP, e.g. 'day',
                'month' etc.
    :return: A pd.Series indicates the round-down integer period
             calculated.
    """
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)
    freq_alias = PERIOD_ALIAS_MAP[OFFSET_ALIAS_MAP[freq]]
    if type(start_time) is pd.Timestamp:
        start_time = start_time.to_period(freq_alias)
    else:
        start_time = start_time.dt.to_period(freq_alias)
    if type(end_time) is pd.Timestamp:
        end_time = end_time.to_period(freq_alias)
    else:
        end_time = end_time.dt.to_period(freq_alias)
    diff = end_time - start_time
    if type(diff) is pd.Series:
        return diff.apply(lambda x: x.n)
    else:
        return diff.n
