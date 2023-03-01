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
from typing import Dict, List, Optional, Tuple, Union
from databricks.automl_runtime.forecast import DATE_OFFSET_KEYWORD_MAP,\
    QUATERLY_OFFSET_ALIAS, NON_DAILY_OFFSET_ALIAS, OFFSET_ALIAS_MAP, PERIOD_ALIAS_MAP

import pandas as pd

_logger = logging.getLogger(__name__)

def make_future_dataframe(
        start_time: Union[pd.Timestamp, Dict[Tuple, pd.Timestamp]],
        end_time: Union[pd.Timestamp, Dict[Tuple, pd.Timestamp]],
        horizon: int,
        frequency: str,
        include_history: bool = True,
        groups: List[Tuple] = None,
        identity_column_names: List[str] = None,
) -> pd.DataFrame:
    """
    Utility function to generate the dataframe with future timestamps.
    :param start_time: the dictionary of the starting time of each time series in training data.
    :param end_time: the dictionary of the end time of each time series in training data.
    :param horizon: int number of periods to forecast forward.
    :param frequency: the frequency of the time series
    :param include_history:
    :param groups: the collection of group(s) to generate forecast predictions.
    :param identity_column_names: Column names of the identity columns
    :return: pd.DataFrame that extends forward
    """
    if groups is None:
        return make_single_future_dataframe(start_time, end_time, horizon, frequency)

    future_df_list = []
    for group in groups:
        if type(start_time) is dict:
            group_start_time = start_time[group]
        else:
            group_start_time = start_time
        if type(end_time) is dict:
            group_end_time = end_time[group]
        else:
            group_end_time = end_time
        df = make_single_future_dataframe(group_start_time, group_end_time, horizon, frequency, include_history)
        for idx, identity_column_name in enumerate(identity_column_names):
            df[identity_column_name] = group[idx]
        future_df_list.append(df)
    return pd.concat(future_df_list)

def make_single_future_dataframe(
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        horizon: int,
        frequency: str,
        include_history: bool = True,
        column_name: str = "ds"
) -> pd.DataFrame:
    """
    Generate future dataframe for one model
    :param start_time: The starting time of time series of the training data.
    :param end_time: The end time of time series of the training data.
    :param horizon: Int number of periods to forecast forward.
    :param frequency: The frequency of the time series
    :param include_history: Boolean to include the historical dates in the data
            frame for predictions.
    :param column_name: column name of the time column. Default is "ds".
    :return:
    """
    offset_freq = DATE_OFFSET_KEYWORD_MAP[OFFSET_ALIAS_MAP[frequency]]
    unit_offset = pd.DateOffset(**offset_freq)
    end_time = pd.Timestamp(end_time)

    if include_history:
        start_time = start_time
    else:
        start_time = end_time + unit_offset

    date_rng = pd.date_range(
        start=start_time,
        end=end_time + unit_offset*horizon,
        freq=unit_offset
    )
    return pd.DataFrame(date_rng, columns=[column_name])

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
    MIN_HORIZONS = 4  # minimum number of horizons in the dataframe
    horizon_dateoffset = pd.DateOffset(**DATE_OFFSET_KEYWORD_MAP[unit]) * horizon

    try:
        if MIN_HORIZONS * horizon_dateoffset + df["ds"].min() <= df["ds"].max():
            return horizon
    except OverflowError:
        pass

    # In order to calculate the validation horizon, we incrementally add offset
    # to the start time to the quarter of total timedelta. We did this since
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
                start_time: pd.Timestamp,
                end_time: pd.Timestamp, 
                freq:str) -> bool:
    """
    Validate the periods given a start time, end time is consistent with given frequency.
    We consider consistency as only integer frequencies between start and end time, e.g.
    3 days for day, 10 hours for hour, but 2 day and 2 hours are not considered consistency
    for day frequency.
    :param start_time: A pandas timestamp.
    :param end_time: A pandas timestamp.
    :param freq: A string that is accepted by OFFSET_ALIAS_MAP, e.g. 'day',
                'month' etc.
    :return: A boolean indicate whether the time interval is
             evenly divisible by the period.
    """
    periods = calculate_period_differences(start_time, end_time, freq)
    diff = pd.to_datetime(end_time) -  pd.DateOffset(
                **DATE_OFFSET_KEYWORD_MAP[OFFSET_ALIAS_MAP[freq]]
            ) * periods == pd.to_datetime(start_time)
    return diff


def calculate_period_differences(
                start_time: pd.Timestamp,
                end_time: pd.Timestamp, 
                freq:str) -> int:
    """
    Calculate the periods given a start time, end time and period frequency.
    :param start_time: A pandas timestamp.
    :param end_time: A pandas timestamp.
    :param freq: A string that is accepted by OFFSET_ALIAS_MAP, e.g. 'day',
                'month' etc.
    :return: A pd.Series indicates the round-down integer period
             calculated.
    """
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)
    freq_alias = PERIOD_ALIAS_MAP[OFFSET_ALIAS_MAP[freq]]
    return  (end_time.to_period(freq_alias) - start_time.to_period(freq_alias)).n
