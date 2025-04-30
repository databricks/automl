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
        frequency_unit: str,
        frequency_quantity: int,
        include_history: bool = True,
        groups: List[Tuple] = None,
        identity_column_names: List[str] = None,
) -> pd.DataFrame:
    """
    Utility function to generate the dataframe with future timestamps.
    :param start_time: the dictionary of the starting time of each time series in training data.
    :param end_time: the dictionary of the end time of each time series in training data.
    :param horizon: int number of periods to forecast forward.
    :param frequency_unit: the frequency unit of the time series
    :param frequency_quantity: the multiplier for the frequency.
    :param include_history:
    :param groups: the collection of group(s) to generate forecast predictions.
    :param identity_column_names: Column names of the identity columns
    :return: pd.DataFrame that extends forward
    """
    if groups is None:
        return make_single_future_dataframe(start_time, end_time, horizon, frequency_unit, frequency_quantity)

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
        df = make_single_future_dataframe(group_start_time, group_end_time, horizon, frequency_unit, frequency_quantity, include_history)
        for idx, identity_column_name in enumerate(identity_column_names):
            df[identity_column_name] = group[idx]
        future_df_list.append(df)
    return pd.concat(future_df_list)

def make_single_future_dataframe(
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        horizon: int,
        frequency_unit: str,
        frequency_quantity: int,
        include_history: bool = True,
        column_name: str = "ds"
) -> pd.DataFrame:
    """
    Generate future dataframe for one model
    :param start_time: The starting time of time series of the training data.
    :param end_time: The end time of time series of the training data.
    :param horizon: Int number of periods to forecast forward.
    :param frequency_unit: The frequency unit of the time series
    :param frequency_quantity: The frequency quantity of the time series
    :param include_history: Boolean to include the historical dates in the data
            frame for predictions.
    :param column_name: column name of the time column. Default is "ds".
    :return:
    """
    offset_freq = DATE_OFFSET_KEYWORD_MAP[OFFSET_ALIAS_MAP[frequency_unit]]
    timestep_offset = pd.DateOffset(**offset_freq) * frequency_quantity
    end_time = pd.Timestamp(end_time)

    if include_history:
        start_time = start_time
    else:
        start_time = end_time + timestep_offset

    date_rng = pd.date_range(
        start=start_time,
        end=end_time + timestep_offset * horizon,
        freq=timestep_offset
    )
    return pd.DataFrame(date_rng, columns=[column_name])

def get_validation_horizon(df: pd.DataFrame, horizon: int, frequency_unit: str, frequency_quantity: int = 1) -> int:
    """
    Return validation_horizon, which is the lesser of `horizon` and one quarter of the dataframe's timedelta
    Since the seasonality period is never more than half of the dataframe's timedelta,
    there is no case where seasonality would affect the validation horizon. (This is prophet's default seasonality
    behavior, and we enforce it for ARIMA.)
    :param df: pd.DataFrame of the historical data
    :param horizon: int number of time into the future for forecasting
    :param frequency_unit: frequency unit of the time series, which must be a pandas offset alias
    :param frequency_quantity: int multiplier for the frequency unit, representing the number of `unit`s 
        per time step in the dataframe. This is useful when the time series has a granularity that 
        spans multiple `unit`s (e.g., if `unit='min'` and `frequency_quantity=5`, it means the data 
        follows a five-minute pattern). To make it backward compatible, defaults to 1.
    :return: horizon used for validation, in terms of the input `unit`
    """
    MIN_HORIZONS = 4  # minimum number of horizons in the dataframe
    horizon_dateoffset = pd.DateOffset(**DATE_OFFSET_KEYWORD_MAP[frequency_unit]) * horizon * frequency_quantity

    try:
        if MIN_HORIZONS * horizon_dateoffset + df["ds"].min() <= df["ds"].max():
            return horizon
    except OverflowError:
        pass

    # In order to calculate the validation horizon, we incrementally add offset
    # to the start time to the quarter of total timedelta. We did this since
    # pd.DateOffset does not support divide by operation.
    timestep_dateoffset = pd.DateOffset(**DATE_OFFSET_KEYWORD_MAP[frequency_unit]) * frequency_quantity
    max_horizon = 0
    cur_timestamp = df["ds"].min()
    while cur_timestamp + timestep_dateoffset <= df["ds"].max():
        cur_timestamp += timestep_dateoffset
        max_horizon += 1
    _logger.info(f"Horizon {horizon_dateoffset} too long relative to dataframe's "
    f"timedelta. Validation horizon will be reduced to {max_horizon//MIN_HORIZONS*timestep_dateoffset}.")
    return max_horizon // MIN_HORIZONS

def generate_cutoffs(df: pd.DataFrame, horizon: int, frequency_unit: str,
                     num_folds: int, seasonal_period: int = 0, 
                     seasonal_unit: Optional[str] = None,
                     frequency_quantity: int = 1) -> List[pd.Timestamp]:
    """
    Generate cutoff times for cross validation with the control of number of folds.
    :param df: pd.DataFrame of the historical data.
    :param horizon: int number of time into the future for forecasting.
    :param frequency_unit: frequency unit of the time series, which must be a pandas offset alias.
    :param num_folds: int number of cutoffs for cross validation.
    :param seasonal_period: length of the seasonality period.
    :param seasonal_unit: Optional frequency unit for the seasonal period. If not specified, the function will use
                          the same frequency unit as the time series.
    :param frequency_quantity: frequency quantity of the time series.
    :return: list of pd.Timestamp cutoffs for cross-validation.
    """
    period = max(0.5 * horizon, 1)  # avoid empty cutoff buckets

    # avoid non-integer months, quaters ands years.
    if frequency_unit in NON_DAILY_OFFSET_ALIAS:
        period = int(period)
        period_dateoffset = pd.DateOffset(**DATE_OFFSET_KEYWORD_MAP[frequency_unit])*frequency_quantity*period
    else:
        offset_kwarg = {list(DATE_OFFSET_KEYWORD_MAP[frequency_unit])[0]: period}
        period_dateoffset = pd.DateOffset(**offset_kwarg) * frequency_quantity

    horizon_dateoffset = pd.DateOffset(**DATE_OFFSET_KEYWORD_MAP[frequency_unit])*frequency_quantity*horizon

    if not seasonal_unit:
        seasonal_unit = frequency_unit

    seasonality_dateoffset = pd.DateOffset(**DATE_OFFSET_KEYWORD_MAP[frequency_unit])*frequency_quantity*seasonal_period

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

def generate_custom_cutoffs(df: pd.DataFrame, horizon: int, frequency_unit: str,
                     split_cutoff: pd.Timestamp, frequency_quantity: int = 1) -> List[pd.Timestamp]:
    """
    Generate custom cutoff times for cross validation based on user-specified split cutoff.
    Period (step size) is 1.
    :param df: pd.DataFrame of the historical data.
    :param horizon: int number of time into the future for forecasting.
    :param frequency_unit: frequency unit of the time series, which must be a pandas offset alias.
    :param split_cutoff: the user-specified cutoff, as the starting point of cutoffs.
    :param frequency_quantity: frequency quantity of the time series.
    For tuning job, it is the cutoff between train and validate split.
    For training job, it is the cutoff bewteen validate and test split.
    :return: list of pd.Timestamp cutoffs for cross-validation.
    """
    # TODO: [ML-43528] expose period as input.
    period = 1 
    period_dateoffset = pd.DateOffset(**DATE_OFFSET_KEYWORD_MAP[frequency_unit])*period*frequency_quantity
    horizon_dateoffset = pd.DateOffset(**DATE_OFFSET_KEYWORD_MAP[frequency_unit])*horizon*frequency_quantity

    # First cutoff is the cutoff bewteen splits
    cutoff = split_cutoff
    result = []
    max_cutoff = max(df["ds"]) - horizon_dateoffset
    while cutoff <= max_cutoff:
        # If data does not exist in data range (cutoff, cutoff + horizon_dateoffset]
        if (not (((df["ds"] > cutoff) & (df["ds"] <= cutoff + horizon_dateoffset)).any())):
            # Next cutoff point is "next date after cutoff in data - horizon_dateoffset"
            closest_date = df[df["ds"] > cutoff].min()["ds"]
            cutoff = closest_date - horizon_dateoffset
        result.append(cutoff)
        cutoff += period_dateoffset
    return result

def is_quaterly_alias(freq: str):
    return freq in QUATERLY_OFFSET_ALIAS

def is_frequency_consistency(
                start_time: pd.Timestamp,
                end_time: pd.Timestamp, 
                frequency_unit:str,
                frequency_quantity: int) -> bool:
    """
    Validate the periods given a start time, end time is consistent with given frequency.
    We consider consistency as only integer frequencies between start and end time, e.g.
    3 days for day, 10 hours for hour, but 2 day and 2 hours are not considered consistency
    for day frequency.
    :param start_time: A pandas timestamp.
    :param end_time: A pandas timestamp.
    :param freq: A string that is accepted by OFFSET_ALIAS_MAP, e.g. 'day',
                'month' etc.
    :param frequency_quantity: the multiplier for the frequency.
    :return: A boolean indicate whether the time interval is
             evenly divisible by the period.
    """
    periods = calculate_period_differences(start_time, end_time, frequency_unit, frequency_quantity)
    # If the difference between start and end time is divisible by the period time
    diff = (pd.to_datetime(end_time) -  pd.DateOffset(
                **DATE_OFFSET_KEYWORD_MAP[OFFSET_ALIAS_MAP[frequency_unit]]
            ) * periods * frequency_quantity) == pd.to_datetime(start_time)
    return diff


def calculate_period_differences(
                start_time: pd.Timestamp,
                end_time: pd.Timestamp, 
                frequency_unit:str,
                frequency_quantity: int) -> int:
    """
    Calculate the periods given a start time, end time and period frequency.
    :param start_time: A pandas timestamp.
    :param end_time: A pandas timestamp.
    :param freq: A string that is accepted by OFFSET_ALIAS_MAP, e.g. 'day',
                'month' etc.
    :param frequency_quantity: An integer that is the multiplier for the frequency.
    :return: A pd.Series indicates the round-down integer period
             calculated.
    """
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)
    freq_alias = PERIOD_ALIAS_MAP[OFFSET_ALIAS_MAP[frequency_unit]]
    # It is intended to get the floor value. And in the later check we will use this floor value to find out if it is not consistent.
    return  (end_time.to_period(freq_alias) - start_time.to_period(freq_alias)).n // frequency_quantity

def apply_preprocess_func(df: pd.DataFrame, preprocess_func: callable, split_col: str) -> pd.DataFrame:
    """
    Apply the preprocessing function to the dataframe. The preprocessing function requires the "y" column
    and the split column to be present, as they are used in the trial notebook. These columns are added
    temporarily and removed after preprocessing.
    see https://src.dev.databricks.com/databricks-eng/universe/-/blob/automl/python/databricks/automl/core/sections/templates/preprocess/finish_with_transform.jinja?L3
    and https://src.dev.databricks.com/databricks-eng/universe/-/blob/automl/python/databricks/automl/core/sections/templates/preprocess/select_columns.jinja?L8-10
    :param df: pd.DataFrame to be preprocessed.
    :param preprocess_func: preprocessing function to be applied to the dataframe.
    :param split_col: name of the split column to be added to the dataframe.
    :return: preprocessed pd.DataFrame.
    """
    df["y"] = None
    df[split_col] = "prediction"
    df = preprocess_func(df)
    df.drop(columns=["y", split_col], inplace=True, errors="ignore")
    return df
