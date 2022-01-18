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
import numpy as np
import pmdarima


def generate_cutoffs(df: pd.DataFrame, horizon: int, unit: str, num_folds: int) -> List[pd.Timestamp]:
    """
    Generate cutoff times for cross validation with the control of number of folds.
    :param df: pd.DataFrame of the historical data
    :param horizon: int number of time into the future for forecasting.
    :param unit: frequency of the timeseries, which must be a pandas offset alias.
    :param num_folds: int number of cutoffs for cross validation.
    :return: list of pd.Timestamp cutoffs for corss-validation.
    """
    period = max(0.5 * horizon, 1)  # avoid empty cutoff buckets
    period = pd.to_timedelta(period, unit=unit)
    horizon = pd.to_timedelta(horizon, unit=unit)

    period_max = 0  # TODO: set period_max properly once different seasonalities are introduced
    seasonality_timedelta = pd.Timedelta(str(period_max) + " days")

    initial = max(3 * horizon, seasonality_timedelta)

    # Last cutoff is "latest date in data - horizon" date
    cutoff = df["ds"].max() - horizon
    if cutoff < df["ds"].min():
        raise ValueError("Less data than horizon.")
    result = [cutoff]
    while result[-1] >= min(df["ds"]) + initial and len(result) < num_folds:
        cutoff -= period
        # If data does not exist in data range (cutoff, cutoff + horizon]
        if not (((df["ds"] > cutoff) & (df["ds"] <= cutoff + horizon)).any()):
            # Next cutoff point is "last date before cutoff in data - horizon"
            if cutoff > df["ds"].min():
                closest_date = df[df["ds"] <= cutoff].max()["ds"]
                cutoff = closest_date - horizon
        # else no data left, leave cutoff as is, it will be dropped.
        result.append(cutoff)
    result = result[:-1]
    if len(result) == 0:
        raise ValueError(
            "Less data than horizon after initial window. Make horizon shorter."
        )
    return list(reversed(result))


def cross_validation(arima_model: pmdarima.arima.ARIMA, df: pd.DataFrame, cutoffs: List[pd.Timestamp]) -> pd.DataFrame:
    """
    Cross-Validation for time series forecasting.

    Computes forecasts from historical cutoff points. The function is a modification of
    prophet.diagnostics.cross_validation that works for ARIMA model.
    :param arima_model: pmdarima.arima.ARIMA object. Fitted ARIMA model.
    :param df: pd.DataFrame of the historical data
    :param cutoffs: list of pd.Timestamp specifying cutoffs to be used during cross validation.
    :return: a pd.DataFrame with the forecast, confidence interval, actual value, and cutoff.
    """
    bins = [df["ds"].min()] + cutoffs + [df["ds"].max()]
    labels = [df["ds"].min()] + cutoffs
    test_df = df[df['ds'] > cutoffs[0]].copy()
    test_df["cutoff"] = pd.to_datetime(pd.cut(test_df["ds"], bins=bins, labels=labels))

    predicts = [single_cutoff_forecast(arima_model, test_df, prev_cutoff, cutoff) for prev_cutoff, cutoff in
                zip(labels, cutoffs)]

    # Update model with data in last cutoff
    last_df = test_df[test_df["cutoff"] == cutoffs[-1]]
    arima_model.update(last_df["y"].values)

    return pd.concat(predicts, axis=0).reset_index(drop=True)


def single_cutoff_forecast(arima_model: pmdarima.arima.ARIMA, test_df: pd.DataFrame, prev_cutoff: pd.Timestamp,
                           cutoff: pd.Timestamp) -> pd.DataFrame:
    """
    Forecast for single cutoff. Used in the cross validation function.
    :param arima_model: pmdarima.arima.ARIMA object. Fitted ARIMA model.
    :param test_df: pd.DataFrame with data to be used for updating model and forecasting.
    :param prev_cutoff: the pd.Timestamp cutoff of the previous forecast.
                        Data between prev_cutoff and cutoff will be used to update the model.
    :param cutoff: pd.Timestamp cutoff of this forecast. The simulated forecast will start from this date.
    :return: a pd.DataFrame with the forecast, confidence interval, actual value, and cutoff.
    """
    # Update the model with data in the previous cutoff
    prev_df = test_df[test_df["cutoff"] == prev_cutoff]
    if not prev_df.empty:
        y_update = prev_df[["ds", "y"]].set_index("ds")
        arima_model.update(y_update)
    # Predict with data in the new cutoff
    new_df = test_df[test_df["cutoff"] == cutoff].copy()
    n_periods = len(new_df["y"].values)
    fc, conf_int = arima_model.predict(n_periods=n_periods, return_conf_int=True)
    fc = fc.tolist()
    conf = np.asarray(conf_int).tolist()

    new_df["yhat"] = fc
    new_df[["yhat_lower", "yhat_upper"]] = conf
    return new_df
