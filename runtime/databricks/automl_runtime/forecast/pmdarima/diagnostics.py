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

from typing import List, Optional

import pandas as pd
import numpy as np
import pmdarima


def cross_validation(arima_model: pmdarima.arima.ARIMA, df: pd.DataFrame,
                     cutoffs: List[pd.Timestamp], exogenous_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Cross-Validation for time series forecasting.

    Computes forecasts from historical cutoff points. The function is a modification of
    prophet.diagnostics.cross_validation that works for ARIMA model.
    :param arima_model: pmdarima.arima.ARIMA object. Fitted ARIMA model.
    :param df: pd.DataFrame of the historical data
    :param cutoffs: list of pd.Timestamp specifying cutoffs to be used during cross validation.
    :param exogenous_cols: Optional list of column names of exogenous variables. If provided, these columns are
        used as additional features in arima model.
    :return: a pd.DataFrame with the forecast, confidence interval, actual value, and cutoff.
    """
    bins = [df["ds"].min()] + cutoffs + [df["ds"].max()]
    labels = [df["ds"].min()] + cutoffs
    test_df = df[df['ds'] > cutoffs[0]].copy()
    test_df["cutoff"] = pd.to_datetime(pd.cut(test_df["ds"], bins=bins, labels=labels))

    predicts = [single_cutoff_forecast(arima_model, test_df, prev_cutoff, cutoff, exogenous_cols) for prev_cutoff, cutoff in
                zip(labels, cutoffs)]

    # Update model with data in last cutoff
    last_df = test_df[test_df["cutoff"] == cutoffs[-1]]
    last_df.set_index("ds", inplace=True)
    y_update = last_df[["y"]]
    X_update = last_df[exogenous_cols] if exogenous_cols else None
    arima_model.update(
        y_update,
        X=X_update)

    return pd.concat(predicts, axis=0).reset_index(drop=True)


def single_cutoff_forecast(arima_model: pmdarima.arima.ARIMA, test_df: pd.DataFrame, prev_cutoff: pd.Timestamp,
                           cutoff: pd.Timestamp, exogenous_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Forecast for single cutoff. Used in the cross validation function.
    :param arima_model: pmdarima.arima.ARIMA object. Fitted ARIMA model.
    :param test_df: pd.DataFrame with data to be used for updating model and forecasting.
    :param prev_cutoff: the pd.Timestamp cutoff of the previous forecast.
                        Data between prev_cutoff and cutoff will be used to update the model.
    :param cutoff: pd.Timestamp cutoff of this forecast. The simulated forecast will start from this date.
    :param exogenous_cols: Optional list of column names of exogenous variables. If provided, these columns are
        used as additional features in arima model.
    :return: a pd.DataFrame with the forecast, confidence interval, actual value, and cutoff.
    """
    # Update the model with data in the previous cutoff
    prev_df = test_df[test_df["cutoff"] == prev_cutoff]
    if not prev_df.empty:
        prev_df.set_index("ds", inplace=True)
        y_update = prev_df[["y"]]
        X_update = prev_df[exogenous_cols] if exogenous_cols else None
        arima_model.update(
            y_update,
            X=X_update)
    # Predict with data in the new cutoff
    new_df = test_df[test_df["cutoff"] == cutoff].copy()
    X_predict = new_df[exogenous_cols] if exogenous_cols else None
    n_periods = len(new_df["y"].values)
    fc, conf_int = arima_model.predict(
        n_periods=n_periods,
        X=X_predict,
        return_conf_int=True)
    fc = fc.tolist()
    conf = np.asarray(conf_int).tolist()

    new_df["yhat"] = fc
    new_df[["yhat_lower", "yhat_upper"]] = conf
    return new_df
