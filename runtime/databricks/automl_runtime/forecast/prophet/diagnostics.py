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
from typing import List
import pandas as pd
import prophet


def generate_cutoffs(model: prophet.forecaster.Prophet, horizon: pd.Timedelta, num_folds: int) -> List[pd.Timestamp]:
    """
    Custom Implementation to Generate cutoff dates for cross validation
    Adding the control of number of folds comparing the method from prophet.
    :param model: Prophet class object. Fitted Prophet model.
    :param horizon: pd.Timedelta forecast horizon.
    :param num_folds: int number of cutoffs for cross validation.
    :return: list of pd.Timestamp cutoffs for corss-validation.
    """
    period = 0.5 * horizon

    period_max = max([s["period"] for s in model.seasonalities.values()]) if model.seasonalities else 0.
    seasonality_dt = pd.Timedelta(str(period_max) + " days")

    initial = max(3 * horizon, seasonality_dt)

    df = model.history.copy().reset_index(drop=True)

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
            "Less data than horizon after initial window. "
            "Make horizon or initial shorter."
        )
    return list(reversed(result))
