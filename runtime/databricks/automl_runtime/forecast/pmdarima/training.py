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
import pickle
import numpy as np
import pmdarima as pm
from pmdarima.arima import StepwiseContext
from prophet.diagnostics import performance_metrics

from databricks.automl_runtime.forecast.pmdarima.diagnostics import generate_cutoffs, cross_validation
from databricks.automl_runtime.forecast import OFFSET_ALIAS_MAP


class ArimaEstimator:
    """
    ARIMA estimator using pmdarima.auto_arima.
    """

    def __init__(self, horizon: int, frequency_unit: str, metric: str, seasonal_periods: List[int],
                 num_folds: int = 20, max_steps: int = 150) -> None:
        """
        :param horizon: Number of periods to forecast forward
        :param frequency_unit: Frequency of the time series
        :param metric: Metric that will be optimized across trials
        :param seasonal_periods: A list of seasonal periods for tuning.
        :param num_folds: Number of folds for cross validation
        :param max_steps: Max steps for stepwise auto_arima
        """
        self._horizon = horizon
        self._frequency_unit = OFFSET_ALIAS_MAP[frequency_unit]
        self._metric = metric
        self._seasonal_periods = seasonal_periods
        self._num_folds = num_folds
        self._max_steps = max_steps

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the ARIMA model with tuning of seasonal period m and with pmdarima.auto_arima.
        :param df: A pd.DataFrame containing the history data. Must have columns ds and y.
        :return: A pd.DataFrame with the best model (pickled) and its metrics from cross validation.
        """
        history_pd = df.sort_values(by=["ds"]).reset_index(drop=True)
        history_pd["ds"] = pd.to_datetime(history_pd["ds"])

        # Impute missing time steps
        history_pd = self._fill_missing_time_steps(history_pd, self._frequency_unit)

        # Generate cutoffs for cross validation
        cutoffs = generate_cutoffs(history_pd, horizon=self._horizon, unit=self._frequency_unit,
                                   num_folds=self._num_folds)

        # Tune seasonal periods
        best_result = None
        best_metric = float("inf")
        for m in self._seasonal_periods:
            result = self._fit_predict(history_pd, cutoffs, m, self._max_steps)
            metric = result["metrics"]["smape"]
            if metric < best_metric:
                best_result = result
                best_metric = metric

        results_pd = pd.DataFrame(best_result["metrics"], index=[0])
        results_pd["pickled_model"] = pickle.dumps(best_result["model"])

        return results_pd

    @staticmethod
    def _fit_predict(df: pd.DataFrame, cutoffs: List[pd.Timestamp], seasonal_period: int, max_steps: int = 150):
        train_df = df[df['ds'] <= cutoffs[0]]
        y_train = train_df[["ds", "y"]].set_index("ds")

        # Train with the initial interval
        with StepwiseContext(max_steps=max_steps):
            arima_model = pm.auto_arima(
                y=y_train,
                m=seasonal_period,
                stepwise=True,
            )

        # Evaluate with cross validation
        df_cv = cross_validation(arima_model, df, cutoffs)
        df_metrics = performance_metrics(df_cv)
        metrics = df_metrics.drop("horizon", axis=1).mean().to_dict()
        # performance_metrics doesn't calculate mape if any y is close to 0
        if "mape" not in metrics:
            metrics["mape"] = np.nan

        return {"metrics": metrics, "model": arima_model}

    @staticmethod
    def _fill_missing_time_steps(df: pd.DataFrame, frequency: str):
        # Forward fill missing time steps
        return df.set_index("ds").resample(rule=OFFSET_ALIAS_MAP[frequency]).pad().reset_index()
