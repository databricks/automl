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
import traceback
from typing import List

import pandas as pd
import pickle
import numpy as np
import pmdarima as pm
from pmdarima.arima import StepwiseContext
from prophet.diagnostics import performance_metrics

from databricks.automl_runtime.forecast.pmdarima.diagnostics import cross_validation
from databricks.automl_runtime.forecast import utils, OFFSET_ALIAS_MAP, DATE_OFFSET_KEYWORD_MAP

_logger = logging.getLogger(__name__)

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
        :param seasonal_periods: A list of seasonal periods for tuning. Units are frequency_unit.
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

        # Check if the time has consistent frequency
        self._validate_ds_freq(history_pd, self._frequency_unit)

        # Impute missing time steps
        history_pd = self._fill_missing_time_steps(history_pd, self._frequency_unit)

        history_periods = utils.calculate_period_differences(
            history_pd['ds'].min(), history_pd['ds'].max(), self._frequency_unit
        )

        # Tune seasonal periods
        best_result = None
        best_metric = float("inf")
        for m in self._seasonal_periods:
            try:
                # this check mirrors the the default behavior by prophet
                if history_periods < 2 * m:
                    _logger.warning(f"Skipping seasonal_period={m} ({self._frequency_unit}). Dataframe timestamps must span at least two seasonality periods, but only spans {history_periods} {self._frequency_unit}""")
                    continue
                # Prophet also rejects the seasonality periods if the seasonality period timedelta is less than the shortest timedelta in the dataframe.
                # However, this cannot happen in ARIMA because _fill_missing_time_steps imputes values for each _frequency_unit,
                # so the minimum valid seasonality period is always 1

                validation_horizon = utils.get_validation_horizon(history_pd, self._horizon, self._frequency_unit)
                cutoffs = utils.generate_cutoffs(
                    history_pd,
                    horizon=validation_horizon,
                    unit=self._frequency_unit,
                    num_folds=self._num_folds,
                )

                result = self._fit_predict(history_pd, cutoffs=cutoffs, seasonal_period=m, max_steps=self._max_steps)
                metric = result["metrics"]["smape"]
                if metric < best_metric:
                    best_result = result
                    best_metric = metric
            except Exception as e:
                _logger.warning(f"Encountered an exception with seasonal_period={m}: {repr(e)}")
                traceback.print_exc()
        if not best_result:
            raise Exception("No model is successfully trained.")

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
        # NOTE: the right closed meanning that the time data after resample
        # will be right-shifted to the closest frequency, e.g. 2020-01-01 03:00:00
        # will be shifted to 2020-01-02 00:00:00, thus we can perform a forward
        # fill to fill the NaN since all the resampled time are after the original
        # time. 
        # Reference: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.resample.html
        df_filled = df.set_index("ds")
        # NOTE: We have to normalize the index before resampling since pandas will
        # normalize the week month quarter and year data during resample causing
        # the closed right not work as expected.
        # TODO(ML-27585): Improve this implementation/file a PR to pandas.
        if OFFSET_ALIAS_MAP[frequency] in ['W', 'MS', 'QS', 'YS']:
            df_filled.index = df_filled.index.normalize()
        df_filled = df_filled.resample(
                rule=OFFSET_ALIAS_MAP[frequency], closed='right'
            ).pad().reset_index()
        # We want to re-align the resampled time to the original start time.
        # However, since all of our supported monthly/quarterly/annualy data
        # are counted from the begining of month/quarter/year, so we need to
        # first shift back one frequency unit to correctly caluclate the offset. 
        df_filled["ds"] = df_filled["ds"] - pd.DateOffset(
            **DATE_OFFSET_KEYWORD_MAP[OFFSET_ALIAS_MAP[frequency]]
        )
        start_ds, modified_start_ds = df["ds"].min(), df_filled["ds"].min()
        if start_ds != modified_start_ds:
            offset = modified_start_ds - start_ds
            df_filled["ds"] = df_filled["ds"] - offset
        return df_filled

    @staticmethod
    def _validate_ds_freq(df: pd.DataFrame, frequency: str):
        start_ds = df["ds"].min()
        consistency = df["ds"].apply(lambda x:
            utils.is_frequency_consistency(start_ds, x, frequency)
        ).all()
        if not consistency:
            raise ValueError(
                f"Input time column includes different frequency than the specified frequency {frequency}."
            )
