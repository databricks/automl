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
from abc import ABC
from enum import Enum
from functools import partial
from typing import Any, Dict, List, Optional

import hyperopt
import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import performance_metrics
from prophet.serialize import model_to_json
from hyperopt import fmin, Trials, SparkTrials

from databricks.automl_runtime.forecast.prophet.diagnostics import cross_validation
from databricks.automl_runtime.forecast import utils, OFFSET_ALIAS_MAP, DATE_OFFSET_KEYWORD_MAP


class ProphetHyperParams(Enum):
    CHANGEPOINT_PRIOR_SCALE = "changepoint_prior_scale"
    SEASONALITY_PRIOR_SCALE = "seasonality_prior_scale"
    HOLIDAYS_PRIOR_SCALE = "holidays_prior_scale"
    SEASONALITY_MODE = "seasonality_mode"


def _prophet_fit_predict(params: Dict[str, Any], history_pd: pd.DataFrame,
                         horizon: int, frequency: str, cutoffs: List[pd.Timestamp],
                         interval_width: int, primary_metric: str,
                         country_holidays: Optional[str] = None,
                         regressors = None, **prophet_kwargs) -> Dict[str, Any]:
    """
    Training function for hyperparameter tuning with hyperopt

    :param params: Input hyperparameters
    :param history_pd: pd.DataFrame containing the history. Must have columns ds (date
            type) and y, the time series
    :param horizon: Forecast horizon_timedelta
    :param frequency: Frequency of the time series
    :param num_folds: Number of folds for cross validation
    :param interval_width: Width of the uncertainty intervals provided for the forecast
    :param primary_metric: Metric that will be optimized across trials
    :param country_holidays: Built-in holidays for the specified country
    :param prophet_kwargs: Optional keyword arguments for Prophet model.
    :return: Dictionary as the format for hyperopt
    """
    input_params = {**params, **prophet_kwargs}
    model = Prophet(interval_width=interval_width, **input_params)
    if country_holidays:
        model.add_country_holidays(country_name=country_holidays)

    if regressors:
        for regressor in regressors:
            model.add_regressor(regressor)

    model.fit(history_pd, iter=200)
    offset_kwarg = DATE_OFFSET_KEYWORD_MAP[OFFSET_ALIAS_MAP[frequency]]
    horizon_offset = pd.DateOffset(**offset_kwarg)*horizon
    # Evaluate Metrics
    df_cv = cross_validation(
        model, horizon=horizon_offset, cutoffs=cutoffs, disable_tqdm=True
    )  # disable tqdm to make it work with ipykernel and reduce the output size
    df_metrics = performance_metrics(df_cv)

    metrics = df_metrics.mean().drop("horizon").to_dict()

    return {"loss": metrics[primary_metric], "metrics": metrics, "status": hyperopt.STATUS_OK}


class ProphetHyperoptEstimator(ABC):
    """
    Class to do hyper-parameter tunings for prophet with hyperopt
    """
    SUPPORTED_METRICS = ["mse", "rmse", "mae", "mape", "mdape", "smape", "coverage"]

    def __init__(self, horizon: int, frequency_unit: str, metric: str, interval_width: int,
                 country_holidays: str, search_space: Dict[str, Any],
                 algo=hyperopt.tpe.suggest, num_folds: int = 5,
                 max_eval: int = 10, trial_timeout: int = None,
                 random_state: int = 0, is_parallel: bool = True,
                 regressors = None, **prophet_kwargs) -> None:
        """
        Initialization

        :param horizon: Number of periods to forecast forward
        :param frequency_unit: Frequency of the time series
        :param metric: Metric that will be optimized across trials
        :param interval_width: Width of the uncertainty intervals provided for the forecast
        :param country_holidays: Built-in holidays for the specified country
        :param search_space: Search space for hyperparameter tuning with hyperopt
        :param algo: Search algorithm
        :param num_folds: Number of folds for cross validation
        :param max_eval: Max number of trials generated in hyperopt
        :param trial_timeout: timeout for hyperopt
        :param random_state: random seed for hyperopt
        :param is_parallel: Indicators to decide that whether run hyperopt in 
        :param regressors: list of column names of external regressors
        :param prophet_kwargs: Optional keyword arguments for Prophet model.
            For information about the parameters see:
            `The Prophet source code <https://github.com/facebook/prophet/blob/master/python/prophet/forecaster.py>`_.
        """
        self._horizon = horizon
        self._frequency_unit = OFFSET_ALIAS_MAP[frequency_unit]
        self._metric = metric
        self._interval_width = interval_width
        self._country_holidays = country_holidays
        self._search_space = search_space
        self._algo = algo
        self._num_folds = num_folds
        self._random_state = np.random.default_rng(random_state)
        self._max_eval = max_eval
        self._timeout = trial_timeout
        self._is_parallel = is_parallel
        self._regressors = regressors
        self._prophet_kwargs = prophet_kwargs

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the Prophet model with hyperparameter tunings
        :param df: pd.DataFrame containing the history. Must have columns ds (date
            type) and y
        :return: DataFrame with model json and metrics in cross validation
        """
        df["ds"] = pd.to_datetime(df["ds"])

        seasonality_mode = ["additive", "multiplicative"]

        validation_horizon = utils.get_validation_horizon(df, self._horizon, self._frequency_unit)
        cutoffs = utils.generate_cutoffs(
            df.reset_index(drop=True),
            horizon=validation_horizon,
            unit=self._frequency_unit,
            num_folds=self._num_folds,
        )

        train_fn = partial(_prophet_fit_predict, history_pd=df, horizon=validation_horizon,
                           frequency=self._frequency_unit, cutoffs=cutoffs,
                           interval_width=self._interval_width,
                           primary_metric=self._metric, country_holidays=self._country_holidays,
                           regressors=self._regressors, **self._prophet_kwargs)

        if self._is_parallel:
            trials = SparkTrials() # pragma: no cover
        else:
            trials = Trials()

        best_result = fmin(
            fn=train_fn,
            space=self._search_space,
            algo=self._algo,
            max_evals=self._max_eval,
            trials=trials,
            timeout=self._timeout,
            rstate=self._random_state)

        # Retrain the model with all history data.
        model = Prophet(changepoint_prior_scale=best_result.get(ProphetHyperParams.CHANGEPOINT_PRIOR_SCALE.value, 0.05),
                        seasonality_prior_scale=best_result.get(ProphetHyperParams.SEASONALITY_PRIOR_SCALE.value, 10.0),
                        holidays_prior_scale=best_result.get(ProphetHyperParams.HOLIDAYS_PRIOR_SCALE.value, 10.0),
                        seasonality_mode=seasonality_mode[best_result.get(ProphetHyperParams.SEASONALITY_MODE.value, 0)],
                        interval_width=self._interval_width,
                        **self._prophet_kwargs)

        if self._country_holidays:
            model.add_country_holidays(country_name=self._country_holidays)
        
        if self._regressors:
            for regressor in self._regressors:
                model.add_regressor(regressor)

        model.fit(df)

        model_json = model_to_json(model)
        metrics = trials.best_trial["result"]["metrics"]

        results_pd = pd.DataFrame({"model_json": model_json}, index=[0])
        results_pd.reset_index(level=0, inplace=True)
        for metric in self.SUPPORTED_METRICS:
            if metric in metrics.keys():
                results_pd[metric] = metrics[metric]
            else:
                results_pd[metric] = np.nan
        results_pd["prophet_params"] = str(best_result)

        return results_pd
