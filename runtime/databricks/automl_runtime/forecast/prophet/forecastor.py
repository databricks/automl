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
from typing import Any, Dict, Optional

import hyperopt
import numpy as np
import pandas as pd

from databricks.automl_runtime.forecast.prophet.diagnostics import generate_cutoffs


class ContextType(Enum):
    JUPYTER = 1
    DATABRICKS = 2


def _prophet_fit_predict(params: Dict[str, Any], history_pd: pd.DataFrame,
                         horizon: int, frequency: str, num_folds: int,
                         interval_width: int, primary_metric: str, country_holidays: Optional[str] = None):
    """
    Training function for hyperparameter tuning with hyperopt
    :param params: input hyperparameters
    :return:
    """
    import pandas as pd
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics

    model = Prophet(interval_width=interval_width, **params)
    if country_holidays:
        model.add_country_holidays(country_name=country_holidays)
    model.fit(history_pd, iter=200)

    # Evaluate Metrics
    horizon_timedelta = pd.to_timedelta(horizon, unit=frequency)
    cutoffs = generate_cutoffs(model, horizon=horizon_timedelta, num_folds=num_folds)
    # Disable tqdm to make it work with the ipykernel and reduce the output size
    df_cv = cross_validation(model, horizon=horizon_timedelta, cutoffs=cutoffs, disable_tqdm=True)
    df_metrics = performance_metrics(df_cv)

    metrics = df_metrics.mean().drop("horizon").to_dict()

    return {"loss": metrics[primary_metric], "metrics": metrics, "status": hyperopt.STATUS_OK}


class ProphetHyperoptEstimator(ABC):
    """
    Class to do hyper-parameter tunings for prophet with hyperopt
    """
    def __init__(self, horizon: int, frequency_unit: str, metric: str, interval_width: int,
                 country_holidays: str, search_space: Dict[str, Any],
                 algo=hyperopt.tpe.suggest, num_folds: int = 5,
                 max_eval: int = 10, trial_timeout: int = None,
                 random_state: int = 0, is_parallel: bool = True) -> None:
        """
        Initialization
        :param model_json: json string of the Prophet model or
        the dictionary of json strings of Prophet model for multi-series forecasting
        :param horizon: Int number of periods to forecast forward.
        """
        self._horizon = horizon
        self._frequency_unit = frequency_unit
        self._metric = metric
        self._interval_width = interval_width
        self._country_holidays = country_holidays
        self._search_space = search_space
        self._algo = algo
        self._num_folds = num_folds
        self._random_state = np.random.RandomState(random_state)
        self._max_eval = max_eval
        self._timeout = trial_timeout
        self._is_parallel = is_parallel

    def fit(self, df: pd.DataFrame):
        """
        Fit the Prophet model with hyperparameter tunings
        :param df: pd.DataFrame containing the history. Must have columns ds (date
            type) and y
        :param kwargs:
        :return:
        """
        import pandas as pd
        from prophet import Prophet
        from prophet.serialize import model_to_json
        from hyperopt import fmin, Trials, SparkTrials

        seasonality_mode = ["additive", "multiplicative"]
        search_space = self._search_space
        algo = self._algo

        train_fn = partial(_prophet_fit_predict, history_pd=df, horizon=self._horizon,
                           frequency=self._frequency_unit, num_folds=self._num_folds,
                           interval_width=self._interval_width,
                           primary_metric=self._metric, country_holidays=self._country_holidays)

        if self._is_parallel:
            trials = SparkTrials() # pragma: no cover
        else:
            trials = Trials()

        best_result = fmin(
            fn=train_fn,
            space=search_space,
            algo=algo,
            max_evals=self._max_eval,
            trials=trials,
            timeout=self._timeout,
            rstate=self._random_state)

        # Retrain the model with all history data.
        model = Prophet(changepoint_prior_scale=best_result.get("changepoint_prior_scale", 0.05),
                        seasonality_prior_scale=best_result.get("seasonality_prior_scale", 10.0),
                        holidays_prior_scale=best_result.get("holidays_prior_scale", 10.0),
                        seasonality_mode=seasonality_mode[best_result.get("seasonality_mode", 0)],
                        interval_width=self._interval_width)

        if self._country_holidays:
            model.add_country_holidays(country_name=self._country_holidays)

        model.fit(df)

        model_json = model_to_json(model)
        metrics = trials.best_trial["result"]["metrics"]

        results_pd = pd.DataFrame({"model_json": model_json}, index=[0])
        results_pd.reset_index(level=0, inplace=True)
        results_pd["mse"] = metrics["mse"]
        results_pd["rmse"] = metrics["rmse"]
        results_pd["mae"] = metrics["mae"]
        results_pd["mape"] = metrics["mape"]
        results_pd["mdape"] = metrics["mdape"]
        results_pd["smape"] = metrics["smape"]
        results_pd["coverage"] = metrics["coverage"]
        results_pd["prophet_params"] = str(best_result)

        return results_pd