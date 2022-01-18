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

import pickle
from typing import Union, Tuple

import pandas as pd
import mlflow
import pmdarima
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.dates import AutoDateLocator, AutoDateFormatter

from databricks.automl_runtime.forecast.pmdarima.model import ArimaModel, MultiSeriesArimaModel

ARIMA_CONDA_ENV = {
    "channels": ["conda-forge"],
    "dependencies": [
        {
            "pip": [
                f"pmdarima=={pmdarima.__version__}",
                f"pickle=={pickle.format_version}",
                f"pandas=={pd.__version__}",
            ]
        }
    ],
    "name": "pmdarima_env",
}


def mlflow_arima_log_model(arima_model: Union[ArimaModel, MultiSeriesArimaModel]) -> None:
    """
    Log the model to mlflow.
    :param arima_model: ARIMA model wrapper
    """
    mlflow.pyfunc.log_model("model", conda_env=ARIMA_CONDA_ENV, python_model=arima_model)


def plot(history_pd: pd.DataFrame, forecast_pd: pd.DataFrame, ax: Axes = None, plot_uncertainty: bool = True,
         xlabel: str = 'ds', ylabel: str = 'y', figsize: Tuple[int, int] = (10, 6), include_legend: bool = False):
    """
    Plot the forecast. Adapted from prophet.plot.plot. See
    https://github.com/facebook/prophet/blob/ba9a5a2c6e2400206017a5ddfd71f5042da9f65b/python/prophet/plot.py#L42.
    :param history_pd: pd.DataFrame of history data.
    :param forecast_pd: pd.DataFrame with forecasts and optionally confidence interval, sorted by time.
    :param ax: Optional matplotlib axes on which to plot.
    :param plot_uncertainty: Optional boolean to plot uncertainty intervals, which will
    only be done if m.uncertainty_samples > 0.
    :param xlabel: Optional label name on X-axis
    :param ylabel: Optional label name on Y-axis
    :param figsize: Optional tuple width, height in inches.
    :param include_legend: Optional boolean to add legend to the plot.
    :return: A matplotlib figure.
    """
    history_pd = history_pd.sort_values(by=["ds"])
    history_pd["ds"] = pd.to_datetime(history_pd["ds"])
    if ax is None:
        fig = plt.figure(facecolor='w', figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()
    fcst_t = forecast_pd['ds'].dt.to_pydatetime()
    ax.plot(history_pd['ds'].dt.to_pydatetime(), history_pd['y'], 'k.', label='Observed data points')
    ax.plot(fcst_t, forecast_pd['yhat'], ls='-', c='#0072B2', label='Forecast')
    if plot_uncertainty and "yhat_lower" in forecast_pd and "yhat_upper" in forecast_pd:
        ax.fill_between(fcst_t, forecast_pd['yhat_lower'], forecast_pd['yhat_upper'],
                        color='#0072B2', alpha=0.2, label='Uncertainty interval')
    # Specify formatting to workaround matplotlib issue #12925
    locator = AutoDateLocator(interval_multiples=False)
    formatter = AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if include_legend:
        ax.legend()
    fig.tight_layout()
    return fig
