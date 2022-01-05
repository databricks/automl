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

import pickle
from typing import Union

import pandas as pd
import mlflow
import pmdarima

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
