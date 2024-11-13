#
# Copyright (C) 2024 Databricks, Inc.
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

import gluonts
import mlflow
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.torch.model.predictor import PyTorchPredictor
from mlflow.utils.environment import _mlflow_conda_env

from databricks.automl_runtime import version
from databricks.automl_runtime.forecast.model import ForecastModel, mlflow_forecast_log_model
from databricks.automl_runtime.forecast.deepar.utils import set_index_and_fill_missing_time_steps

DEEPAR_ADDITIONAL_PIP_DEPS = [
    f"gluonts[torch]=={gluonts.__version__}",
    f"pandas=={pd.__version__}",
    f"databricks-automl-runtime=={version.__version__}"
]

DEEPAR_CONDA_ENV = _mlflow_conda_env(
    additional_pip_deps=DEEPAR_ADDITIONAL_PIP_DEPS
)


class DeepARModel(ForecastModel):
    """
    DeepAR mlflow model wrapper for forecasting.
    """

    def __init__(self, model: PyTorchPredictor, horizon: int, frequency: str,
                 num_samples: int,
                 target_col: str, time_col: str,
                 id_cols: Optional[List[str]] = None) -> None:
        """
        Initialize the DeepAR mlflow Python model wrapper
        :param model: DeepAR model
        :param horizon: the number of periods to forecast forward
        :param frequency: the frequency of the time series
        :param num_samples: the number of samples to draw from the distribution
        :param target_col: the target column name
        :param time_col: the time column name
        :param id_cols: the column names of the identity columns for multi-series time series; None for single series
        """

        super().__init__()
        self._model = model
        self._horizon = horizon
        self._frequency = frequency
        self._num_samples = num_samples
        self._target_col = target_col
        self._time_col = time_col
        self._id_cols = id_cols

    @property
    def model_env(self):
        return DEEPAR_CONDA_ENV

    def predict(self,
                context: mlflow.pyfunc.model.PythonModelContext,
                model_input: pd.DataFrame) -> pd.DataFrame:
        """
        Predict the future dataframe given the history dataframe
        :param context: A :class:`~PythonModelContext` instance containing artifacts that the model
                        can use to perform inference.
        :param model_input: Input dataframe that contains the history data
        :return: predicted pd.DataFrame that starts after the last timestamp in the input dataframe,
                                and predicts the horizon using the mean of the samples
        """
        required_cols = [self._target_col, self._time_col]
        if self._id_cols:
            required_cols += self._id_cols
        self._validate_cols(model_input, required_cols)

        forecast_sample_list = self.predict_samples(model_input, num_samples=self._num_samples)

        pred_df = pd.concat(
            [
                forecast.mean_ts.rename('yhat').reset_index().assign(item_id=forecast.item_id)
                for forecast in forecast_sample_list
            ],
            ignore_index=True
        )

        pred_df = pred_df.rename(columns={'index': self._time_col})
        if self._id_cols:
            id_col_name = '-'.join(self._id_cols)
            pred_df = pred_df.rename(columns={'item_id': id_col_name})
        else:
            pred_df = pred_df.drop(columns='item_id')

        pred_df[self._time_col] = pred_df[self._time_col].dt.to_timestamp()

        return pred_df

    def predict_samples(self,
                        model_input: pd.DataFrame,
                        num_samples: int = None) -> List[gluonts.model.forecast.SampleForecast]:
        """
        Predict the future samples given the history dataframe
        :param model_input: Input dataframe that contains the history data
        :param num_samples: the number of samples to draw from the distribution
        :return: List of SampleForecast, where each SampleForecast contains num_samples sampled forecasts
        """
        if num_samples is None:
            num_samples = self._num_samples

        model_input_transformed = set_index_and_fill_missing_time_steps(model_input,
                                                                        self._time_col,
                                                                        self._frequency,
                                                                        self._id_cols)

        test_ds = PandasDataset(model_input_transformed, target=self._target_col, freq=self._frequency)

        forecast_iter = self._model.predict(test_ds, num_samples=num_samples)
        forecast_sample_list = list(forecast_iter)

        return forecast_sample_list


def mlflow_deepar_log_model(deepar_model: DeepARModel,
                            sample_input: pd.DataFrame = None) -> None:
    """
    Log the DeepAR model to mlflow
    :param deepar_model: DeepAR mlflow PythonModel wrapper
    :param sample_input: sample input Dataframes for model inference
    """
    mlflow_forecast_log_model(deepar_model, sample_input)
