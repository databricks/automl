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
from abc import ABC, abstractmethod
from typing import List

import pandas as pd
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.models.signature import infer_signature, ModelSignature
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE


class ForecastModel(ABC, mlflow.pyfunc.PythonModel):
    @abstractmethod
    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def model_env(self):
        pass  # pragma: no cover

    @staticmethod
    def _validate_cols(df: pd.DataFrame, required_cols: List[str]):
        df_cols = set(df.columns)
        required_cols_set = set(required_cols)
        if not required_cols_set.issubset(df_cols):
            raise MlflowException(
                message=(
                    f"Input data columns '{list(df_cols)}' do not contain the required columns '{required_cols}'"
                ),
                error_code=INVALID_PARAMETER_VALUE,
            )

    def infer_signature(self, sample_input: pd.DataFrame = None) -> ModelSignature:
        signature = infer_signature(sample_input, self.predict(context=None, model_input=sample_input))
        return signature


def mlflow_forecast_log_model(forecast_model: ForecastModel,
                             sample_input: pd.DataFrame = None) -> None:
    """
    Log the model to mlflow
    :param forecast_model: Forecast model wrapper
    :param sample_input: sample input Dataframes for model inference
    """
    # log the model without signature if infer_signature is failed.
    try:
        signature = forecast_model.infer_signature(sample_input)
    except Exception: # noqa
        signature = None
    mlflow.pyfunc.log_model("model", conda_env=forecast_model.model_env,
                            python_model=forecast_model, signature=signature)
