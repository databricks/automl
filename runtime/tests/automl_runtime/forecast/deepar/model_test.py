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

import unittest

import mlflow
import pandas as pd
import torch
import torch.nn as nn
from gluonts.dataset.field_names import FieldName
from gluonts.transform import InstanceSplitter, TestSplitSampler
from gluonts.torch.model.predictor import PyTorchPredictor

from databricks.automl_runtime.forecast.deepar.model import (
    DeepARModel, 
    mlflow_deepar_log_model, 
    DEEPAR_ADDITIONAL_PIP_DEPS
)


class TestDeepARModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Adapted from https://github.com/awslabs/gluonts/blob/dev/test/torch/model/test_torch_predictor.py
        class RandomNetwork(nn.Module):
            def __init__(
                    self,
                    prediction_length: int,
                    context_length: int,
            ) -> None:
                super().__init__()
                self.prediction_length = prediction_length
                self.context_length = context_length
                self.net = nn.Linear(context_length, prediction_length)
                torch.nn.init.uniform_(self.net.weight, -1.0, 1.0)

            def forward(self, past_target):
                out = self.net(past_target.float())
                return out.unsqueeze(1)

        cls.context_length = 5
        cls.prediction_length = 5

        cls.pred_net = RandomNetwork(
            prediction_length=cls.context_length, context_length=cls.context_length
        )

        cls.transformation = InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=TestSplitSampler(),
            past_length=cls.context_length,
            future_length=cls.prediction_length,
        )

        cls.model = PyTorchPredictor(
            prediction_length=cls.prediction_length,
            input_names=["past_target"],
            prediction_net=cls.pred_net,
            batch_size=16,
            input_transform=cls.transformation,
            device="cpu",
        )

    def test_model_save_and_load_single_series(self):
        target_col = "sales"
        time_col = "date"

        deepar_model = DeepARModel(
            model=self.model,
            horizon=self.prediction_length,
            frequency="d",
            num_samples=1,
            target_col=target_col,
            time_col=time_col,
        )

        num_rows = 10
        sample_input = pd.concat(
            [
                pd.to_datetime(
                    pd.Series(range(num_rows), name=time_col).apply(
                        lambda i: f"2020-10-{3 * i + 1}"
                    )
                ),
                pd.Series(range(num_rows), name=target_col),
            ],
            axis=1,
        )

        with mlflow.start_run() as run:
            mlflow_deepar_log_model(deepar_model, sample_input)

        run_id = run.info.run_id

        # check if all additional dependencies are logged
        self._check_requirements(run_id)

        # load the model and predict
        loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
        pred_df = loaded_model.predict(sample_input)

        assert pred_df.columns.tolist() == [time_col, "yhat"]
        assert len(pred_df) == self.prediction_length
        assert pred_df[time_col].min() > sample_input[time_col].max()

    def test_model_save_and_load_multi_series(self):
        target_col = "sales"
        time_col = "date"
        id_col = "store"

        deepar_model = DeepARModel(
            model=self.model,
            horizon=self.prediction_length,
            num_samples=1,
            frequency="d",
            target_col=target_col,
            time_col=time_col,
            id_cols=[id_col],
        )

        num_rows_per_ts = 10
        sample_input_base = pd.concat(
            [
                pd.to_datetime(
                    pd.Series(range(num_rows_per_ts), name=time_col).apply(
                        lambda i: f"2020-10-{3 * i + 1}"
                    )
                ),
                pd.Series(range(num_rows_per_ts), name=target_col),
            ],
            axis=1,
        )
        sample_input = pd.concat([sample_input_base.copy(), sample_input_base.copy()], ignore_index=True)
        sample_input[id_col] = [1] * num_rows_per_ts + [2] * num_rows_per_ts

        with mlflow.start_run() as run:
            mlflow_deepar_log_model(deepar_model, sample_input)

        run_id = run.info.run_id

        # check if all additional dependencies are logged
        self._check_requirements(run_id)

        # load the model and predict
        loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

        pred_df = loaded_model.predict(sample_input)

        assert pred_df.columns.tolist() == [time_col, "yhat", id_col]
        assert len(pred_df) == self.prediction_length * 2
        assert pred_df[time_col].min() > sample_input[time_col].max()

    def _check_requirements(self, run_id: str):
        # read requirements.txt from the run
        requirements_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/model/requirements.txt")
        with open(requirements_path, "r") as f:
            requirements = f.read()
        # check if all additional dependencies are logged
        for dependency in DEEPAR_ADDITIONAL_PIP_DEPS:
            self.assertIn(dependency, requirements, f"requirements.txt should contain {dependency} but got {requirements}")
