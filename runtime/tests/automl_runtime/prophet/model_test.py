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

import unittest

import pandas as pd
import mlflow
import numpy as np
from prophet import Prophet
from prophet.serialize import model_to_json

from databricks.automl_runtime.forecast.prophet.model import mlflow_prophet_log_model, \
    MultiSeriesProphetModel, ProphetModel


class TestProphetModel(unittest.TestCase):

    def setUp(self) -> None:
        num_rows = 9
        self.X = pd.concat([
            pd.to_datetime(pd.Series(range(num_rows), name="ds").apply(lambda i: f"2020-07-{3*i+1}")),
            pd.Series(range(num_rows), name="y")
        ], axis=1)
        self.model = Prophet()
        self.model.fit(self.X)
        self.expected_y = np.array([0,  1.00000000e+00,  2.00000000e+00,  3.00000000e+00,
                                    4.00000000e+00,  5.00000000e+00,  6.00000000e+00,
                                    7.00000000e+00, 8.00000000e+00,  9.26072713e+00])

    def test_model_save_and_load(self):
        model_json = model_to_json(self.model)
        prophet_model = ProphetModel(model_json, 1)

        with mlflow.start_run() as run:
            mlflow_prophet_log_model(prophet_model)
        # Load the saved model from mlflow
        run_id = run.info.run_id
        prophet_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

        # Check the prediction with the saved model
        prophet_model.predict(self.X)
        forecast_pd = prophet_model._model_impl.python_model.predict_timeseries()
        np.testing.assert_array_almost_equal(np.array(forecast_pd["yhat"]), self.expected_y)

    def test_model_save_and_load_multi_series(self):
        model_json = model_to_json(self.model)
        multi_series_model_json = {"1": model_json, "2": model_json}
        multi_series_start = {"1": pd.Timestamp("2020-07-01"), "2": pd.Timestamp("2020-07-01")}
        prophet_model = MultiSeriesProphetModel(multi_series_model_json, multi_series_start,
                                                "2020-07-25", 1, "d")
        with mlflow.start_run() as run:
            mlflow_prophet_log_model(prophet_model)

        # Load the saved model from mlflow
        run_id = run.info.run_id
        prophet_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

        # Check the prediction with the saved model
        ids = pd.DataFrame(multi_series_model_json.keys(), columns=["ts_id"])
        prophet_model.predict(ids)
        prophet_model._model_impl.python_model.predict_timeseries()
