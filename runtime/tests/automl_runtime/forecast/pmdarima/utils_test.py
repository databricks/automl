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
import pickle

import pandas as pd
import mlflow
import numpy as np
from pmdarima.arima import auto_arima, StepwiseContext

from databricks.automl_runtime.forecast.pmdarima.model import ArimaModel, MultiSeriesArimaModel
from databricks.automl_runtime.forecast.pmdarima.utils import mlflow_arima_log_model


class TestUtils(unittest.TestCase):

    def setUp(self) -> None:
        num_rows = 9
        self.X = pd.concat([
            pd.to_datetime(pd.Series(range(num_rows), name="date").apply(lambda i: f"2020-10-{i + 1}")),
            pd.Series(range(num_rows), name="y")
        ], axis=1)
        with StepwiseContext(max_steps=1):
            model = auto_arima(y=self.X.set_index("date"), m=1)
        self.pickled_model = pickle.dumps(model)

    def test_mlflow_arima_log_model(self):
        arima_model = ArimaModel(self.pickled_model, horizon=1, frequency='d',
                                 start_ds=pd.to_datetime("2020-10-01"), end_ds=pd.to_datetime("2020-10-09"),
                                 time_col="date")
        with mlflow.start_run() as run:
            mlflow_arima_log_model(arima_model)

        # Load the saved model from mlflow
        run_id = run.info.run_id
        loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

        # Make sure can call predict with the saved model
        loaded_model.predict(self.X.drop("y", axis=1))

        # Check the forecasts (including in-sample ones) with the saved model
        forecast_pd = loaded_model._model_impl.python_model.predict_timeseries()
        expected_yhat = np.array([4.002726, 0.068215, 1.572192, 2.354237, 3.519087,
                                  4.385046, 5.478589, 6.378852, 7.441261, 8.360274])
        np.testing.assert_array_almost_equal(np.array(forecast_pd["yhat"]), expected_yhat)

    def test_mlflow_arima_log_model_multiseries(self):
        pickled_model_dict = {"1": self.pickled_model, "2": self.pickled_model}
        start_ds_dict = {"1": pd.Timestamp("2020-10-01"), "2": pd.Timestamp("2020-10-01")}
        end_ds_dict = {"1": pd.Timestamp("2020-10-09"), "2": pd.Timestamp("2020-10-09")}
        multiseries_arima_model = MultiSeriesArimaModel(pickled_model_dict, horizon=1, frequency='d',
                                                        start_ds_dict=start_ds_dict, end_ds_dict=end_ds_dict,
                                                        time_col="date", id_cols=["id"])
        with mlflow.start_run() as run:
            mlflow_arima_log_model(multiseries_arima_model)

        # Load the saved model from mlflow
        run_id = run.info.run_id
        prophet_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

        # Make sure can make forecasts (including in-sample ones) with the saved model
        prophet_model._model_impl.python_model.predict_timeseries()

        # Check the prediction specified dates with the saved model
        test_df = pd.DataFrame({
            "date": [pd.to_datetime("2020-10-05"), pd.to_datetime("2020-10-05"),
                     pd.to_datetime("2020-11-04"), pd.to_datetime("2020-11-04")],
            "id": ["1", "2", "1", "2"],
        })
        expected_yhat = np.array([3.519087, 3.519087, 5.833953, 5.833953])
        yhat = prophet_model.predict(test_df)
        np.testing.assert_array_almost_equal(np.array(yhat), expected_yhat)
