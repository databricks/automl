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
import pytest
import mlflow
import numpy as np
from prophet import Prophet
from prophet.serialize import model_to_json
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ErrorCode, INVALID_PARAMETER_VALUE

from databricks.automl_runtime.forecast.prophet.model import mlflow_prophet_log_model, \
    MultiSeriesProphetModel, ProphetModel, OFFSET_ALIAS_MAP


class TestProphetModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        num_rows = 9
        cls.X = pd.concat([
            pd.to_datetime(pd.Series(range(num_rows), name="ds").apply(lambda i: f"2020-10-{3*i+1}")),
            pd.Series(range(num_rows), name="y")
        ], axis=1)
        cls.model = Prophet()
        cls.model.fit(cls.X)
        cls.expected_y = np.array([0,  1.00000000e+00,  2.00000000e+00,  3.00000000e+00,
                                   4.00000000e+00,  5.00000000e+00,  6.00000000e+00,
                                   7.00000000e+00, 8.00000000e+00,  8.333333e+00])
        cls.model_json = model_to_json(cls.model)

    def test_model_save_and_load(self):
        prophet_model = ProphetModel(self.model_json, 1, "d", "ds", "y")

        with mlflow.start_run() as run:
            mlflow_prophet_log_model(prophet_model)
        # Load the saved model from mlflow
        run_id = run.info.run_id
        prophet_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

        # Check the prediction with the saved model
        prophet_model.predict(self.X)
        forecast_pd = prophet_model._model_impl.python_model.predict_timeseries()
        np.testing.assert_array_almost_equal(np.array(forecast_pd["yhat"]), self.expected_y)

    def test_make_future_dataframe(self):
        for feq_unit in OFFSET_ALIAS_MAP:
            prophet_model = ProphetModel(self.model_json, 1, feq_unit, "ds", "y")
            future_df = prophet_model._make_future_dataframe(1)
            expected_time = pd.Timestamp("2020-10-25") + pd.Timedelta(1, feq_unit)
            self.assertEqual(future_df.iloc[-1]["ds"], expected_time,
                             f"Wrong future dataframe generated with frequency {feq_unit}:"
                             f" Expect {expected_time}, but get {future_df.iloc[-1]['ds']}")

    def test_model_save_and_load_multi_series(self):
        multi_series_model_json = {"1": self.model_json, "2": self.model_json}
        multi_series_start = {"1": pd.Timestamp("2020-07-01"), "2": pd.Timestamp("2020-07-01")}
        prophet_model = MultiSeriesProphetModel(multi_series_model_json, multi_series_start,
                                                "2020-07-25", 1, "days", "time", "target", ["id"])
        with mlflow.start_run() as run:
            mlflow_prophet_log_model(prophet_model)

        # Load the saved model from mlflow
        run_id = run.info.run_id
        prophet_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

        # Check the prediction with the saved model
        ids = pd.DataFrame(multi_series_model_json.keys(), columns=["ts_id"])
        # Check model_predict functions
        prophet_model._model_impl.python_model.model_predict(ids)
        prophet_model._model_impl.python_model.predict_timeseries()

        # Check predict API
        num_rows = 2
        test_df = pd.concat([
            pd.to_datetime(pd.Series(range(num_rows), name="time").apply(lambda i: f"2020-11-{3*i+1}")),
            pd.Series(range(num_rows), name="id").apply(lambda i: f"{i%2+1}")
        ], axis=1)
        forecast_pd = prophet_model.predict(test_df)
        self.assertListEqual(list(forecast_pd.columns), ["id", "time", "target"])
        np.testing.assert_array_almost_equal(np.array(forecast_pd["target"]),
                                             np.array([10.333333, 11.333333]))

    def test_validate_predict_cols(self):
        prophet_model = ProphetModel(self.model_json, 1, "d", "time", "target")
        test_df = pd.concat([
            pd.to_datetime(pd.Series(range(2), name="ds").apply(lambda i: f"2020-11-{3*i+1}")),
            pd.Series(range(2), name="id").apply(lambda i: f"{i%2+1}")
        ], axis=1)
        with mlflow.start_run() as run:
            mlflow_prophet_log_model(prophet_model)
        # Load the saved model from mlflow
        run_id = run.info.run_id
        prophet_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

        with pytest.raises(MlflowException, match="Input data columns") as e:
            prophet_model.predict(test_df)
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    def test_validate_predict_cols_multi_series(self):
        multi_series_model_json = {"1": self.model_json, "2": self.model_json}
        multi_series_start = {"1": pd.Timestamp("2020-07-01"), "2": pd.Timestamp("2020-07-01")}
        prophet_model = MultiSeriesProphetModel(multi_series_model_json, multi_series_start,
                                                "2020-07-25", 1, "days", "ds", "y", ["id1"])
        test_df = pd.concat([
            pd.to_datetime(pd.Series(range(2), name="ds").apply(lambda i: f"2020-11-{3*i+1}")),
            pd.Series(range(2), name="id").apply(lambda i: f"{i%2+1}")
        ], axis=1)
        with mlflow.start_run() as run:
            mlflow_prophet_log_model(prophet_model)
        # Load the saved model from mlflow
        run_id = run.info.run_id
        prophet_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

        with pytest.raises(MlflowException, match="Input data columns") as e:
            prophet_model.predict(test_df)
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

