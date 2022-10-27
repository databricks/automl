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
import datetime

import pandas as pd
import pytest
import mlflow
import numpy as np
from pandas._testing import assert_frame_equal
from prophet import Prophet
from prophet.serialize import model_to_json
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ErrorCode, INTERNAL_ERROR

from databricks.automl_runtime.forecast.prophet.model import mlflow_prophet_log_model, \
    MultiSeriesProphetModel, ProphetModel, OFFSET_ALIAS_MAP


class TestProphetModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        num_rows = 9
        cls.X = pd.concat([
            pd.to_datetime(pd.Series(range(num_rows), name="ds")
                           .apply(lambda i: f"2020-10-{3*i+1}")),
            pd.Series(range(num_rows), name="y")
        ], axis=1)
        cls.model = Prophet()
        cls.model.fit(cls.X)
        cls.expected_y = np.array([0,  1.00000000e+00,  2.00000000e+00,  3.00000000e+00,
                                   4.00000000e+00,  5.00000000e+00,  6.00000000e+00,
                                   7.00000000e+00, 8.00000000e+00,  8.333333e+00])
        cls.model_json = model_to_json(cls.model)

    def test_model_save_and_load(self):
        prophet_model = ProphetModel(self.model_json, 1, "d", "ds")

        with mlflow.start_run() as run:
            mlflow_prophet_log_model(prophet_model)
        # Load the saved model from mlflow
        run_id = run.info.run_id
        prophet_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

        # Check the prediction with the saved model
        prophet_model.predict(self.X)
        forecast_pd = prophet_model._model_impl.python_model.predict_timeseries()
        np.testing.assert_array_almost_equal(np.array(forecast_pd["yhat"]), self.expected_y)
        forecast_future_pd = prophet_model._model_impl.python_model.predict_timeseries(include_history=False)
        self.assertEqual(len(forecast_future_pd), 1)

    def test_make_future_dataframe(self):
        for feq_unit in OFFSET_ALIAS_MAP:
            # Temporally disable the year month and quater since we
            # don't have full support yet.
            if OFFSET_ALIAS_MAP[feq_unit] in ['YS', 'MS', 'QS']:
                continue
            prophet_model = ProphetModel(self.model_json, 1, feq_unit, "ds")
            future_df = prophet_model._make_future_dataframe(1)
            expected_time = pd.Timestamp("2020-10-25") + pd.Timedelta(1, feq_unit)
            self.assertEqual(future_df.iloc[-1]["ds"], expected_time,
                             f"Wrong future dataframe generated with frequency {feq_unit}:"
                             f" Expect {expected_time}, but get {future_df.iloc[-1]['ds']}")

    def test_model_save_and_load_multi_series(self):
        multi_series_model_json = {"1": self.model_json, "2": self.model_json}
        multi_series_start = {"1": pd.Timestamp("2020-07-01"), "2": pd.Timestamp("2020-07-01")}
        prophet_model = MultiSeriesProphetModel(multi_series_model_json, multi_series_start,
                                                "2020-07-25", 1, "days", "time", ["id"])
        test_df = pd.DataFrame({
            "time": [pd.to_datetime("2020-11-01"), pd.to_datetime("2020-11-01"),
                     pd.to_datetime("2020-11-04"), pd.to_datetime("2020-11-04")],
            "id": ["1", "2", "1", "2"],
        })
        with mlflow.start_run() as run:
            mlflow_prophet_log_model(prophet_model, sample_input=test_df)

        # Load the saved model from mlflow
        run_id = run.info.run_id
        loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

        # Check the prediction with the saved model
        ids = pd.DataFrame(multi_series_model_json.keys(), columns=["ts_id"])
        # Check model_predict functions
        loaded_model._model_impl.python_model.model_predict(ids)
        loaded_model._model_impl.python_model.predict_timeseries()
        forecast_future_pd = loaded_model._model_impl.python_model.predict_timeseries(include_history=False)
        self.assertEqual(len(forecast_future_pd), 2)

        # Check predict API
        expected_test_df = test_df.copy()
        forecast_y = loaded_model.predict(test_df)
        np.testing.assert_array_almost_equal(np.array(forecast_y),
                                             np.array([10.333333, 10.333333, 11.333333, 11.333333]))
        # Make sure that the input dataframe is unchanged
        assert_frame_equal(test_df, expected_test_df)

        # Check predict API works with one-row dataframe
        loaded_model.predict(test_df[0:1])

    def test_model_save_and_load_multi_series_multi_ids(self):
        multi_series_model_json = {"1-1": self.model_json, "2-1": self.model_json}
        multi_series_start = {"1-1": pd.Timestamp("2020-07-01"), "2-1": pd.Timestamp("2020-07-01")}
        prophet_model = MultiSeriesProphetModel(multi_series_model_json, multi_series_start,
                                                "2020-07-25", 1, "days", "time", ["id1", "id2"])
        # The id of the last row does not match to any saved model. It should return nan.
        test_df = pd.DataFrame({
            "time": [pd.to_datetime("2020-11-01"), pd.to_datetime("2020-11-01"),
                     pd.to_datetime("2020-11-04"), pd.to_datetime("2020-11-04")],
            "id1": ["1", "2", "1", "1"],
            "id2": ["1", "1", "1", "2"],
        })
        with mlflow.start_run() as run:
            mlflow_prophet_log_model(prophet_model, sample_input=test_df)

        # Load the saved model from mlflow
        run_id = run.info.run_id
        loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

        # Check the prediction with the saved model
        ids = pd.DataFrame(multi_series_model_json.keys(), columns=["ts_id"])
        # Check model_predict functions
        loaded_model._model_impl.python_model.model_predict(ids)
        loaded_model._model_impl.python_model.predict_timeseries()
        forecast_future_pd = loaded_model._model_impl.python_model.predict_timeseries(include_history=False)
        self.assertEqual(len(forecast_future_pd), 2)

        # Check predict API
        expected_test_df = test_df.copy()
        forecast_y = loaded_model.predict(test_df)
        np.testing.assert_array_almost_equal(np.array(forecast_y),
                                             np.array([10.333333, 10.333333, 11.333333, np.nan]))
        # Make sure that the input dataframe is unchanged
        assert_frame_equal(test_df, expected_test_df)

    def test_predict_success_multi_series_one_row(self):
        multi_series_model_json = {"1": self.model_json, "2": self.model_json}
        multi_series_start = {"1": pd.Timestamp("2020-07-01"), "2": pd.Timestamp("2020-07-01")}
        prophet_model = MultiSeriesProphetModel(multi_series_model_json, multi_series_start,
                                                "2020-07-25", 1, "days", "time", ["id"])
        test_df = pd.DataFrame({
            "time": [pd.to_datetime("2020-11-01")], "id": ["1"]
        })
        yhat = prophet_model.predict(None, test_df)
        self.assertEqual(1, len(yhat))

    def test_predict_success_datetime_date(self):
        prophet_model = ProphetModel(self.model_json, 1, "d", "ds")
        test_df = pd.DataFrame({
            "ds": [datetime.date(2020, 10, 8), datetime.date(2020, 12, 10)]
        })
        expected_test_df = test_df.copy()
        yhat = prophet_model.predict(None, test_df)
        self.assertEqual(2, len(yhat))
        pd.testing.assert_frame_equal(test_df, expected_test_df)  # check the input dataframe is unchanged

    def test_predict_success_string(self):
        prophet_model = ProphetModel(self.model_json, 1, "d", "ds")
        test_df = pd.DataFrame({
            "ds": ["2020-10-08", "2020-12-10"]
        })
        expected_test_df = test_df.copy()
        yhat = prophet_model.predict(None, test_df)
        self.assertEqual(2, len(yhat))
        pd.testing.assert_frame_equal(test_df, expected_test_df)  # check the input dataframe is unchanged

    def test_validate_predict_cols(self):
        prophet_model = ProphetModel(self.model_json, 1, "d", "time")
        test_df = pd.DataFrame({
            "date": [pd.to_datetime("2020-11-01"), pd.to_datetime("2020-11-04")],
            "id": ["1", "2"],
        })
        with mlflow.start_run() as run:
            mlflow_prophet_log_model(prophet_model)
        # Load the saved model from mlflow
        run_id = run.info.run_id
        prophet_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

        with pytest.raises(MlflowException, match="Model is missing inputs") as e:
            prophet_model.predict(test_df)
        assert e.value.error_code == ErrorCode.Name(INTERNAL_ERROR)

    def test_validate_predict_cols_multi_series(self):
        multi_series_model_json = {"1": self.model_json, "2": self.model_json}
        multi_series_start = {"1": pd.Timestamp("2020-07-01"), "2": pd.Timestamp("2020-07-01")}
        prophet_model = MultiSeriesProphetModel(multi_series_model_json, multi_series_start,
                                                "2020-07-25", 1, "days", "ds", ["id1"])
        sample_df = pd.DataFrame({
            "ds": [pd.to_datetime("2020-11-01"), pd.to_datetime("2020-11-01"),
                   pd.to_datetime("2020-11-04"), pd.to_datetime("2020-11-04")],
            "id1": ["1", "2", "1", "2"],
        })
        test_df = pd.DataFrame({
            "time": [pd.to_datetime("2020-11-01"), pd.to_datetime("2020-11-04")],
            "id": ["1", "2"],
        })
        with mlflow.start_run() as run:
            mlflow_prophet_log_model(prophet_model, sample_input=sample_df)
        # Load the saved model from mlflow
        run_id = run.info.run_id
        prophet_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

        with pytest.raises(MlflowException, match="Model is missing inputs") as e:
            prophet_model.predict(test_df)
        assert e.value.error_code == ErrorCode.Name(INTERNAL_ERROR)
