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
from prophet.serialize import model_from_json
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ErrorCode, INTERNAL_ERROR

from databricks.automl_runtime.forecast.prophet.model import (
    mlflow_prophet_log_model,
    MultiSeriesProphetModel,
    ProphetModel,
    OFFSET_ALIAS_MAP,
    DATE_OFFSET_KEYWORD_MAP,
    PROPHET_ADDITIONAL_PIP_DEPS
)

PROPHET_MODEL_JSON = '{"growth": "linear", "n_changepoints": 6, "specified_changepoints": false, "changepoint_range": 0.8, "yearly_seasonality": "auto", "weekly_seasonality": "auto", "daily_seasonality": "auto", "seasonality_mode": "additive", "seasonality_prior_scale": 10.0, "changepoint_prior_scale": 0.05, "holidays_prior_scale": 10.0, "mcmc_samples": 0, "interval_width": 0.8, "uncertainty_samples": 1000, "y_scale": 8.0, "logistic_floor": false, "country_holidays": null, "component_modes": {"additive": ["weekly", "additive_terms", "extra_regressors_additive", "holidays"], "multiplicative": ["multiplicative_terms", "extra_regressors_multiplicative"]}, "changepoints": "{\\"name\\":\\"ds\\",\\"index\\":[1,2,3,4,5,6],\\"data\\":[\\"2020-10-04T00:00:00.000\\",\\"2020-10-07T00:00:00.000\\",\\"2020-10-10T00:00:00.000\\",\\"2020-10-13T00:00:00.000\\",\\"2020-10-16T00:00:00.000\\",\\"2020-10-19T00:00:00.000\\"]}", "history_dates": "{\\"name\\":\\"ds\\",\\"index\\":[0,1,2,3,4,5,6,7,8],\\"data\\":[\\"2020-10-01T00:00:00.000\\",\\"2020-10-04T00:00:00.000\\",\\"2020-10-07T00:00:00.000\\",\\"2020-10-10T00:00:00.000\\",\\"2020-10-13T00:00:00.000\\",\\"2020-10-16T00:00:00.000\\",\\"2020-10-19T00:00:00.000\\",\\"2020-10-22T00:00:00.000\\",\\"2020-10-25T00:00:00.000\\"]}", "train_holiday_names": null, "start": 1601510400.0, "t_scale": 2073600.0, "holidays": null, "history": "{\\"schema\\":{\\"fields\\":[{\\"name\\":\\"ds\\",\\"type\\":\\"datetime\\"},{\\"name\\":\\"y\\",\\"type\\":\\"integer\\"},{\\"name\\":\\"floor\\",\\"type\\":\\"integer\\"},{\\"name\\":\\"t\\",\\"type\\":\\"number\\"},{\\"name\\":\\"y_scaled\\",\\"type\\":\\"number\\"}],\\"pandas_version\\":\\"1.4.0\\"},\\"data\\":[{\\"ds\\":\\"2020-10-01T00:00:00.000\\",\\"y\\":0,\\"floor\\":0,\\"t\\":0.0,\\"y_scaled\\":0.0},{\\"ds\\":\\"2020-10-04T00:00:00.000\\",\\"y\\":1,\\"floor\\":0,\\"t\\":0.125,\\"y_scaled\\":0.125},{\\"ds\\":\\"2020-10-07T00:00:00.000\\",\\"y\\":2,\\"floor\\":0,\\"t\\":0.25,\\"y_scaled\\":0.25},{\\"ds\\":\\"2020-10-10T00:00:00.000\\",\\"y\\":3,\\"floor\\":0,\\"t\\":0.375,\\"y_scaled\\":0.375},{\\"ds\\":\\"2020-10-13T00:00:00.000\\",\\"y\\":4,\\"floor\\":0,\\"t\\":0.5,\\"y_scaled\\":0.5},{\\"ds\\":\\"2020-10-16T00:00:00.000\\",\\"y\\":5,\\"floor\\":0,\\"t\\":0.625,\\"y_scaled\\":0.625},{\\"ds\\":\\"2020-10-19T00:00:00.000\\",\\"y\\":6,\\"floor\\":0,\\"t\\":0.75,\\"y_scaled\\":0.75},{\\"ds\\":\\"2020-10-22T00:00:00.000\\",\\"y\\":7,\\"floor\\":0,\\"t\\":0.875,\\"y_scaled\\":0.875},{\\"ds\\":\\"2020-10-25T00:00:00.000\\",\\"y\\":8,\\"floor\\":0,\\"t\\":1.0,\\"y_scaled\\":1.0}]}", "train_component_cols": "{\\"schema\\":{\\"fields\\":[{\\"name\\":\\"additive_terms\\",\\"type\\":\\"integer\\"},{\\"name\\":\\"weekly\\",\\"type\\":\\"integer\\"},{\\"name\\":\\"multiplicative_terms\\",\\"type\\":\\"integer\\"}],\\"pandas_version\\":\\"1.4.0\\"},\\"data\\":[{\\"additive_terms\\":1,\\"weekly\\":1,\\"multiplicative_terms\\":0},{\\"additive_terms\\":1,\\"weekly\\":1,\\"multiplicative_terms\\":0},{\\"additive_terms\\":1,\\"weekly\\":1,\\"multiplicative_terms\\":0},{\\"additive_terms\\":1,\\"weekly\\":1,\\"multiplicative_terms\\":0},{\\"additive_terms\\":1,\\"weekly\\":1,\\"multiplicative_terms\\":0},{\\"additive_terms\\":1,\\"weekly\\":1,\\"multiplicative_terms\\":0}]}", "changepoints_t": [0.125, 0.25, 0.375, 0.5, 0.625, 0.75], "seasonalities": [["weekly"], {"weekly": {"period": 7, "fourier_order": 3, "prior_scale": 10.0, "mode": "additive", "condition_name": null}}], "extra_regressors": [[], {}], "fit_kwargs": {}, "params": {"lp__": [[202.053]], "k": [[1.19777]], "m": [[0.0565623]], "delta": [[-0.86152, 0.409957, -0.103241, 0.528979, 0.535181, -0.509356]], "sigma_obs": [[2.53056e-13]], "beta": [[-0.00630566, 0.016248, 0.0318587, -0.068705, 0.0029986, -0.00410522]], "trend": [[0.0565623, 0.206283, 0.248314, 0.341589, 0.421959, 0.568452, 0.781842, 0.931562, 1.08128]]}, "__prophet_version": "1.1.1"}'

class BaseTest(unittest.TestCase):
    def _check_requirements(self, run_id: str):
        # read requirements.txt from the run
        requirements_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/model/requirements.txt")
        with open(requirements_path, "r") as f:
            requirements = f.read()
        # check if all additional dependencies are logged
        for dependency in PROPHET_ADDITIONAL_PIP_DEPS:
            self.assertIn(dependency, requirements, f"requirements.txt should contain {dependency} but got {requirements}")

class TestProphetModel(BaseTest):
    @classmethod
    def setUpClass(cls) -> None:
        num_rows = 9
        cls.X = pd.concat(
            [
                pd.to_datetime(
                    pd.Series(range(num_rows), name="ds").apply(
                        lambda i: f"2020-10-{3*i+1}"
                    )
                ),
                pd.Series(range(num_rows), name="y"),
            ],
            axis=1,
        )
        cls.expected_y = np.array(
            [
                6.399995e-07,
                1.000005e00,
                2.000010e00,
                3.000014e00,
                4.000019e00,
                5.000024e00,
                6.000029e00,
                7.000035e00,
                8.000039e00,
                8.794826e00,
            ]
        )
        cls.model_json = PROPHET_MODEL_JSON
        cls.model = model_from_json(cls.model_json)

    def test_model_save_and_load(self):
        prophet_model = ProphetModel(self.model_json, 1, "d", "ds")

        with mlflow.start_run() as run:
            mlflow_prophet_log_model(prophet_model)
        
        run_id = run.info.run_id

        # Check additonal requirements logged correctly
        self._check_requirements(run_id)

        # Load the saved model from mlflow
        prophet_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

        # Check the prediction with the saved model
        prophet_model.predict(self.X)
        forecast_pd = prophet_model._model_impl.python_model.predict_timeseries()
        np.testing.assert_array_almost_equal(
            np.array(forecast_pd["yhat"]), self.expected_y
        )
        forecast_future_pd = prophet_model._model_impl.python_model.predict_timeseries(
            include_history=False
        )
        self.assertEqual(len(forecast_future_pd), 1)

    def test_make_future_dataframe(self):
        for feq_unit in OFFSET_ALIAS_MAP:
            # Temporally disable the year, month and quater since we
            # don't have full support yet.
            if OFFSET_ALIAS_MAP[feq_unit] in ['YS', 'MS', 'QS']:
                continue
            prophet_model = ProphetModel(self.model_json, 1, feq_unit, "ds")
            future_df = prophet_model.make_future_dataframe(1)
            offset_kw_arg = DATE_OFFSET_KEYWORD_MAP[OFFSET_ALIAS_MAP[feq_unit]]
            expected_time = pd.Timestamp("2020-10-25") + pd.DateOffset(**offset_kw_arg)
            self.assertEqual(future_df.iloc[-1]["ds"], expected_time,
                             f"Wrong future dataframe generated with frequency {feq_unit}:"
                             f" Expect {expected_time}, but get {future_df.iloc[-1]['ds']}")

    def test_predict_success_datetime_date(self):
        prophet_model = ProphetModel(self.model_json, 1, "d", "ds")
        test_df = pd.DataFrame(
            {"ds": [datetime.date(2020, 10, 8), datetime.date(2020, 12, 10)]}
        )
        expected_test_df = test_df.copy()
        yhat = prophet_model.predict(None, test_df)
        self.assertEqual(2, len(yhat))
        pd.testing.assert_frame_equal(
            test_df, expected_test_df
        )  # check the input dataframe is unchanged

    def test_predict_success_string(self):
        prophet_model = ProphetModel(self.model_json, 1, "d", "ds")
        test_df = pd.DataFrame({"ds": ["2020-10-08", "2020-12-10"]})
        expected_test_df = test_df.copy()
        yhat = prophet_model.predict(None, test_df)
        self.assertEqual(2, len(yhat))
        pd.testing.assert_frame_equal(
            test_df, expected_test_df
        )  # check the input dataframe is unchanged

    def test_validate_predict_cols(self):
        prophet_model = ProphetModel(self.model_json, 1, "d", "time")
        test_df = pd.DataFrame(
            {
                "date": [pd.to_datetime("2020-11-01"), pd.to_datetime("2020-11-04")],
                "id": ["1", "2"],
            }
        )
        with mlflow.start_run() as run:
            mlflow_prophet_log_model(prophet_model)
        # Load the saved model from mlflow
        run_id = run.info.run_id
        prophet_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

        with pytest.raises(MlflowException, match="Model is missing inputs") as e:
            prophet_model.predict(test_df)
        assert e.value.error_code == ErrorCode.Name(INTERNAL_ERROR)


class TestMultiSeriesProphetModel(BaseTest):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model_json = PROPHET_MODEL_JSON
        cls.multi_series_model_json = {("1",): cls.model_json, ("2",): cls.model_json}
        cls.multi_series_start = {
            ("1",): pd.Timestamp("2020-07-01"),
            ("2",): pd.Timestamp("2020-07-01"),
        }
        cls.prophet_model = MultiSeriesProphetModel(
            model_json=cls.multi_series_model_json,
            timeseries_starts=cls.multi_series_start,
            timeseries_end="2020-07-25",
            horizon=1,
            frequency="days",
            time_col="time",
            id_cols=["id"],
        )

    def test_model_save_and_load(self):
        test_df = pd.DataFrame(
            {
                "time": [
                    pd.to_datetime("2020-11-01"),
                    pd.to_datetime("2020-11-01"),
                    pd.to_datetime("2020-11-04"),
                    pd.to_datetime("2020-11-04"),
                ],
                "id": ["1", "2", "1", "2"],
            }
        )
        with mlflow.start_run() as run:
            mlflow_prophet_log_model(self.prophet_model, sample_input=test_df)

        
        run_id = run.info.run_id

        # Check additonal requirements logged correctly
        self._check_requirements(run_id)

        # Load the saved model from mlflow
        loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

        # Check the prediction with the saved model
        future_df = loaded_model._model_impl.python_model.make_future_dataframe(include_history=False)
        # Check model_predict functions
        forecast_pd = loaded_model._model_impl.python_model.model_predict(future_df)
        expected_columns = {"id", "ds", "yhat", "yhat_lower", "yhat_upper"}
        self.assertTrue(expected_columns.issubset(set(forecast_pd.columns)))
        forecast_pd = loaded_model._model_impl.python_model.predict_timeseries()
        expected_columns = {"id", "ds", "yhat", "yhat_lower", "yhat_upper"}
        self.assertTrue(expected_columns.issubset(set(forecast_pd.columns)))

        forecast_future_pd = loaded_model._model_impl.python_model.predict_timeseries(
            include_history=False
        )
        self.assertEqual(len(forecast_future_pd), 2)

        # Check predict API
        expected_test_df = test_df.copy()
        forecast_y = loaded_model.predict(test_df)
        np.testing.assert_array_almost_equal(
            np.array(forecast_y), np.array([10.794835, 10.794835, 12.65636, 12.65636])
        )
        # Make sure that the input dataframe is unchanged
        assert_frame_equal(test_df, expected_test_df)

        # Check predict API works with one-row dataframe
        loaded_model.predict(test_df[0:1])

    def test_model_save_and_load_multi_ids(self):
        multi_series_model_json = {("1", "1"): self.model_json, ("2", "1"): self.model_json}
        multi_series_start = {
            ("1", "1"): pd.Timestamp("2020-07-01"),
            ("2", "1"): pd.Timestamp("2020-07-01"),
        }
        prophet_model = MultiSeriesProphetModel(
            multi_series_model_json,
            multi_series_start,
            "2020-07-25",
            1,
            "days",
            "time",
            ["id1", "id2"],
        )
        # The id of the last row does not match to any saved model. It should return nan.
        test_df = pd.DataFrame(
            {
                "time": [
                    pd.to_datetime("2020-11-01"),
                    pd.to_datetime("2020-11-01"),
                    pd.to_datetime("2020-11-04"),
                    pd.to_datetime("2020-11-04"),
                ],
                "id1": ["1", "2", "1", "1"],
                "id2": ["1", "1", "1", "2"],
            }
        )
        with mlflow.start_run() as run:
            mlflow_prophet_log_model(prophet_model, sample_input=test_df)
 
        run_id = run.info.run_id

        # Check additonal requirements logged correctly
        self._check_requirements(run_id)

        # Load the saved model from mlflow
        loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

        # Check the prediction with the saved model
        future_df = loaded_model._model_impl.python_model.make_future_dataframe(include_history=False)
        # Check model_predict functions
        forecast_pd = loaded_model._model_impl.python_model.model_predict(future_df)
        expected_columns = {"id1", "id2", "ds", "yhat", "yhat_lower", "yhat_upper"}
        self.assertTrue(expected_columns.issubset(set(forecast_pd.columns)))
        forecast_pd = loaded_model._model_impl.python_model.predict_timeseries()
        expected_columns = {"id1", "id2", "ds", "yhat", "yhat_lower", "yhat_upper"}
        self.assertTrue(expected_columns.issubset(set(forecast_pd.columns)))
        forecast_future_pd = loaded_model._model_impl.python_model.predict_timeseries(
            include_history=False
        )
        self.assertEqual(len(forecast_future_pd), 2)

        # Check predict API
        expected_test_df = test_df.copy()
        forecast_y = loaded_model.predict(test_df)
        np.testing.assert_array_almost_equal(
            np.array(forecast_y), np.array([10.794835, 10.794835, 12.65636, np.nan])
        )
        # Make sure that the input dataframe is unchanged
        assert_frame_equal(test_df, expected_test_df)

    def test_predict_success_one_row(self):
        test_df = pd.DataFrame({"time": [pd.to_datetime("2020-11-01")], "id": ["1"]})
        yhat = self.prophet_model.predict(None, test_df)
        self.assertEqual(1, len(yhat))

    def test_validate_predict_cols(self):
        prophet_model = MultiSeriesProphetModel(
            model_json=self.multi_series_model_json,
            timeseries_starts=self.multi_series_start,
            timeseries_end="2020-07-25",
            horizon=1,
            frequency="days",
            time_col="ds",
            id_cols=["id1"],
        )
        sample_df = pd.DataFrame(
            {
                "ds": [
                    pd.to_datetime("2020-11-01"),
                    pd.to_datetime("2020-11-01"),
                    pd.to_datetime("2020-11-04"),
                    pd.to_datetime("2020-11-04"),
                ],
                "id1": ["1", "2", "1", "2"],
            }
        )
        test_df = pd.DataFrame(
            {
                "time": [pd.to_datetime("2020-11-01"), pd.to_datetime("2020-11-04")],
                "id": ["1", "2"],
            }
        )
        with mlflow.start_run() as run:
            mlflow_prophet_log_model(prophet_model, sample_input=sample_df)
        # Load the saved model from mlflow
        run_id = run.info.run_id
        prophet_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

        with pytest.raises(MlflowException, match="Model is missing inputs") as e:
            prophet_model.predict(test_df)
        assert e.value.error_code == ErrorCode.Name(INTERNAL_ERROR)

    def test_make_future_dataframe(self):
        future_df = self.prophet_model.make_future_dataframe(include_history=False)
        self.assertCountEqual(future_df.columns, {"ds", "id"})
        self.assertEqual(2, future_df.shape[0])

    def test_make_future_dataframe_multi_ids(self):
        multi_series_model_json = {(1, "1"): self.model_json, (2, "1"): self.model_json}
        multi_series_start = {
            (1, "1"): pd.Timestamp("2020-07-01"),
            (2, "1"): pd.Timestamp("2020-07-01"),
        }
        prophet_model = MultiSeriesProphetModel(
            multi_series_model_json,
            multi_series_start,
            "2020-07-25",
            1,
            "days",
            "time",
            ["id1", "id2"],
        )
        future_df = prophet_model.make_future_dataframe(include_history=False)
        self.assertCountEqual(future_df.columns, {"ds", "id1", "id2"})
        # Make sure keep the column types for identity columns
        self.assertTrue(future_df.dtypes["id1"] == "int")
        self.assertTrue(future_df.dtypes["id2"] == "object")
        self.assertEqual(2, future_df.shape[0])

    def test_make_future_dataframe_invalid_group(self):
        with pytest.raises(ValueError, match="Invalid groups:"):
            future_df = self.prophet_model.make_future_dataframe(groups=[(1,)])
