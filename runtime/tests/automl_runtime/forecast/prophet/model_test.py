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
from unittest.mock import patch
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

class BaseProphetModelTest(unittest.TestCase):
    def _check_requirements(self, run_id: str):
        # read requirements.txt from the run
        requirements_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/model/requirements.txt")
        with open(requirements_path, "r") as f:
            requirements = f.read()
        # check if all additional dependencies are logged
        for dependency in PROPHET_ADDITIONAL_PIP_DEPS:
            self.assertIn(dependency, requirements, f"requirements.txt should contain {dependency} but got {requirements}")

class TestProphetModel(BaseProphetModelTest):
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
        prophet_model = ProphetModel(self.model_json, 1, "d", 1, "ds", )

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
    
    @patch("databricks.automl_runtime.forecast.prophet.model.ProphetModel._predict_impl")
    def test_predict_timeseries_with_preprocess_func(self, mock_predict_impl):
        # Mock the output of _predict_impl
        mock_predict_impl.side_effect = lambda df: df

        # Define a preprocess function
        def preprocess_func(df):
            df["feature"] = df["feature"] * 2
            return df

        # Create a ProphetModel instance with preprocess_func
        prophet_model = ProphetModel(
            model_json=PROPHET_MODEL_JSON,
            horizon=3,
            frequency_unit="d",
            frequency_quantity=1,
            time_col="time",
            preprocess_func=preprocess_func,
            split_col="split"
        )

        # Input DataFrame
        input_df = pd.DataFrame({"time": ["2020-10-01", "2020-10-02", "2020-10-03"], "feature": [1, 2, 3]})

        # Call predict_timeseries
        result = prophet_model.predict_timeseries(future_df=input_df)

        # Assertions
        mock_predict_impl.assert_called_once()
        self.assertEqual(len(result), 3)

        # Check if the preprocess_func was applied
        processed_df = mock_predict_impl.call_args[0][0]  # Get the DataFrame passed to _predict_impl
        self.assertTrue((processed_df["feature"] == [2, 4, 6]).all())  # Check if "y" was doubled
        self.assertIn("ds", processed_df.columns)  # Ensure "ds" column exists

    def test_make_future_dataframe(self):
        for feq_unit in OFFSET_ALIAS_MAP:
            # Temporally disable the year, month and quarter since we
            # don't have full support yet.
            if OFFSET_ALIAS_MAP[feq_unit] in ['YS', 'MS', 'QS']:
                continue
            prophet_model = ProphetModel(self.model_json, 1, feq_unit, 1, "ds")
            future_df = prophet_model.make_future_dataframe(1)
            offset_kw_arg = DATE_OFFSET_KEYWORD_MAP[OFFSET_ALIAS_MAP[feq_unit]]
            expected_time = pd.Timestamp("2020-10-25") + pd.DateOffset(**offset_kw_arg)
            self.assertEqual(future_df.iloc[-1]["ds"], expected_time,
                             f"Wrong future dataframe generated with frequency {feq_unit}:"
                             f" Expect {expected_time}, but get {future_df.iloc[-1]['ds']}")

    def test_make_future_dataframe_with_multiple_frequency_quantities(self):
        for frequency_quantity in [1, 5, 10, 15, 30]:
            prophet_model = ProphetModel(self.model_json, 1, "min", frequency_quantity, "ds")
            future_df = prophet_model.make_future_dataframe(1)
            offset_kw_arg = DATE_OFFSET_KEYWORD_MAP[OFFSET_ALIAS_MAP["min"]]
            expected_time = pd.Timestamp("2020-10-25") + pd.DateOffset(**offset_kw_arg)*frequency_quantity
            self.assertEqual(future_df.iloc[-1]["ds"], expected_time,
                             f"Wrong future dataframe generated with frequency min:"
                             f" Expect {expected_time}, but get {future_df.iloc[-1]['ds']}")

    def test_predict_success_datetime_date(self):
        prophet_model = ProphetModel(self.model_json, 1, "d", 1, "ds")
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
        prophet_model = ProphetModel(self.model_json, 1, "d", 1, "ds")
        test_df = pd.DataFrame({"ds": ["2020-10-08", "2020-12-10"]})
        expected_test_df = test_df.copy()
        yhat = prophet_model.predict(None, test_df)
        self.assertEqual(2, len(yhat))
        pd.testing.assert_frame_equal(
            test_df, expected_test_df
        )  # check the input dataframe is unchanged

    def test_predict_multiple_frequency_quantities(self):
        for frequency_quantity in [1, 5, 10, 15, 30]:
            prophet_model = ProphetModel(self.model_json, 1, "min", frequency_quantity, "ds")
            test_df = pd.DataFrame({"ds": ["2020-10-08", "2020-12-10"]})
            expected_test_df = test_df.copy()
            yhat = prophet_model.predict(None, test_df)
            self.assertEqual(2, len(yhat))
            pd.testing.assert_frame_equal(
                test_df, expected_test_df
            )  # check the input dataframe is unchanged

    def test_validate_predict_cols(self):
        prophet_model = ProphetModel(self.model_json, 1, "d", 1, "time")
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

    def test_predict_with_preprocess_func(self):
        def preprocess_func(df):
            df["y"] = df["y"] * 2
            return df
        prophet_model = ProphetModel(self.model_json, 1, "d", 1, "ds", "split", preprocess_func)
        test_df = pd.DataFrame(
            {
                "ds": [pd.to_datetime("2020-11-01"), pd.to_datetime("2020-11-04")], 
                "split": ["train", "test"],
                "y": [1, 2]
            }
        )
        yhat = prophet_model.predict(None, test_df)
        self.assertEqual(2, len(yhat))


class TestMultiSeriesProphetModel(BaseProphetModelTest):
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
            frequency_unit="days",
            frequency_quantity=1,
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
            1,
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
            frequency_unit="days",
            frequency_quantity=1,
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

    def test_make_future_dataframe_multiple_frequency_quantities(self):

        for frequency_quantity in [1, 5, 10, 15, 30]:
            prophet_model = MultiSeriesProphetModel(
                model_json=self.multi_series_model_json,
                timeseries_starts=self.multi_series_start,
                timeseries_end="2020-07-25",
                horizon=1,
                frequency_unit="min",
                frequency_quantity=frequency_quantity,
                time_col="time",
                id_cols=["id"],
            )
            future_df = prophet_model.make_future_dataframe(include_history=False)
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
            1,
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


    def test_predict_with_preprocess_func(self):
        def preprocess_func(df):
            df["y"] = df["y"] + 1
            return df
        multi_series_model_json = {("1", ): self.model_json, ("2", ): self.model_json}
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
            1, 
            "ds", 
            ["id"],
            "split", 
            preprocess_func)
        test_df = pd.DataFrame(
            {
                "ds": [pd.to_datetime("2020-11-01"), pd.to_datetime("2020-11-02")], 
                "split": ["train", "test"],
                "id": ["1", "2"],
            }
        )
        yhat = prophet_model.predict(None, test_df)
        self.assertEqual(2, len(yhat))

    @patch("databricks.automl_runtime.forecast.prophet.model.MultiSeriesProphetModel._predict_impl")
    def test_predict_timeseries(self, mock_predict_impl):
        # Mock the output of _predict_impl
        mock_predict_impl.side_effect = lambda df, horizon, include_history: pd.DataFrame({
            "ds": df["ds"],
            "feature": df["feature"],
            "id": df["id"]
        })

        # Define a preprocess function
        def preprocess_func(df):
            df["feature"] = df["feature"] * 2
            return df

        # Create a MultiSeriesProphetModel instance
        model_json = {
            ("id1",): '{"model": "mock_model_1"}',
            ("id2",): '{"model": "mock_model_2"}'
        }
        timeseries_starts = {("id1",): pd.Timestamp("2020-01-01"), ("id2",): pd.Timestamp("2020-01-01")}
        timeseries_end = "2020-12-31"
        prophet_model = MultiSeriesProphetModel(
            model_json=model_json,
            timeseries_starts=timeseries_starts,
            timeseries_end=timeseries_end,
            horizon=3,
            frequency_unit="d",
            frequency_quantity=1,
            time_col="time",
            id_cols=["id"],
            preprocess_func=preprocess_func,
            split_col="split"
        )

        # Input DataFrame
        input_df = pd.DataFrame({
            "time": ["2020-10-01", "2020-10-02", "2020-10-03", "2020-10-01", "2020-10-02", "2020-10-03"],
            "feature": [1, 2, 3, 4, 5, 6],
            "id": ["id1", "id1", "id1", "id2", "id2", "id2"]
        })

        # Call predict_timeseries
        result = prophet_model.predict_timeseries(future_df=input_df)

        # Assertions
        mock_predict_impl.assert_called()
        self.assertEqual(len(result), 6)
        self.assertIn("feature", result.columns)
        self.assertIn("ds", result.columns)
        self.assertIn("id", result.columns)

        # Check the calls to _predict_impl
        calls = mock_predict_impl.call_args_list
        self.assertEqual(len(calls), 2)  # Ensure _predict_impl is called twice (once per group)

        # Check the first call
        first_call_df = calls[0][0][0]  # Get the DataFrame passed in the first call
        self.assertTrue((first_call_df["feature"] == [2, 4, 6]).all())
        self.assertTrue((first_call_df["id"] == ["id1", "id1", "id1"]).all())

        # Check the second call
        second_call_df = calls[1][0][0]  # Get the DataFrame passed in the second call
        self.assertTrue((second_call_df["feature"] == [8, 10, 12]).all())
        self.assertTrue((second_call_df["id"] == ["id2", "id2", "id2"]).all())

class TestProphetModelCategoryEncoders(BaseProphetModelTest):
    """Test category_encoders dependency inclusion"""
    
    @classmethod
    def setUpClass(cls) -> None:
        cls.model_json = PROPHET_MODEL_JSON

    def test_category_encoders_in_requirements(self):
        """Test that category_encoders is included in model requirements"""
        prophet_model = ProphetModel(self.model_json, 1, "d", 1, "ds")
        
        with mlflow.start_run() as run:
            mlflow_prophet_log_model(prophet_model)
        
        run_id = run.info.run_id
        
        # Read requirements.txt from the run
        requirements_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/model/requirements.txt")
        with open(requirements_path, "r") as f:
            requirements = f.read()
        
        # Verify category_encoders is included in requirements
        self.assertIn("category_encoders", requirements, "category_encoders should be included in model requirements")
        
        # Verify the specific version is included (from PROPHET_ADDITIONAL_PIP_DEPS)
        import category_encoders
        expected_dep = f"category_encoders=={category_encoders.__version__}"
        self.assertIn(expected_dep, requirements, f"Specific category_encoders version {expected_dep} should be in requirements")

    def test_model_with_category_encoding_preprocessing(self):
        """Test that models work correctly with category encoding preprocessing functions"""
        import category_encoders as ce
        
        def preprocess_func_with_category_encoding(df):
            """Preprocessing function that uses category_encoders"""
            # Simulate categorical encoding preprocessing
            if 'category_col' in df.columns:
                encoder = ce.BinaryEncoder(cols=['category_col'])
                df = encoder.fit_transform(df)
            return df
        
        prophet_model = ProphetModel(
            model_json=self.model_json,
            horizon=1,
            frequency_unit="d",
            frequency_quantity=1,
            time_col="ds",
            split_col="split",
            preprocess_func=preprocess_func_with_category_encoding
        )
        
        # Test data with categorical column
        test_df = pd.DataFrame({
            "ds": [pd.to_datetime("2020-11-01"), pd.to_datetime("2020-11-04")],
            "category_col": ["A", "B"],
            "split": ["train", "test"]
        })
        
        # This should work without errors if category_encoders is properly available
        yhat = prophet_model.predict(None, test_df)
        self.assertEqual(2, len(yhat))

    def test_multiseries_model_with_category_encoding_preprocessing(self):
        """Test that multi-series models work with category encoding preprocessing"""
        import category_encoders as ce
        
        def preprocess_func_with_category_encoding(df):
            """Preprocessing function that uses category_encoders for multi-series"""
            if 'category_col' in df.columns:
                # Use target encoder which is commonly used in multi-series scenarios
                encoder = ce.TargetEncoder(cols=['category_col'])
                # For this test, we'll just transform without fitting since we don't have a real target
                df = df.copy()
                df['category_col'] = df['category_col'].astype('category').cat.codes
            return df
        
        multi_series_model_json = {("1",): self.model_json, ("2",): self.model_json}
        multi_series_start = {
            ("1",): pd.Timestamp("2020-07-01"),
            ("2",): pd.Timestamp("2020-07-01"),
        }
        
        prophet_model = MultiSeriesProphetModel(
            model_json=multi_series_model_json,
            timeseries_starts=multi_series_start,
            timeseries_end="2020-07-25",
            horizon=1,
            frequency_unit="days",
            frequency_quantity=1,
            time_col="ds",
            id_cols=["id"],
            split_col="split",
            preprocess_func=preprocess_func_with_category_encoding
        )
        
        test_df = pd.DataFrame({
            "ds": [pd.to_datetime("2020-11-01"), pd.to_datetime("2020-11-02")],
            "id": ["1", "2"],
            "category_col": ["X", "Y"],
            "split": ["train", "test"]
        })
        
        # This should work without errors if category_encoders is properly available
        yhat = prophet_model.predict(None, test_df)
        self.assertEqual(2, len(yhat))

    def test_category_encoders_version_compatibility(self):
        """Test that the correct version of category_encoders is specified in dependencies"""
        # Verify that category_encoders is in PROPHET_ADDITIONAL_PIP_DEPS
        category_encoders_deps = [dep for dep in PROPHET_ADDITIONAL_PIP_DEPS if "category_encoders" in dep]
        self.assertEqual(len(category_encoders_deps), 1, "category_encoders should be in PROPHET_ADDITIONAL_PIP_DEPS")
        
        # Verify the format includes version specification
        category_encoders_dep = category_encoders_deps[0]
        self.assertIn("==", category_encoders_dep, "category_encoders dependency should specify exact version")
        
        # Verify it matches the currently installed version
        import category_encoders
        expected_dep = f"category_encoders=={category_encoders.__version__}"
        self.assertEqual(category_encoders_dep, expected_dep, 
                        f"Dependency should match installed version: {expected_dep}")

    def test_model_environment_includes_category_encoders(self):
        """Test that the model environment includes category_encoders"""
        prophet_model = ProphetModel(self.model_json, 1, "d", 1, "ds")
        
        # Get the model environment
        model_env = prophet_model.model_env
        
        # Navigate to pip dependencies: dependencies list -> find dict with 'pip' key -> get pip list
        dependencies = model_env.get('dependencies', [])
        pip_deps = []
        for dep in dependencies:
            if isinstance(dep, dict) and 'pip' in dep:
                pip_deps = dep['pip']
                break
        
        category_encoders_found = any("category_encoders" in dep for dep in pip_deps)
        self.assertTrue(category_encoders_found, "category_encoders should be in model environment pip dependencies")