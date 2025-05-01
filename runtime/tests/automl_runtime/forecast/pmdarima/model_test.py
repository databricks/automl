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

import unittest
import pickle
import datetime
from unittest import mock

import mlflow
import pytest
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ErrorCode, INVALID_PARAMETER_VALUE
from pmdarima.arima import ARIMA

from databricks.automl_runtime.forecast.pmdarima.model import (
    ArimaModel, 
    MultiSeriesArimaModel, 
    AbstractArimaModel,
    mlflow_arima_log_model, 
    ARIMA_ADDITIONAL_PIP_DEPS,
)


class TestArimaModel(unittest.TestCase):

    def setUp(self) -> None:
        self.num_rows = 9
        self.start_ds = pd.Timestamp("2020-10-01")
        self.horizon = 1
        self.freq = 'W'
        self.frequency_quantity=1
        dates = AbstractArimaModel._get_ds_indices(self.start_ds, periods=self.num_rows, frequency_unit=self.freq, frequency_quantity=self.frequency_quantity)
        self.df = pd.concat([
            pd.Series(dates, name='date'),
            pd.Series(range(self.num_rows), name="y")
        ], axis=1)
        model = ARIMA(order=(2, 0, 2), suppress_warnings=True)
        model.fit(self.df.set_index("date"))
        pickled_model = pickle.dumps(model)
        self.arima_model = ArimaModel(pickled_model,
                                      horizon=self.horizon,
                                      frequency_unit=self.freq,
                                      frequency_quantity=self.frequency_quantity,
                                      start_ds=self.start_ds,
                                      end_ds=pd.Timestamp("2020-11-26"),
                                      time_col="date")

    def test_make_future_dataframe(self):
        future_df = self.arima_model.make_future_dataframe(include_history=False)
        self.assertCountEqual(future_df.columns, {"ds"})
        self.assertEqual(1, future_df.shape[0])

    def test_predict_timeseries_success(self):
        forecast_pd = self.arima_model.predict_timeseries()
        expected_columns = {"yhat", "yhat_lower", "yhat_upper"}
        expected_ds = AbstractArimaModel._get_ds_indices(
            self.start_ds,
            periods=self.num_rows + self.horizon,
            frequency_unit=self.freq,
            frequency_quantity=self.frequency_quantity)
        self.assertTrue(expected_columns.issubset(set(forecast_pd.columns)))
        self.assertEqual(10, forecast_pd.shape[0])
        pd.testing.assert_series_equal(pd.Series(expected_ds, name='ds'), forecast_pd["ds"])
        # Test forecast without history data
        forecast_future_pd = self.arima_model.predict_timeseries(include_history=False)
        self.assertEqual(len(forecast_future_pd), self.horizon)

    def test_predict_success(self):
        test_df = pd.DataFrame({
            "date": [pd.to_datetime("2020-10-08"), pd.to_datetime("2020-12-10")]
        })
        expected_test_df = test_df.copy()
        yhat = self.arima_model.predict(context=None, model_input=test_df)
        self.assertEqual(2, len(yhat))
        pd.testing.assert_frame_equal(test_df, expected_test_df)  # check the input dataframe is unchanged

    def test_predict_success_datetime_date(self):
        test_df = pd.DataFrame({
            "date": [datetime.date(2020, 10, 8), datetime.date(2020, 12, 10)]
        })
        expected_test_df = test_df.copy()
        yhat = self.arima_model.predict(context=None, model_input=test_df)
        self.assertEqual(2, len(yhat))
        pd.testing.assert_frame_equal(test_df, expected_test_df)  # check the input dataframe is unchanged

    def test_predict_success_string(self):
        test_df = pd.DataFrame({
            "date": ["2020-10-08", "2020-12-10"]
        })
        expected_test_df = test_df.copy()
        yhat = self.arima_model.predict(context=None, model_input=test_df)
        self.assertEqual(2, len(yhat))
        pd.testing.assert_frame_equal(test_df, expected_test_df)  # check the input dataframe is unchanged

    def test_predict_failure_unmatched_frequency(self):
        test_df = pd.DataFrame({
            "date": [pd.to_datetime("2020-10-08"), pd.to_datetime("2020-12-10"), pd.to_datetime("2020-11-06")]
        })
        with pytest.raises(MlflowException, match="includes different frequency") as e:
            self.arima_model.predict(context=None, model_input=test_df)
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    def test_predict_failure_invalid_time_range(self):
        test_df = pd.DataFrame({
            "date": [pd.to_datetime("2020-09-24"), pd.to_datetime("2020-10-08")]
        })
        with pytest.raises(MlflowException, match="includes time earlier than the history data that the model was "
                                                  "trained on") as e:
            self.arima_model.predict(context=None, model_input=test_df)
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    def test_predict_failure_invalid_time_col_name(self):
        test_df = pd.DataFrame({
            "invalid_time_col_name": [pd.to_datetime("2020-10-08"), pd.to_datetime("2020-12-10")]
        })
        with pytest.raises(MlflowException, match="Input data columns") as e:
            self.arima_model.predict(context=None, model_input=test_df)
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


class TestArimaModelDate(unittest.TestCase):

    def setUp(self) -> None:
        self.num_rows = 9
        self.start_ds = datetime.date(2020, 10, 1)
        self.horizon = 1
        self.freq = 'W'
        self.frequency_quantity = 1
        dates = AbstractArimaModel._get_ds_indices(
            pd.to_datetime(self.start_ds), periods=self.num_rows, frequency_unit=self.freq, frequency_quantity=self.frequency_quantity)
        self.df = pd.concat([
            pd.Series(dates, name='date'),
            pd.Series(range(self.num_rows), name="y")
        ], axis=1)
        model = ARIMA(order=(2, 0, 2), suppress_warnings=True)
        model.fit(self.df.set_index("date"))
        pickled_model = pickle.dumps(model)
        self.arima_model = ArimaModel(pickled_model,
                                      horizon=self.horizon,
                                      frequency_unit=self.freq,
                                      frequency_quantity=self.frequency_quantity,
                                      start_ds=self.start_ds,
                                      end_ds=pd.Timestamp("2020-11-26"),
                                      time_col="date")

    def test_predict_success(self):
        test_df = pd.DataFrame({
            "date": [pd.to_datetime("2020-10-08"), pd.to_datetime("2020-12-10")]
        })
        expected_test_df = test_df.copy()
        yhat = self.arima_model.predict(context=None, model_input=test_df)
        self.assertEqual(2, len(yhat))
        pd.testing.assert_frame_equal(test_df, expected_test_df)  # check the input dataframe is unchanged


class TestArimaModelWithExogenous(unittest.TestCase):

    def setUp(self) -> None:
        self.num_rows = 10
        self.start_ds = pd.Timestamp("2020-10-01")
        self.horizon = 1
        self.freq = 'W'
        self.frequency_quantity = 1
        dates = AbstractArimaModel._get_ds_indices(self.start_ds, periods=self.num_rows, frequency_unit=self.freq, frequency_quantity=self.frequency_quantity)
        self.df = pd.concat([
            pd.Series(dates, name='date'),
            pd.Series(range(self.num_rows), name="y"),
            pd.Series(range(self.num_rows), name="x1"),
            pd.Series(range(self.num_rows), name="x2")
        ], axis=1)
        train_df = self.df.set_index("date")
        self.X = train_df.drop(["y"], axis=1)
        self.exogenous_cols = ["x1", "x2"]
        model = ARIMA(order=(2, 0, 2), suppress_warnings=True)
        model.fit(train_df[["y"]], X=self.X)
        pickled_model = pickle.dumps(model)
        self.arima_model = ArimaModel(pickled_model,
                                      horizon=self.horizon,
                                      frequency_unit=self.freq,
                                      frequency_quantity=self.frequency_quantity,
                                      start_ds=self.start_ds,
                                      end_ds=pd.Timestamp("2020-11-26"),
                                      time_col="date",
                                      exogenous_cols=self.exogenous_cols)

    def test_predict_timeseries_success(self):
        forecast_pd = self.arima_model.predict_timeseries(future_df=self.df)
        expected_columns = {"yhat", "yhat_lower", "yhat_upper"}
        self.assertTrue(expected_columns.issubset(set(forecast_pd.columns)))
        self.assertEqual(10, forecast_pd.shape[0])
        pd.testing.assert_series_equal(self.df["date"], forecast_pd["ds"], check_names=False)
        # Test forecast without history data
        forecast_future_pd = self.arima_model.predict_timeseries(include_history=False, future_df=self.df)
        self.assertEqual(len(forecast_future_pd), self.horizon)

    def test_predict_success(self):
        test_df = pd.DataFrame({
            "date": [pd.to_datetime("2020-10-08"), pd.to_datetime("2020-12-3"), pd.to_datetime("2020-12-10")],
            "x1": [1, 2, 3],
            "x2": [4, 5, 6]
        })
        expected_test_df = test_df.copy()
        yhat = self.arima_model.predict(context=None, model_input=test_df)
        self.assertEqual(3, len(yhat))
        pd.testing.assert_frame_equal(test_df, expected_test_df)  # check the input dataframe is unchanged


class TestMultiSeriesArimaModel(unittest.TestCase):

    def setUp(self) -> None:
        num_rows = 9
        self.df = pd.concat([
            pd.to_datetime(pd.Series(range(num_rows), name="date").apply(lambda i: f"2020-{i + 1:02d}-13")),
            pd.Series(range(num_rows), name="y")
        ], axis=1)
        model = ARIMA(order=(2, 0, 2), suppress_warnings=True)
        model.fit(self.df.set_index("date"))
        self.pickled_model = pickle.dumps(model)
        pickled_model_dict = {("1",): self.pickled_model, ("2",): self.pickled_model}
        start_ds_dict = {("1",): pd.Timestamp("2020-01-13"), ("2",): pd.Timestamp("2020-01-13")}
        end_ds_dict = {("1",): pd.Timestamp("2020-09-13"), ("2",): pd.Timestamp("2020-09-13")}
        self.arima_model = MultiSeriesArimaModel(pickled_model_dict,
                                                 horizon=1,
                                                 frequency_unit='month',
                                                 frequency_quantity=1,
                                                 start_ds_dict=start_ds_dict,
                                                 end_ds_dict=end_ds_dict,
                                                 time_col="date",
                                                 id_cols=["id"])

    def test_predict_timeseries_success(self):
        forecast_pd = self.arima_model.predict_timeseries()
        expected_columns = {"id", "ds", "yhat", "yhat_lower", "yhat_upper"}
        self.assertCountEqual(expected_columns, set(forecast_pd.columns))
        self.assertEqual(20, forecast_pd.shape[0])
        # Test forecast without history data
        forecast_future_pd = self.arima_model.predict_timeseries(include_history=False)
        self.assertEqual(len(forecast_future_pd), 2)

    def test_predict_success(self):
        test_df = pd.DataFrame({
            "date": [pd.to_datetime("2020-05-13"), pd.to_datetime("2020-05-13"),
                     pd.to_datetime("2020-12-13"), pd.to_datetime("2020-12-13")],
            "id": ["1", "2", "1", "2"],
        })
        expected_test_df = test_df.copy()
        yhat = self.arima_model.predict(context=None, model_input=test_df)
        self.assertEqual(4, len(yhat))
        pd.testing.assert_frame_equal(test_df, expected_test_df)  # check the input dataframe is unchanged

    def test_predict_success_one_row(self):
        test_df = pd.DataFrame({"date": [pd.to_datetime("2020-11-13")], "id": ["1"]})
        yhat = self.arima_model.predict(context=None, model_input=test_df)
        self.assertEqual(1, len(yhat))

    def test_predict_fail_unseen_id(self):
        test_df = pd.DataFrame({
            "date": [pd.to_datetime("2020-10-13"), pd.to_datetime("2020-10-13"),
                     pd.to_datetime("2020-11-13"), pd.to_datetime("2020-11-13")],
            "id": ["1", "2", "1", "3"],
        })
        with pytest.raises(MlflowException, match="includes unseen values in id columns") as e:
            self.arima_model.predict(context=None, model_input=test_df)
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    def test_predict_failure_unmatched_frequency(self):
        test_df = pd.DataFrame({
            "date": [pd.to_datetime("2020-10-05"), pd.to_datetime("2020-10-05 12:30"),
                     pd.to_datetime("2020-11-04"), pd.to_datetime("2020-11-04")],
            "id": ["1", "2", "1", "2"],
        })
        with pytest.raises(MlflowException, match="includes different frequency") as e:
            self.arima_model.predict(context=None, model_input=test_df)
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    def test_predict_failure_invalid_time_range(self):
        test_df = pd.DataFrame({
            "date": [pd.to_datetime("2020-10-13"), pd.to_datetime("2000-10-13"),
                     pd.to_datetime("2020-11-13"), pd.to_datetime("2020-11-13")],
            "id": ["1", "2", "1", "2"],
        })
        with pytest.raises(MlflowException, match="includes time earlier than the history data that the model was "
                                                  "trained on") as e:
            self.arima_model.predict(context=None, model_input=test_df)
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    def test_predict_failure_invalid_time_col_name(self):
        test_df = pd.DataFrame({
            "time": [pd.to_datetime("2020-05-13"), pd.to_datetime("2000-05-13"),
                     pd.to_datetime("2020-11-13"), pd.to_datetime("2020-11-13")],
            "invalid_id_col_name": ["1", "2", "1", "2"],
        })
        with pytest.raises(MlflowException, match="Input data columns") as e:
            self.arima_model.predict(context=None, model_input=test_df)
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    def test_make_future_dataframe(self):
        future_df = self.arima_model.make_future_dataframe(include_history=False)
        self.assertCountEqual(future_df.columns, {"ds", "id"})
        self.assertEqual(2, future_df.shape[0])

    def test_make_future_dataframe_multi_ids(self):
        pickled_model_dict = {(1, "1"): self.pickled_model, (2, "1"): self.pickled_model}
        start_ds_dict = {(1, "1"): pd.Timestamp("2020-01-13"), (2, "1"): pd.Timestamp("2020-01-13")}
        end_ds_dict = {(1, "1"): pd.Timestamp("2020-09-13"), (2, "1"): pd.Timestamp("2020-09-13")}
        arima_model = MultiSeriesArimaModel(pickled_model_dict,
                                            horizon=1,
                                            frequency_unit='month',
                                            frequency_quantity=1,
                                            start_ds_dict=start_ds_dict,
                                            end_ds_dict=end_ds_dict,
                                            time_col="date",
                                            id_cols=["id1", "id2"])
        future_df = arima_model.make_future_dataframe(include_history=False)
        self.assertCountEqual(future_df.columns, {"ds", "id1", "id2"})
        # Make sure keep the column types for identity columns
        self.assertTrue(future_df.dtypes["id1"] == "int")
        self.assertTrue(future_df.dtypes["id2"] == "object")
        self.assertEqual(2, future_df.shape[0])

    def test_make_future_dataframe_invalid_group(self):
        with pytest.raises(ValueError, match="Invalid groups:"):
            future_df = self.arima_model.make_future_dataframe(groups=[(1,)])



class TestMultiSeriesArimaModelWithExogenous(unittest.TestCase):

    def setUp(self) -> None:
        num_rows = 10
        self.df = pd.concat([
            pd.to_datetime(pd.Series(range(num_rows), name="date").apply(lambda i: f"2020-{i + 1:02d}-13")),
            pd.Series(range(num_rows), name="y"),
            pd.Series(range(num_rows), name="x1"),
            pd.Series(range(num_rows), name="x2"),
            pd.Series(["1" if i < 5 else "2" for i in range(num_rows)], name="id")  # Add id column with different values
        ], axis=1)
        train_df = self.df.set_index("date")
        self.exogenous_cols = ["x1", "x2"]
        self.X = train_df[self.exogenous_cols]

        model = ARIMA(order=(2, 1, 2), suppress_warnings=True)
        model.fit(train_df[["y"]], X=self.X)
        pickled_model = pickle.dumps(model)
        pickled_model_dict = {("1",): pickled_model, ("2",): pickled_model}
        start_ds_dict = {("1",): pd.Timestamp("2020-01-13"), ("2",): pd.Timestamp("2020-01-13")}
        end_ds_dict = {("1",): pd.Timestamp("2020-09-13"), ("2",): pd.Timestamp("2020-09-13")}
        self.arima_model = MultiSeriesArimaModel(pickled_model_dict,
                                                 horizon=1,
                                                 frequency_unit='month',
                                                 frequency_quantity=1,
                                                 start_ds_dict=start_ds_dict,
                                                 end_ds_dict=end_ds_dict,
                                                 time_col="date",
                                                 id_cols=["id"],
                                                 exogenous_cols=self.exogenous_cols)

    def test_predict_timeseries_success(self):
        forecast_pd = self.arima_model.predict_timeseries(future_df=self.df)
        expected_columns = {"yhat", "yhat_lower", "yhat_upper"}
        self.assertTrue(expected_columns.issubset(set(forecast_pd.columns)))
        self.assertEqual(18, forecast_pd.shape[0])
        # Test forecast without history data
        forecast_future_pd = self.arima_model.predict_timeseries(include_history=False, future_df=self.df)
        self.assertEqual(len(forecast_future_pd), 2)

    def test_predict_success(self):
        test_df = pd.DataFrame({
            "date": [pd.to_datetime("2020-05-13"), pd.to_datetime("2020-05-13"),
                     pd.to_datetime("2020-10-13"), pd.to_datetime("2020-10-13"),
                     pd.to_datetime("2020-11-13"), pd.to_datetime("2020-11-13"),
                     pd.to_datetime("2020-12-13"), pd.to_datetime("2020-12-13")],
            "id": ["1", "2", "1", "2", "1", "2", "1", "2"],
            "x1": [1, 1, 2, 2, 4, 4, 8, 8],
            "x2": [1, 1, 2, 2, 4, 4, 8, 8]
        })
        expected_test_df = test_df.copy()
        yhat = self.arima_model.predict(context=None, model_input=test_df)
        self.assertEqual(8, len(yhat))
        pd.testing.assert_frame_equal(test_df, expected_test_df)  # check the input dataframe is unchanged


class TestAbstractArimaModel(unittest.TestCase):

    def test_validate_cols_success(self):
        test_df = pd.DataFrame({"date": []})
        AbstractArimaModel._validate_cols(test_df, ["date"])

    def test_validate_cols_invalid_id_col_name(self):
        test_df = pd.DataFrame({"date": [], "invalid_id_col_name": [], })
        with pytest.raises(MlflowException, match="Input data columns") as e:
            AbstractArimaModel._validate_cols(test_df, ["date", "id"])
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    def test_get_ds_weekly(self):
        expected_ds = pd.to_datetime(
            ['2022-01-01 12:30:00', '2022-01-08 12:30:00',
             '2022-01-15 12:30:00', '2022-01-22 12:30:00',
             '2022-01-29 12:30:00', '2022-02-05 12:30:00',
             '2022-02-12 12:30:00', '2022-02-19 12:30:00']
        )
        ds_indices = AbstractArimaModel._get_ds_indices(
            start_ds=pd.Timestamp("2022-01-01 12:30"),
            periods=8,
            frequency_unit='W',
            frequency_quantity=1)
        pd.testing.assert_index_equal(expected_ds, ds_indices)

    def test_get_ds_hourly(self):
        expected_ds = pd.to_datetime(
            ['2021-12-10 09:23:00', '2021-12-10 10:23:00',
             '2021-12-10 11:23:00', '2021-12-10 12:23:00',
             '2021-12-10 13:23:00', '2021-12-10 14:23:00',
             '2021-12-10 15:23:00', '2021-12-10 16:23:00',
             '2021-12-10 17:23:00', '2021-12-10 18:23:00']
        )
        ds_indices = AbstractArimaModel._get_ds_indices(
            start_ds=pd.Timestamp("2021-12-10 09:23"),
            periods=10,
            frequency_unit='H',
            frequency_quantity=1)
        pd.testing.assert_index_equal(expected_ds, ds_indices)


class TestLogModel(unittest.TestCase):

    def setUp(self) -> None:
        num_rows = 9
        self.df = pd.concat([
            pd.to_datetime(pd.Series(range(num_rows), name="date").apply(lambda i: f"2020-10-{i + 1}")),
            pd.Series(range(num_rows), name="y")
        ], axis=1)
        model = ARIMA(order=(2, 0, 2), suppress_warnings=True)
        model.fit(self.df.set_index("date"))
        self.pickled_model = pickle.dumps(model)

    def test_mlflow_arima_log_model(self):
        arima_model = ArimaModel(self.pickled_model, horizon=1, frequency_unit='d', frequency_quantity=1,
                                 start_ds=pd.to_datetime("2020-10-01"), end_ds=pd.to_datetime("2020-10-09"),
                                 time_col="date")
        with mlflow.start_run() as run:
            mlflow_arima_log_model(arima_model)

        # Load the saved model from mlflow
        run_id = run.info.run_id

        # Check additonal requirements logged correctly
        self._check_requirements(run_id)

        # Load the model
        loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

        # Make sure can make forecasts with the saved model
        loaded_model.predict(self.df.drop("y", axis=1))
        loaded_model._model_impl.python_model.predict_timeseries()

    def test_mlflow_arima_log_model_multiseries(self):
        pickled_model_dict = {("1",): self.pickled_model, ("2",): self.pickled_model}
        start_ds_dict = {("1",): pd.Timestamp("2020-10-01"), ("2",): pd.Timestamp("2020-10-01")}
        end_ds_dict = {("1",): pd.Timestamp("2020-10-09"), ("2",): pd.Timestamp("2020-10-09")}
        multiseries_arima_model = MultiSeriesArimaModel(pickled_model_dict,
                                                        horizon=1,
                                                        frequency_unit='d',
                                                        frequency_quantity=1,
                                                        start_ds_dict=start_ds_dict,
                                                        end_ds_dict=end_ds_dict,
                                                        time_col="date",
                                                        id_cols=["id"])
        with mlflow.start_run() as run:
            mlflow_arima_log_model(multiseries_arima_model)

        # Load the saved model from mlflow
        run_id = run.info.run_id

        # Check additonal requirements logged correctly
        self._check_requirements(run_id)
        
        # Load the model
        loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

        # Make sure can make forecasts with the saved model
        loaded_model._model_impl.python_model.predict_timeseries()
        test_df = pd.DataFrame({
            "date": [pd.to_datetime("2020-10-05"), pd.to_datetime("2020-10-05"),
                     pd.to_datetime("2020-11-04"), pd.to_datetime("2020-11-04")],
            "id": ["1", "2", "1", "2"],
        })
        loaded_model.predict(test_df)

        # Make sure can make forecasts for one-row dataframe
        loaded_model.predict(test_df[0:1])

    def _check_requirements(self, run_id: str):
        # read requirements.txt from the run
        requirements_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/model/requirements.txt")
        with open(requirements_path, "r") as f:
            requirements = f.read()
        # check if all additional dependencies are logged
        for dependency in ARIMA_ADDITIONAL_PIP_DEPS:
            self.assertIn(dependency, requirements, f"requirements.txt should contain {dependency} but got {requirements}")

class TestArimaModelFrequencyQuantity(unittest.TestCase):

    def setUp(self) -> None:
        self.num_rows = 9
        self.start_ds = pd.Timestamp("2020-10-01")
        self.horizon = 1
        self.freq = 'min'
        frequency_quantities = [1, 5, 10, 15, 30]
        self.quantity_model_pairs = []

        for frequency_quantity in frequency_quantities:
            dates = AbstractArimaModel._get_ds_indices(self.start_ds, periods=self.num_rows, frequency_unit=self.freq, frequency_quantity=frequency_quantity)
            df = pd.concat([
                pd.Series(dates, name='date'),
                pd.Series(range(self.num_rows), name="y")
            ], axis=1)
            model = ARIMA(order=(2, 0, 2), suppress_warnings=True)
            model.fit(df.set_index("date"))
            pickled_model = pickle.dumps(model)
            self.quantity_model_pairs.append((frequency_quantity, ArimaModel(pickled_model,
                                      horizon=self.horizon,
                                      frequency_unit=self.freq,
                                      frequency_quantity=frequency_quantity,
                                      start_ds=self.start_ds,
                                      end_ds=dates.max(),
                                      time_col="date")))

    def test_make_future_dataframe(self):
        for frequency_quantity, arima_model in self.quantity_model_pairs:
            future_df = arima_model.make_future_dataframe(include_history=False)
            self.assertCountEqual(future_df.columns, {"ds"})
            self.assertEqual(1, future_df.shape[0])

    def test_predict_timeseries_success(self):
        for frequency_quantity, arima_model in self.quantity_model_pairs:
            forecast_pd = arima_model.predict_timeseries()
            expected_columns = {"yhat", "yhat_lower", "yhat_upper"}
            expected_ds = AbstractArimaModel._get_ds_indices(
                self.start_ds,
                periods=self.num_rows + self.horizon,
                frequency_unit=self.freq,
                frequency_quantity=frequency_quantity)
            self.assertTrue(expected_columns.issubset(set(forecast_pd.columns)))
            self.assertEqual(10, forecast_pd.shape[0])
            pd.testing.assert_series_equal(pd.Series(expected_ds, name='ds'), forecast_pd["ds"])
            # Test forecast without history data
            forecast_future_pd = arima_model.predict_timeseries(include_history=False)
            self.assertEqual(len(forecast_future_pd), self.horizon)

    def test_predict_success(self):
        for frequency_quantity, arima_model in self.quantity_model_pairs:
            test_df = pd.DataFrame({
                "date": [pd.to_datetime("2020-10-01") + self.num_rows*pd.DateOffset(minutes=frequency_quantity), 
                         pd.to_datetime("2020-10-01") + (self.num_rows+1)*pd.DateOffset(minutes=frequency_quantity)]
            })
            expected_test_df = test_df.copy()
            yhat = arima_model.predict(context=None, model_input=test_df)
            self.assertEqual(2, len(yhat))
            pd.testing.assert_frame_equal(test_df, expected_test_df)  # check the input dataframe is unchanged

    def test_predict_success_datetime_date(self):
        for _, arima_model in self.quantity_model_pairs:
            test_df = pd.DataFrame({
                "date": [datetime.datetime(2020, 10, 1, 6, 0, 0), datetime.datetime(2020, 10, 1, 6, 30, 0)]
            })
            expected_test_df = test_df.copy()
            yhat = arima_model.predict(context=None, model_input=test_df)
            self.assertEqual(2, len(yhat))
            pd.testing.assert_frame_equal(test_df, expected_test_df)  # check the input dataframe is unchanged

    def test_predict_success_string(self):
        for _, arima_model in self.quantity_model_pairs:
            test_df = pd.DataFrame({
                "date": ["2020-10-01 06:00:00", "2020-10-01 06:30:00"]
            })
            expected_test_df = test_df.copy()
            yhat = arima_model.predict(context=None, model_input=test_df)
            self.assertEqual(2, len(yhat))
            pd.testing.assert_frame_equal(test_df, expected_test_df)  # check the input dataframe is unchanged

    def test_predict_failure_unmatched_frequency(self):
        for frequency_quantity, arima_model in self.quantity_model_pairs:
            if frequency_quantity == 1: continue
            test_df = pd.DataFrame({
                "date": [pd.to_datetime("2020-10-01 00:00:00"), pd.to_datetime("2020-10-01 00:01:00"), pd.to_datetime("2020-10-01 00:04:00")]
            })
            with pytest.raises(MlflowException, match="includes different frequency") as e:
                arima_model.predict(context=None, model_input=test_df)
            assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    def test_predict_failure_invalid_time_range(self):
        for _, arima_model in self.quantity_model_pairs:
            test_df = pd.DataFrame({
                "date": [pd.to_datetime("2020-09-30 00:00:00"), pd.to_datetime("2020-10-01 00:01:00")]
            })
            with pytest.raises(MlflowException, match="includes time earlier than the history data that the model was "
                                                    "trained on") as e:
                arima_model.predict(context=None, model_input=test_df)
            assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    def test_predict_failure_invalid_time_col_name(self):
        for _, arima_model in self.quantity_model_pairs:
            test_df = pd.DataFrame({
                "invalid_time_col_name": [pd.to_datetime("2020-10-08"), pd.to_datetime("2020-12-10")]
            })
            with pytest.raises(MlflowException, match="Input data columns") as e:
                arima_model.predict(context=None, model_input=test_df)
            assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

class TestArimaModelWithPreprocess(unittest.TestCase):
    def setUp(self) -> None:
        self.num_rows = 9
        self.start_ds = pd.Timestamp("2020-10-01")
        self.horizon = 1
        self.freq = 'W'
        self.frequency_quantity = 1
        dates = AbstractArimaModel._get_ds_indices(self.start_ds, periods=self.num_rows, frequency_unit=self.freq, frequency_quantity=self.frequency_quantity)
        self.df = pd.concat([
            pd.Series(dates, name='date'),
            pd.Series(range(self.num_rows), name="y"),
            pd.Series(range(self.num_rows), name="x1"),
            pd.Series(range(self.num_rows), name="x2")
        ], axis=1)
        model = ARIMA(order=(2, 0, 2), suppress_warnings=True)
        model.fit(self.df[["y", "date"]].set_index("date"), exogenous=self.df[["x1", "x2"]])
        pickled_model = pickle.dumps(model)

        # Create a mock preprocess function that doubles y values
        def preprocess_func(df):
            df = df.copy()
            df["y"] = df["y"] * 2
            return df
        
        self.mock_preprocess = mock.Mock(side_effect=preprocess_func)

        self.arima_model = ArimaModel(pickled_model,
                                     horizon=self.horizon,
                                     frequency_unit=self.freq,
                                     frequency_quantity=self.frequency_quantity,
                                     start_ds=self.start_ds,
                                     end_ds=pd.Timestamp("2020-11-26"),
                                     time_col="date",
                                     exogenous_cols=["x1", "x2"],
                                     split_col="split",
                                     preprocess_func=self.mock_preprocess)

    def test_predict_timeseries_with_preprocess(self):
        future_df = self.df.copy()
        future_df = pd.concat([future_df, pd.DataFrame({
            "date": [pd.to_datetime("2020-12-17"), pd.to_datetime("2020-12-24")],
            "x1": [1, 2],
            "x2": [3, 4]
        })], axis=0)
        future_df["split"] = "prediction"
        
        forecast_pd = self.arima_model.predict_timeseries(future_df=future_df)
        expected_columns = {"yhat", "yhat_lower", "yhat_upper"}
        self.assertTrue(expected_columns.issubset(set(forecast_pd.columns)))
        self.assertEqual(11, forecast_pd.shape[0])
        
        # Verify that preprocess_func was called with the correct argument
        self.mock_preprocess.assert_called_once()
        call_arg = self.mock_preprocess.call_args[0][0]
        
        # Verify the structure of the dataframe passed to preprocess_func
        expected_call = future_df.copy()
        expected_call["y"] = None
        expected_call["split"] = "prediction"
        pd.testing.assert_frame_equal(call_arg, expected_call)
        
        # Verify the return value from preprocess_func
        # The preprocess function doubles y values
        expected_return = expected_call.copy()
        expected_return["y"] = expected_return["y"] * 2 if expected_return["y"] is not None else 0
        
        # Get the actual return value from the call
        actual_return = self.mock_preprocess.side_effect(call_arg)
        pd.testing.assert_frame_equal(actual_return, expected_return)

    def test_predict_with_preprocess(self):
        test_df = pd.DataFrame({
            "date": [pd.to_datetime("2020-12-17"), pd.to_datetime("2020-12-24")],
            "x1": [1, 2],
            "x2": [3, 4]
        })
        
        yhat = self.arima_model.predict(context=None, model_input=test_df)
        self.assertEqual(2, len(yhat))
        
        # Verify that preprocess_func was called with the correct argument
        self.mock_preprocess.assert_called_once()
        call_arg = self.mock_preprocess.call_args[0][0]
        
        # Verify the structure of the dataframe passed to preprocess_func
        expected_call = test_df.copy()
        expected_call["y"] = None
        expected_call["split"] = "prediction"
        pd.testing.assert_frame_equal(call_arg, expected_call)
        
        # Verify the return value from preprocess_func
        # The preprocess function doubles y values
        expected_return = expected_call.copy()
        expected_return["y"] = expected_return["y"] * 2 if expected_return["y"] is not None else 0
        
        # Get the actual return value from the call
        actual_return = self.mock_preprocess.side_effect(call_arg)
        pd.testing.assert_frame_equal(actual_return, expected_return)

class TestMultiSeriesArimaModelWithPreprocess(unittest.TestCase):
    def setUp(self) -> None:
        num_rows = 9
        self.df = pd.concat([
            pd.to_datetime(pd.Series(range(num_rows), name="date").apply(lambda i: f"2020-{i + 1:02d}-13")),
            pd.Series(range(num_rows), name="y")
        ], axis=1)
        model = ARIMA(order=(2, 0, 2), suppress_warnings=True)
        model.fit(self.df.set_index("date"))
        self.pickled_model = pickle.dumps(model)
        pickled_model_dict = {("1",): self.pickled_model, ("2",): self.pickled_model}
        start_ds_dict = {("1",): pd.Timestamp("2020-01-13"), ("2",): pd.Timestamp("2020-01-13")}
        end_ds_dict = {("1",): pd.Timestamp("2020-09-13"), ("2",): pd.Timestamp("2020-09-13")}

        # Create a mock preprocess function that adds 1 to y values
        def preprocess_func(df):
            df = df.copy()
            df["y"] = df["y"] + 1
            return df
        
        self.mock_preprocess = mock.Mock(side_effect=preprocess_func)

        self.arima_model = MultiSeriesArimaModel(pickled_model_dict,
                                                horizon=1,
                                                frequency_unit='month',
                                                frequency_quantity=1,
                                                start_ds_dict=start_ds_dict,
                                                end_ds_dict=end_ds_dict,
                                                time_col="date",
                                                id_cols=["id"],
                                                split_col="split",
                                                preprocess_func=self.mock_preprocess)

    def test_predict_with_preprocess(self):
        test_df = pd.DataFrame({
            "date": [pd.to_datetime("2020-05-13"), pd.to_datetime("2020-05-13"),
                     pd.to_datetime("2020-12-13"), pd.to_datetime("2020-12-13")],
            "id": ["1", "2", "1", "2"]
        })
        
        yhat = self.arima_model.predict(context=None, model_input=test_df)
        self.assertEqual(4, len(yhat))
        
        # Verify that preprocess_func was called three times:
        # 1. In predict() for the entire dataframe
        # 2. For each time series (id=1 and id=2)
        self.assertEqual(len(self.mock_preprocess.call_args_list), 3)
        
        # First call: entire dataframe
        first_call_arg = self.mock_preprocess.call_args_list[0][0][0]
        expected_first_call = test_df.copy()
        expected_first_call["ts_id"] = expected_first_call[["id"]].apply(tuple, axis=1)
        expected_first_call["y"] = None
        expected_first_call["split"] = "prediction"
        pd.testing.assert_frame_equal(first_call_arg, expected_first_call)
        
        # Verify the return value from preprocess_func
        # The preprocess function adds 1 to y values
        expected_return = expected_first_call.copy()
        expected_return["y"] = expected_return["y"] + 1 if expected_return["y"] is not None else 1
        
        # Get the actual return value from the first call
        actual_return = self.mock_preprocess.side_effect(first_call_arg)
        pd.testing.assert_frame_equal(actual_return, expected_return)
