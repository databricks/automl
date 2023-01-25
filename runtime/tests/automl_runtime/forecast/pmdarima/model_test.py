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

import mlflow
import pytest
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ErrorCode, INVALID_PARAMETER_VALUE
from pmdarima.arima import ARIMA

from databricks.automl_runtime.forecast.pmdarima.model import ArimaModel, MultiSeriesArimaModel, AbstractArimaModel, \
    mlflow_arima_log_model


class TestArimaModel(unittest.TestCase):

    def setUp(self) -> None:
        self.num_rows = 9
        self.start_ds = pd.Timestamp("2020-10-01")
        self.horizon = 1
        self.freq = 'W'
        dates = AbstractArimaModel._get_ds_indices(self.start_ds,
                                                   periods=self.num_rows,
                                                   frequency=self.freq)
        self.X = pd.concat([
            pd.Series(dates, name='date'),
            pd.Series(range(self.num_rows), name="y")
        ],
                           axis=1)
        model = ARIMA(order=(2, 0, 2), suppress_warnings=True)
        model.fit(self.X.set_index("date"))
        pickled_model = pickle.dumps(model)
        self.arima_model = ArimaModel(pickled_model,
                                      horizon=self.horizon,
                                      frequency=self.freq,
                                      start_ds=self.start_ds,
                                      end_ds=pd.Timestamp("2020-11-26"),
                                      time_col="date")

    def test_predict_timeseries_success(self):
        forecast_pd = self.arima_model.predict_timeseries()
        expected_columns = {"yhat", "yhat_lower", "yhat_upper"}
        expected_ds = AbstractArimaModel._get_ds_indices(
            self.start_ds,
            periods=self.num_rows + self.horizon,
            frequency=self.freq)
        self.assertTrue(expected_columns.issubset(set(forecast_pd.columns)))
        self.assertEqual(10, forecast_pd.shape[0])
        pd.testing.assert_series_equal(pd.Series(expected_ds, name='ds'),
                                       forecast_pd["ds"])
        # Test forecast without history data
        forecast_future_pd = self.arima_model.predict_timeseries(
            include_history=False)
        self.assertEqual(len(forecast_future_pd), self.horizon)

    def test_predict_success(self):
        test_df = pd.DataFrame({
            "date":
            [pd.to_datetime("2020-10-08"),
             pd.to_datetime("2020-12-10")]
        })
        expected_test_df = test_df.copy()
        yhat = self.arima_model.predict(None, test_df)
        self.assertEqual(2, len(yhat))
        pd.testing.assert_frame_equal(
            test_df,
            expected_test_df)  # check the input dataframe is unchanged

    def test_predict_success_datetime_date(self):
        test_df = pd.DataFrame({
            "date": [datetime.date(2020, 10, 8),
                     datetime.date(2020, 12, 10)]
        })
        expected_test_df = test_df.copy()
        yhat = self.arima_model.predict(None, test_df)
        self.assertEqual(2, len(yhat))
        pd.testing.assert_frame_equal(
            test_df,
            expected_test_df)  # check the input dataframe is unchanged

    def test_predict_success_string(self):
        test_df = pd.DataFrame({"date": ["2020-10-08", "2020-12-10"]})
        expected_test_df = test_df.copy()
        yhat = self.arima_model.predict(None, test_df)
        self.assertEqual(2, len(yhat))
        pd.testing.assert_frame_equal(
            test_df,
            expected_test_df)  # check the input dataframe is unchanged

    def test_predict_failure_unmatched_frequency(self):
        test_df = pd.DataFrame({
            "date": [
                pd.to_datetime("2020-10-08"),
                pd.to_datetime("2020-12-10"),
                pd.to_datetime("2020-11-06")
            ]
        })
        with pytest.raises(MlflowException,
                           match="includes different frequency") as e:
            self.arima_model.predict(None, test_df)
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    def test_predict_failure_invalid_time_range(self):
        test_df = pd.DataFrame({
            "date":
            [pd.to_datetime("2020-09-24"),
             pd.to_datetime("2020-10-08")]
        })
        with pytest.raises(
                MlflowException,
                match=
                "includes time earlier than the history data that the model was "
                "trained on") as e:
            self.arima_model.predict(None, test_df)
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    def test_predict_failure_invalid_time_col_name(self):
        test_df = pd.DataFrame({
            "invalid_time_col_name":
            [pd.to_datetime("2020-10-08"),
             pd.to_datetime("2020-12-10")]
        })
        with pytest.raises(MlflowException, match="Input data columns") as e:
            self.arima_model.predict(None, test_df)
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


class TestArimaModelDate(unittest.TestCase):

    def setUp(self) -> None:
        self.num_rows = 9
        self.start_ds = datetime.date(2020, 10, 1)
        self.horizon = 1
        self.freq = 'W'
        dates = AbstractArimaModel._get_ds_indices(pd.to_datetime(
            self.start_ds),
                                                   periods=self.num_rows,
                                                   frequency=self.freq)
        self.X = pd.concat([
            pd.Series(dates, name='date'),
            pd.Series(range(self.num_rows), name="y")
        ],
                           axis=1)
        model = ARIMA(order=(2, 0, 2), suppress_warnings=True)
        model.fit(self.X.set_index("date"))
        pickled_model = pickle.dumps(model)
        self.arima_model = ArimaModel(pickled_model,
                                      horizon=self.horizon,
                                      frequency=self.freq,
                                      start_ds=self.start_ds,
                                      end_ds=pd.Timestamp("2020-11-26"),
                                      time_col="date")

    def test_predict_success(self):
        test_df = pd.DataFrame({
            "date":
            [pd.to_datetime("2020-10-08"),
             pd.to_datetime("2020-12-10")]
        })
        expected_test_df = test_df.copy()
        yhat = self.arima_model.predict(None, test_df)
        self.assertEqual(2, len(yhat))
        pd.testing.assert_frame_equal(
            test_df,
            expected_test_df)  # check the input dataframe is unchanged


class TestMultiSeriesArimaModel(unittest.TestCase):

    def setUp(self) -> None:
        num_rows = 9
        self.X = pd.concat([
            pd.to_datetime(
                pd.Series(
                    range(num_rows),
                    name="date").apply(lambda i: f"2020-{i + 1:02d}-13")),
            pd.Series(range(num_rows), name="y")
        ],
                           axis=1)
        model = ARIMA(order=(2, 0, 2), suppress_warnings=True)
        model.fit(self.X.set_index("date"))
        self.pickled_model = pickle.dumps(model)
        pickled_model_dict = {
            ("1", ): self.pickled_model,
            ("2", ): self.pickled_model
        }
        start_ds_dict = {
            ("1", ): pd.Timestamp("2020-01-13"),
            ("2", ): pd.Timestamp("2020-01-13")
        }
        end_ds_dict = {
            ("1", ): pd.Timestamp("2020-09-13"),
            ("2", ): pd.Timestamp("2020-09-13")
        }
        self.arima_model = MultiSeriesArimaModel(pickled_model_dict,
                                                 horizon=1,
                                                 frequency='month',
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
        forecast_future_pd = self.arima_model.predict_timeseries(
            include_history=False)
        self.assertEqual(len(forecast_future_pd), 2)

    def test_predict_success(self):
        test_df = pd.DataFrame({
            "date": [
                pd.to_datetime("2020-05-13"),
                pd.to_datetime("2020-05-13"),
                pd.to_datetime("2020-12-13"),
                pd.to_datetime("2020-12-13")
            ],
            "id": ["1", "2", "1", "2"],
        })
        expected_test_df = test_df.copy()
        yhat = self.arima_model.predict(None, test_df)
        self.assertEqual(4, len(yhat))
        pd.testing.assert_frame_equal(
            test_df,
            expected_test_df)  # check the input dataframe is unchanged

    def test_predict_success_one_row(self):
        test_df = pd.DataFrame({
            "date": [pd.to_datetime("2020-11-13")],
            "id": ["1"]
        })
        yhat = self.arima_model.predict(None, test_df)
        self.assertEqual(1, len(yhat))

    def test_predict_fail_unseen_id(self):
        test_df = pd.DataFrame({
            "date": [
                pd.to_datetime("2020-10-13"),
                pd.to_datetime("2020-10-13"),
                pd.to_datetime("2020-11-13"),
                pd.to_datetime("2020-11-13")
            ],
            "id": ["1", "2", "1", "3"],
        })
        with pytest.raises(MlflowException,
                           match="includes unseen values in id columns") as e:
            self.arima_model.predict(None, test_df)
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    def test_predict_failure_unmatched_frequency(self):
        test_df = pd.DataFrame({
            "date": [
                pd.to_datetime("2020-10-05"),
                pd.to_datetime("2020-10-05 12:30"),
                pd.to_datetime("2020-11-04"),
                pd.to_datetime("2020-11-04")
            ],
            "id": ["1", "2", "1", "2"],
        })
        with pytest.raises(MlflowException,
                           match="includes different frequency") as e:
            self.arima_model.predict(None, test_df)
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    def test_predict_failure_invalid_time_range(self):
        test_df = pd.DataFrame({
            "date": [
                pd.to_datetime("2020-10-13"),
                pd.to_datetime("2000-10-13"),
                pd.to_datetime("2020-11-13"),
                pd.to_datetime("2020-11-13")
            ],
            "id": ["1", "2", "1", "2"],
        })
        with pytest.raises(
                MlflowException,
                match=
                "includes time earlier than the history data that the model was "
                "trained on") as e:
            self.arima_model.predict(None, test_df)
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    def test_predict_failure_invalid_time_col_name(self):
        test_df = pd.DataFrame({
            "time": [
                pd.to_datetime("2020-05-13"),
                pd.to_datetime("2000-05-13"),
                pd.to_datetime("2020-11-13"),
                pd.to_datetime("2020-11-13")
            ],
            "invalid_id_col_name": ["1", "2", "1", "2"],
        })
        with pytest.raises(MlflowException, match="Input data columns") as e:
            self.arima_model.predict(None, test_df)
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    def test_make_future_dataframe(self):
        future_df = self.arima_model.make_future_dataframe(
            include_history=False)
        self.assertCountEqual(future_df.columns, {"ds", "id"})
        self.assertEqual(2, future_df.shape[0])

    def test_make_future_dataframe_multi_ids(self):
        pickled_model_dict = {
            (1, "1"): self.pickled_model,
            (2, "1"): self.pickled_model
        }
        start_ds_dict = {
            (1, "1"): pd.Timestamp("2020-01-13"),
            (2, "1"): pd.Timestamp("2020-01-13")
        }
        end_ds_dict = {
            (1, "1"): pd.Timestamp("2020-09-13"),
            (2, "1"): pd.Timestamp("2020-09-13")
        }
        arima_model = MultiSeriesArimaModel(pickled_model_dict,
                                            horizon=1,
                                            frequency='month',
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
            future_df = self.arima_model.make_future_dataframe(groups=[(1, )])


class TestAbstractArimaModel(unittest.TestCase):

    def test_validate_cols_success(self):
        test_df = pd.DataFrame({"date": []})
        AbstractArimaModel._validate_cols(test_df, ["date"])

    def test_validate_cols_invalid_id_col_name(self):
        test_df = pd.DataFrame({
            "date": [],
            "invalid_id_col_name": [],
        })
        with pytest.raises(MlflowException, match="Input data columns") as e:
            AbstractArimaModel._validate_cols(test_df, ["date", "id"])
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    def test_get_ds_weekly(self):
        expected_ds = pd.to_datetime([
            '2022-01-01 12:30:00', '2022-01-08 12:30:00',
            '2022-01-15 12:30:00', '2022-01-22 12:30:00',
            '2022-01-29 12:30:00', '2022-02-05 12:30:00',
            '2022-02-12 12:30:00', '2022-02-19 12:30:00'
        ])
        ds_indices = AbstractArimaModel._get_ds_indices(
            start_ds=pd.Timestamp("2022-01-01 12:30"),
            periods=8,
            frequency='W')
        pd.testing.assert_index_equal(expected_ds, ds_indices)

    def test_get_ds_hourly(self):
        expected_ds = pd.to_datetime([
            '2021-12-10 09:23:00', '2021-12-10 10:23:00',
            '2021-12-10 11:23:00', '2021-12-10 12:23:00',
            '2021-12-10 13:23:00', '2021-12-10 14:23:00',
            '2021-12-10 15:23:00', '2021-12-10 16:23:00',
            '2021-12-10 17:23:00', '2021-12-10 18:23:00'
        ])
        ds_indices = AbstractArimaModel._get_ds_indices(
            start_ds=pd.Timestamp("2021-12-10 09:23"),
            periods=10,
            frequency='H')
        pd.testing.assert_index_equal(expected_ds, ds_indices)


class TestLogModel(unittest.TestCase):

    def setUp(self) -> None:
        num_rows = 9
        self.X = pd.concat([
            pd.to_datetime(
                pd.Series(range(num_rows),
                          name="date").apply(lambda i: f"2020-10-{i + 1}")),
            pd.Series(range(num_rows), name="y")
        ],
                           axis=1)
        model = ARIMA(order=(2, 0, 2), suppress_warnings=True)
        model.fit(self.X.set_index("date"))
        self.pickled_model = pickle.dumps(model)

    def test_mlflow_arima_log_model(self):
        arima_model = ArimaModel(self.pickled_model,
                                 horizon=1,
                                 frequency='d',
                                 start_ds=pd.to_datetime("2020-10-01"),
                                 end_ds=pd.to_datetime("2020-10-09"),
                                 time_col="date")
        with mlflow.start_run() as run:
            mlflow_arima_log_model(arima_model)

        # Load the saved model from mlflow
        run_id = run.info.run_id
        loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

        # Make sure can make forecasts with the saved model
        loaded_model.predict(self.X.drop("y", axis=1))
        loaded_model._model_impl.python_model.predict_timeseries()

    def test_mlflow_arima_log_model_multiseries(self):
        pickled_model_dict = {
            ("1", ): self.pickled_model,
            ("2", ): self.pickled_model
        }
        start_ds_dict = {
            ("1", ): pd.Timestamp("2020-10-01"),
            ("2", ): pd.Timestamp("2020-10-01")
        }
        end_ds_dict = {
            ("1", ): pd.Timestamp("2020-10-09"),
            ("2", ): pd.Timestamp("2020-10-09")
        }
        multiseries_arima_model = MultiSeriesArimaModel(
            pickled_model_dict,
            horizon=1,
            frequency='d',
            start_ds_dict=start_ds_dict,
            end_ds_dict=end_ds_dict,
            time_col="date",
            id_cols=["id"])
        with mlflow.start_run() as run:
            mlflow_arima_log_model(multiseries_arima_model)

        # Load the saved model from mlflow
        run_id = run.info.run_id
        loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

        # Make sure can make forecasts with the saved model
        loaded_model._model_impl.python_model.predict_timeseries()
        test_df = pd.DataFrame({
            "date": [
                pd.to_datetime("2020-10-05"),
                pd.to_datetime("2020-10-05"),
                pd.to_datetime("2020-11-04"),
                pd.to_datetime("2020-11-04")
            ],
            "id": ["1", "2", "1", "2"],
        })
        loaded_model.predict(test_df)

        # Make sure can make forecasts for one-row dataframe
        loaded_model.predict(test_df[0:1])
