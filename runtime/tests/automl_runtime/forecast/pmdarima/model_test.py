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
import pytest

import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ErrorCode, INVALID_PARAMETER_VALUE
from pmdarima.arima import ARIMA

from databricks.automl_runtime.forecast.pmdarima.model import ArimaModel, MultiSeriesArimaModel, AbstractArimaModel


class TestArimaModel(unittest.TestCase):

    def setUp(self) -> None:
        num_rows = 9
        self.X = pd.concat([
            pd.to_datetime(pd.Series(range(num_rows), name="date").apply(lambda i: f"2020-10-{i + 1}")),
            pd.Series(range(num_rows), name="y")
        ], axis=1)
        model = ARIMA(order=(2, 0, 2), suppress_warnings=True)
        model.fit(self.X.set_index("date"))
        pickled_model = pickle.dumps(model)
        self.arima_model = ArimaModel(pickled_model, horizon=1, frequency='days',
                                      start_ds=pd.to_datetime("2020-10-01"), end_ds=pd.to_datetime("2020-10-09"),
                                      time_col="date")

    def test_predict_timeseries_success(self):
        forecast_pd = self.arima_model.predict_timeseries()
        expected_columns = {"yhat", "yhat_lower", "yhat_upper"}
        self.assertTrue(expected_columns.issubset(set(forecast_pd.columns)))
        self.assertEqual(10, forecast_pd.shape[0])

    def test_predict_success(self):
        test_df = pd.DataFrame({
            "date": [pd.to_datetime("2020-10-05"), pd.to_datetime("2020-11-04")]
        })
        expected_test_df = test_df.copy()
        yhat = self.arima_model.predict(None, test_df)
        self.assertEqual(10, len(yhat))
        pd.testing.assert_frame_equal(test_df, expected_test_df)  # check the input dataframe is unchanged

    def test_predict_failure_unmatched_frequency(self):
        test_df = pd.DataFrame({
            "date": [pd.to_datetime("2020-10-05"), pd.to_datetime("2020-11-04"), pd.to_datetime("2020-11-06 12:30")]
        })
        with pytest.raises(MlflowException, match="includes different frequency") as e:
            self.arima_model.predict(None, test_df)
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    def test_predict_failure_invalid_time_range(self):
        test_df = pd.DataFrame({
            "date": [pd.to_datetime("2000-10-05"), pd.to_datetime("2020-11-04")]
        })
        with pytest.raises(MlflowException, match="includes time earlier than the history data that the model was "
                                                  "trained on") as e:
            self.arima_model.predict(None, test_df)
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    def test_predict_failure_invalid_time_col_name(self):
        test_df = pd.DataFrame({
            "invalid_time_col_name": [pd.to_datetime("2020-10-05"), pd.to_datetime("2020-11-04")]
        })
        with pytest.raises(MlflowException, match="Input data columns") as e:
            self.arima_model.predict(None, test_df)
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


class TestMultiSeriesArimaModel(unittest.TestCase):

    def setUp(self) -> None:
        num_rows = 9
        self.X = pd.concat([
            pd.to_datetime(pd.Series(range(num_rows), name="date").apply(lambda i: f"2020-10-{i + 1}")),
            pd.Series(range(num_rows), name="y")
        ], axis=1)
        model = ARIMA(order=(2, 0, 2), suppress_warnings=True)
        model.fit(self.X.set_index("date"))
        pickled_model = pickle.dumps(model)
        pickled_model_dict = {"1": pickled_model, "2": pickled_model}
        start_ds_dict = {"1": pd.Timestamp("2020-10-01"), "2": pd.Timestamp("2020-10-01")}
        end_ds_dict = {"1": pd.Timestamp("2020-10-09"), "2": pd.Timestamp("2020-10-09")}
        self.arima_model = MultiSeriesArimaModel(pickled_model_dict, horizon=1, frequency='d',
                                                 start_ds_dict=start_ds_dict, end_ds_dict=end_ds_dict,
                                                 time_col="date", id_cols=["id"])

    def test_predict_timeseries_success(self):
        forecast_pd = self.arima_model.predict_timeseries()
        expected_columns = {"yhat", "yhat_lower", "yhat_upper"}
        self.assertTrue(expected_columns.issubset(set(forecast_pd.columns)))
        self.assertEqual(20, forecast_pd.shape[0])

    def test_predict_success(self):
        test_df = pd.DataFrame({
            "date": [pd.to_datetime("2020-10-05"), pd.to_datetime("2020-10-05"),
                     pd.to_datetime("2020-11-04"), pd.to_datetime("2020-11-04")],
            "id": ["1", "2", "1", "2"],
        })
        expected_test_df = test_df.copy()
        yhat = self.arima_model.predict(None, test_df)
        self.assertEqual(10, len(yhat))
        pd.testing.assert_frame_equal(test_df, expected_test_df)  # check the input dataframe is unchanged

    def test_predict_fail_unseen_id(self):
        test_df = pd.DataFrame({
            "date": [pd.to_datetime("2020-10-05"), pd.to_datetime("2020-10-05"),
                     pd.to_datetime("2020-11-04"), pd.to_datetime("2020-11-04")],
            "id": ["1", "2", "1", "3"],
        })
        with pytest.raises(MlflowException, match="includes unseen values in id columns") as e:
            self.arima_model.predict(None, test_df)
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    def test_predict_failure_unmatched_frequency(self):
        test_df = pd.DataFrame({
            "date": [pd.to_datetime("2020-10-05"), pd.to_datetime("2020-10-05 12:30"),
                     pd.to_datetime("2020-11-04"), pd.to_datetime("2020-11-04")],
            "id": ["1", "2", "1", "2"],
        })
        with pytest.raises(MlflowException, match="includes different frequency") as e:
            self.arima_model.predict(None, test_df)
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    def test_predict_failure_invalid_time_range(self):
        test_df = pd.DataFrame({
            "date": [pd.to_datetime("2020-10-05"), pd.to_datetime("2000-10-05"),
                     pd.to_datetime("2020-11-04"), pd.to_datetime("2020-11-04")],
            "id": ["1", "2", "1", "2"],
        })
        with pytest.raises(MlflowException, match="includes time earlier than the history data that the model was "
                                                  "trained on") as e:
            self.arima_model.predict(None, test_df)
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    def test_predict_failure_invalid_time_col_name(self):
        test_df = pd.DataFrame({
            "time": [pd.to_datetime("2020-10-05"), pd.to_datetime("2000-10-05"),
                     pd.to_datetime("2020-11-04"), pd.to_datetime("2020-11-04")],
            "invalid_id_col_name": ["1", "2", "1", "2"],
        })
        with pytest.raises(MlflowException, match="Input data columns") as e:
            self.arima_model.predict(None, test_df)
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


class TestAbstractArimaModel(unittest.TestCase):

    def test_validate_cols_success(self):
        test_df = pd.DataFrame({"date": []})
        AbstractArimaModel._validate_cols(test_df, ["date"])

    def test_validate_cols_invalid_id_col_name(self):
        test_df = pd.DataFrame({"date": [], "invalid_id_col_name": [], })
        with pytest.raises(MlflowException, match="Input data columns") as e:
            AbstractArimaModel._validate_cols(test_df, ["date", "id"])
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
