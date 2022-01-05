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
import pytest

import pandas as pd
import mlflow
import numpy as np
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ErrorCode, INVALID_PARAMETER_VALUE
from pmdarima.arima import auto_arima, StepwiseContext

from databricks.automl_runtime.forecast.pmdarima.model import ArimaModel, MultiSeriesArimaModel


class TestArimaModel(unittest.TestCase):

    def setUp(self) -> None:
        num_rows = 9
        self.X = pd.concat([
            pd.to_datetime(pd.Series(range(num_rows), name="date").apply(lambda i: f"2020-10-{i + 1}")),
            pd.Series(range(num_rows), name="y")
        ], axis=1)
        with StepwiseContext(max_steps=1):
            model = auto_arima(y=self.X.set_index("date"), m=1)
        pickled_model = pickle.dumps(model)
        self.arima_model = ArimaModel(pickled_model, horizon=1, frequency='d',
                                      start_ds=pd.to_datetime("2020-10-01"), end_ds=pd.to_datetime("2020-10-09"),
                                      time_col="date")

    def test_predict_timeseries_success(self):
        forecast_pd = self.arima_model.predict_timeseries()
        expected_yhat = np.array([4.002726, 0.068215, 1.572192, 2.354237, 3.519087,
                                  4.385046, 5.478589, 6.378852, 7.441261, 8.360274])
        np.testing.assert_array_almost_equal(np.array(forecast_pd["yhat"]), expected_yhat)

    def test_predict_success(self):
        test_df = pd.DataFrame({
            "date": [pd.to_datetime("2020-10-05"), pd.to_datetime("2020-11-04")]
        })
        expected_test_df = test_df.copy()
        expected_yhat = np.array([3.519087, 5.833953])
        yhat = self.arima_model.predict(None, test_df)
        np.testing.assert_array_almost_equal(np.array(yhat), expected_yhat)
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
            "time": [pd.to_datetime("2020-10-05"), pd.to_datetime("2020-11-04")]
        })
        with pytest.raises(MlflowException, match="Input data columns") as e:
            ArimaModel._validate_cols(test_df, ["invalid_time_col"])
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    def test_validate_cols_invalid_time_col_name(self):
        test_df = pd.DataFrame({"date": []})
        with pytest.raises(MlflowException, match="Input data columns") as e:
            ArimaModel._validate_cols(test_df, ["invalid_time_col"])
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


class TestMultiSeriesArimaModel(unittest.TestCase):

    def setUp(self) -> None:
        num_rows = 9
        self.X = pd.concat([
            pd.to_datetime(pd.Series(range(num_rows), name="date").apply(lambda i: f"2020-10-{i + 1}")),
            pd.Series(range(num_rows), name="y")
        ], axis=1)
        with StepwiseContext(max_steps=1):
            model = auto_arima(y=self.X.set_index("date"), m=1)
        pickled_model = pickle.dumps(model)
        pickled_model_dict = {"1": pickled_model, "2": pickled_model}
        start_ds_dict = {"1": pd.Timestamp("2020-10-01"), "2": pd.Timestamp("2020-10-01")}
        end_ds_dict = {"1": pd.Timestamp("2020-10-09"), "2": pd.Timestamp("2020-10-09")}
        self.arima_model = MultiSeriesArimaModel(pickled_model_dict, horizon=1, frequency='d',
                                                 start_ds_dict=start_ds_dict, end_ds_dict=end_ds_dict,
                                                 time_col="date", id_cols=["id"])

    def test_predict_timeseries_success(self):
        forecast_pd = self.arima_model.predict_timeseries()
        expected_yhat_one_series = np.array([4.002726, 0.068215, 1.572192, 2.354237, 3.519087,
                                             4.385046, 5.478589, 6.378852, 7.441261, 8.360274])
        expected_yhat = np.append(expected_yhat_one_series, expected_yhat_one_series)
        np.testing.assert_array_almost_equal(np.array(forecast_pd["yhat"]), expected_yhat)

    def test_predict_success(self):
        test_df = pd.DataFrame({
            "date": [pd.to_datetime("2020-10-05"), pd.to_datetime("2020-10-05"),
                     pd.to_datetime("2020-11-04"), pd.to_datetime("2020-11-04")],
            "id": ["1", "2", "1", "2"],
        })
        expected_test_df = test_df.copy()
        expected_yhat = np.array([3.519087, 3.519087, 5.833953, 5.833953])
        yhat = self.arima_model.predict(None, test_df)
        np.testing.assert_array_almost_equal(np.array(yhat), expected_yhat)
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
            "id": ["1", "2", "1", "2"],
        })
        with pytest.raises(MlflowException, match="Input data columns") as e:
            ArimaModel._validate_cols(test_df, ["invalid_time_col"])
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    def test_validate_cols_invalid_id_col_name(self):
        test_df = pd.DataFrame({"date": [], "id": [], })
        with pytest.raises(MlflowException, match="Input data columns") as e:
            ArimaModel._validate_cols(test_df, ["invalid_id_col"])
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
