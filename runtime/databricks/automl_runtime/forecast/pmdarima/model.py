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
import pickle
from abc import ABC, abstractmethod
from typing import List, Dict

import pandas as pd
import mlflow
import pmdarima
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

from databricks.automl_runtime.forecast import OFFSET_ALIAS_MAP


class AbstractArimaModel(ABC, mlflow.pyfunc.PythonModel):
    @abstractmethod
    def __init__(self):
        super().__init__()

    def load_context(self, context: mlflow.pyfunc.model.PythonModelContext) -> None:
        """
        Loads artifacts from the specified PythonModelContext.

        Loads artifacts from the specified PythonModelContext that can be used by
        PythonModel.predict when evaluating inputs. When loading an MLflow model with
        load_pyfunc, this method is called as soon as the PythonModel is constructed.
        :param context: A PythonModelContext instance containing artifacts that the model
                        can use to perform inference.
        """
        from pmdarima.arima import ARIMA  # noqa: F401

    @staticmethod
    def _validate_cols(df: pd.DataFrame, required_cols: List[str]):
        df_cols = set(df.columns)
        required_cols_set = set(required_cols)
        if not required_cols_set.issubset(df_cols):
            raise MlflowException(
                message=(
                    f"Input data columns '{list(df_cols)}' do not contain the required columns '{required_cols}'"
                ),
                error_code=INVALID_PARAMETER_VALUE,
            )


class ArimaModel(AbstractArimaModel):
    """
    ARIMA mlflow model wrapper for univariate forecasting.
    """

    def __init__(self, pickled_model: bytes, horizon: int, frequency: str,
                 start_ds: pd.Timestamp, end_ds: pd.Timestamp, time_col: str) -> None:
        """
        Initialize the mlflow Python model wrapper for ARIMA.
        :param pickled_model: the pickled ARIMA model as a bytes object.
        :param horizon: int number of periods to forecast forward.
        :param frequency: the frequency of the time series
        :param start_ds: the start time of training data
        :param end_ds: the end time of training data
        :param time_col: the column name of the time column
        """
        super().__init__()
        self._pickled_model = pickled_model
        self._horizon = horizon
        self._frequency = OFFSET_ALIAS_MAP[frequency]
        self._start_ds = start_ds
        self._end_ds = end_ds
        self._time_col = time_col

    def model(self) -> pmdarima.arima.ARIMA:
        """
        Deserialize the ARIMA model by pickle.
        :return: ARIMA model
        """
        return pickle.loads(self._pickled_model)

    def predict_timeseries(self, horizon: int = None) -> pd.DataFrame:
        """
        Predict target column for given horizon and history data.
        :param horizon: int number of periods to forecast forward.
        :return: A pd.DataFrame with the forecasts and confidence intervals for given horizon and history data.
        """
        horizon = horizon or self._horizon
        future_pd = self._forecast(horizon)
        in_sample_pd = self._predict_in_sample()
        return pd.concat([in_sample_pd, future_pd]).reset_index(drop=True)

    def predict(self, context: mlflow.pyfunc.model.PythonModelContext, model_input: pd.DataFrame) -> pd.Series:
        """
        Predict API from mlflow.pyfunc.PythonModel.

        Returns the prediction values for given timestamps in the input dataframe. If an input timestamp
        to predict does not match the original frequency that the model trained on, an exception will be thrown.
        :param context: A PythonModelContext instance containing artifacts that the model
                        can use to perform inference.
        :param model_input: The input dataframe of the model. Should have the same time column name
                            as the training data of the ARIMA model.
        :return: A pd.Series with the prediction values.
        """
        self._validate_cols(model_input, [self._time_col])
        result_df = self._predict_impl(model_input)
        return result_df["yhat"]

    def _predict_impl(self, input_df: pd.DataFrame) -> pd.DataFrame:
        df = input_df.rename(columns={self._time_col: "ds"})
        # Check if the time has correct frequency
        diff = (df["ds"] - self._start_ds) / pd.Timedelta(1, unit=self._frequency)
        if not diff.apply(float.is_integer).all():
            raise MlflowException(
                message=(
                    f"Input time column '{self._time_col}' includes different frequency."
                ),
                error_code=INVALID_PARAMETER_VALUE,
            )
        # Validate the time range
        pred_start_ds = min(df["ds"])
        if pred_start_ds < self._start_ds:
            raise MlflowException(
                message=(
                    f"Input time column '{self._time_col}' includes time earlier than "
                    "the history data that the model was trained on."
                ),
                error_code=INVALID_PARAMETER_VALUE,
            )
        preds_pds = []
        # Out-of-sample prediction if needed
        horizon = int((max(df["ds"]) - self._end_ds) / pd.Timedelta(1, unit=self._frequency))
        if horizon > 0:
            future_pd = self._forecast(horizon)
            preds_pds.append(future_pd)
        # In-sample prediction if needed
        if pred_start_ds <= self._end_ds:
            in_sample_pd = self._predict_in_sample(start_ds=pred_start_ds, end_ds=self._end_ds)
            preds_pds.append(in_sample_pd)
        # Map predictions back to given timestamps
        preds_pd = pd.concat(preds_pds).set_index("ds")
        df = df.set_index("ds").join(preds_pd, how="left").reset_index()
        return df

    def _predict_in_sample(self, start_ds: pd.Timestamp = None, end_ds: pd.Timestamp = None) -> pd.DataFrame:
        if start_ds and end_ds:
            start_idx = int((start_ds - self._start_ds) / pd.Timedelta(1, unit=self._frequency))
            end_idx = int((end_ds - self._start_ds) / pd.Timedelta(1, unit=self._frequency))
        else:
            start_ds = self._start_ds
            end_ds = self._end_ds
            start_idx, end_idx = None, None
        preds_in_sample, conf_in_sample = self.model().predict_in_sample(
            start=start_idx, end=end_idx, return_conf_int=True)
        dates_in_sample = pd.date_range(start=start_ds, end=end_ds, freq=self._frequency)
        in_sample_pd = pd.DataFrame({'ds': dates_in_sample, 'yhat': preds_in_sample})
        in_sample_pd[["yhat_lower", "yhat_upper"]] = conf_in_sample
        return in_sample_pd

    def _forecast(self, horizon: int = None) -> pd.DataFrame:
        horizon = horizon or self._horizon
        preds, conf = self.model().predict(horizon, return_conf_int=True)
        dates = pd.date_range(start=self._end_ds, periods=horizon + 1, freq=self._frequency)[1:]
        preds_pd = pd.DataFrame({'ds': dates, 'yhat': preds})
        preds_pd[["yhat_lower", "yhat_upper"]] = conf
        return preds_pd


class MultiSeriesArimaModel(AbstractArimaModel):
    """
    ARIMA mlflow model wrapper for multivariate forecasting.
    """

    def __init__(self, pickled_model_dict: Dict[str, bytes], horizon: int, frequency: str,
                 start_ds_dict: Dict[str, pd.Timestamp], end_ds_dict: Dict[str, pd.Timestamp],
                 time_col: str, id_cols: List[str]) -> None:
        """
        Initialize the mlflow Python model wrapper for multiseries ARIMA.
        :param pickled_model_dict: the dictionary of binarized ARIMA models for different time series.
        :param horizon: int number of periods to forecast forward.
        :param frequency: the frequency of the time series
        :param start_ds_dict: the dictionary of the starting time of each time series in training data.
        :param end_ds_dict: the dictionary of the starting time of each time series  in training data.
        :param time_col: the column name of the time column
        :param id_cols: the column names of the identity columns for multi-series time series
        """
        super().__init__()
        self._pickled_models = pickled_model_dict
        self._horizon = horizon
        self._frequency = frequency
        self._starts = start_ds_dict
        self._ends = end_ds_dict
        self._time_col = time_col
        self._id_cols = id_cols

    def model(self, id_: str) -> pmdarima.arima.ARIMA:
        """
        Deserialize the ARIMA model for specified time series by pickle.
        :param: id for specified time series.
        :return: ARIMA model
        """
        return pickle.loads(self._pickled_models[id_])

    def predict_timeseries(self, horizon: int = None) -> pd.DataFrame:
        """
        Predict target column for given horizon and history data.
        :param horizon: Int number of periods to forecast forward.
        :return: A pd.DataFrame with the forecast components.
        """
        horizon = horizon or self._horizon
        ids = self._pickled_models.keys()
        preds_dfs = list(map(lambda id_: self._predict_timeseries_single_id(id_, horizon), ids))
        return pd.concat(preds_dfs).reset_index(drop=True)

    def _predict_timeseries_single_id(self, id_: str, horizon: int) -> pd.DataFrame:
        arima_model_single_id = ArimaModel(self._pickled_models[id_], self._horizon, self._frequency,
                                           self._starts[id_], self._ends[id_], self._time_col)
        preds_df = arima_model_single_id.predict_timeseries(horizon)
        preds_df["ts_id"] = id_
        return preds_df

    def predict(self, context: mlflow.pyfunc.model.PythonModelContext, model_input: pd.DataFrame) -> pd.Series:
        """
        Predict API from mlflow.pyfunc.PythonModel.

        Returns the prediction values for given timestamps in the input dataframe. If an input timestamp
        to predict does not match the original frequency that the model trained on, an exception will be thrown.
        :param context: A PythonModelContext instance containing artifacts that the model
                        can use to perform inference.
        :param model_input: input dataframe of the model. Should have the same time column
                            and identity columns names as the training data of the ARIMA model.
        :return: A pd.Series with the prediction values.
        """
        self._validate_cols(model_input, self._id_cols + [self._time_col])
        df = model_input.copy()
        df["ts_id"] = df[self._id_cols].apply(lambda r: "-".join(r.values.astype(str)), axis=1)
        known_ids = set(self._pickled_models.keys())
        ids = set(df["ts_id"].unique())
        if not ids.issubset(known_ids):
            raise MlflowException(
                message=(
                    f"Input data includes unseen values in id columns '{self._id_cols}'."
                    f"Expected combined ids: {known_ids}\n"
                    f"Got ids: {ids}\n"
                ),
                error_code=INVALID_PARAMETER_VALUE,
            )
        preds_df = df.groupby("ts_id").apply(self._predict_single_id).reset_index(drop=True)
        df = df.merge(preds_df, how="left", on=[self._time_col, "ts_id"])  # merge predictions to original order
        return df["yhat"]

    def _predict_single_id(self, df: pd.DataFrame) -> pd.DataFrame:
        id_ = df["ts_id"].to_list()[0]
        arima_model_single_id = ArimaModel(self._pickled_models[id_], self._horizon, self._frequency,
                                           self._starts[id_], self._ends[id_], self._time_col)
        df["yhat"] = arima_model_single_id.predict(None, df).to_list()
        return df
