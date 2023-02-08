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
from abc import abstractmethod
from typing import List, Dict, Tuple, Optional, Union

import pandas as pd
import mlflow
import pmdarima
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.environment import _mlflow_conda_env

from databricks.automl_runtime.forecast import OFFSET_ALIAS_MAP, DATE_OFFSET_KEYWORD_MAP
from databricks.automl_runtime.forecast.model import ForecastModel, mlflow_forecast_log_model
from databricks.automl_runtime.forecast.utils import calculate_period_differences, is_frequency_consistency, \
    make_future_dataframe, make_single_future_dataframe


ARIMA_CONDA_ENV = _mlflow_conda_env(
    additional_pip_deps=[
        f"pmdarima=={pmdarima.__version__}",
        f"pickle=={pickle.format_version}",
        f"pandas=={pd.__version__}",
    ]
)


class AbstractArimaModel(ForecastModel):
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

    @property
    def model_env(self):
        return ARIMA_CONDA_ENV

    @staticmethod
    def _get_ds_indices(start_ds: pd.Timestamp, periods: int, frequency: str) -> pd.DatetimeIndex:
        """
        Create a DatetimeIndex with specified starting time and frequency, whose length is the given periods.
        :param start_ds: the pd.Timestamp as the start of the DatetimeIndex.
        :param periods: the length of the DatetimeIndex.
        :param frequency: the frequency of the DatetimeIndex.
        :return: a DatetimeIndex.
        """
        ds_indices = pd.date_range(
            start=start_ds,
            periods=periods,
            freq=pd.DateOffset(**DATE_OFFSET_KEYWORD_MAP[frequency])
        )
        modified_start_ds = ds_indices.min()
        if start_ds != modified_start_ds:
            offset = modified_start_ds - start_ds
            ds_indices = ds_indices - offset
        return ds_indices


class ArimaModel(AbstractArimaModel):
    """
    ARIMA mlflow model wrapper for univariate forecasting.
    """

    def __init__(self, pickled_model: bytes, horizon: int, frequency: str,
                 start_ds: pd.Timestamp, end_ds: pd.Timestamp,
                 time_col: str, exogenous_cols: Optional[List[str]] = None) -> None:
        """
        Initialize the mlflow Python model wrapper for ARIMA.
        :param pickled_model: the pickled ARIMA model as a bytes object.
        :param horizon: int number of periods to forecast forward.
        :param frequency: the frequency of the time series
        :param start_ds: the start time of training data
        :param end_ds: the end time of training data
        :param time_col: the column name of the time column
        :param exogenous_cols: Optional list of column names of exogenous variables. If provided, these columns are
        used as additional features in arima model.
        """
        super().__init__()
        self._pickled_model = pickled_model
        self._horizon = horizon
        self._frequency = OFFSET_ALIAS_MAP[frequency]
        self._start_ds = pd.to_datetime(start_ds)
        self._end_ds = pd.to_datetime(end_ds)
        self._time_col = time_col
        self._exogenous_cols = exogenous_cols

    def model(self) -> pmdarima.arima.ARIMA:
        """
        Deserialize the ARIMA model by pickle.
        :return: ARIMA model
        """
        return pickle.loads(self._pickled_model)

    def predict_timeseries(
        self,
        horizon: int = None,
        include_history: bool = True,
        df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Predict target column for given horizon_timedelta and history data.
        :param horizon: int number of periods to forecast forward.
        :param include_history: Boolean to include the historical dates in the data
            frame for predictions.
        :param df: A pd.Dataframe containing regressors (exogenous variables), if they were used to train the model.
        :return: A pd.DataFrame with the forecasts and confidence intervals for given horizon_timedelta and history data.
        """
        horizon = horizon or self._horizon
        X = None
        if self._exogenous_cols and df is not None:
            time_col = self._time_col if self._time_col in df.columns else "ds"
            X = (df[df[time_col] > self._end_ds].set_index(time_col))[self._exogenous_cols]
        future_pd = self._forecast(horizon, X)
        if include_history:
            in_sample_pd = self._predict_in_sample(start_ds=self._start_ds, end_ds=self._end_ds, X=X)
            return pd.concat([in_sample_pd, future_pd]).reset_index(drop=True)
        else:
            return future_pd

    def make_future_dataframe(self, horizon: int = None, include_history: bool = True) -> pd.DataFrame:
        """
        Generate future dataframe by calling the API from prophet
        :param horizon: Int number of periods to forecast forward.
        :param include_history: Boolean to include the historical dates in the data
            frame for predictions.
        :return: pd.Dataframe that extends forward from the end of self.history for the
        requested number of periods.
        """
        return make_single_future_dataframe(
            start_time=self._start_ds,
            end_time=self._end_ds,
            horizon=horizon or self._horizon,
            frequency=self._frequency,
            include_history=include_history
        )

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
        df["ds"] = pd.to_datetime(df["ds"], infer_datetime_format=True)
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
        # Check if the time has correct frequency
        consistency = df["ds"].apply(lambda x: 
            is_frequency_consistency(self._start_ds, x, self._frequency)
        ).all()
        if not consistency:
            raise MlflowException(
                message=(
                    f"Input time column '{self._time_col}' includes different frequency."
                ),
                error_code=INVALID_PARAMETER_VALUE,
            )
        preds_pds = []
        # Out-of-sample prediction if needed
        horizon = calculate_period_differences(self._end_ds, max(df["ds"]), self._frequency)
        if horizon > 0:
            X_future = df[df["ds"] > self._end_ds].set_index("ds")
            future_pd = self._forecast(
                horizon,
                X=X_future[self._exogenous_cols] if self._exogenous_cols else None)
            preds_pds.append(future_pd)
        # In-sample prediction if needed
        if pred_start_ds <= self._end_ds:
            X_in_sample = df[df["ds"] <= self._end_ds].set_index("ds")
            in_sample_pd = self._predict_in_sample(
                start_ds=pred_start_ds,
                end_ds=self._end_ds,
                X=X_in_sample[self._exogenous_cols] if self._exogenous_cols else None)
            preds_pds.append(in_sample_pd)
        # Map predictions back to given timestamps
        preds_pd = pd.concat(preds_pds).set_index("ds")
        df = df.set_index("ds").join(preds_pd, how="left").reset_index()
        return df

    def _predict_in_sample(
        self,
        start_ds: pd.Timestamp = None,
        end_ds: pd.Timestamp = None,
        X: pd.DataFrame = None) -> pd.DataFrame:
        if start_ds and end_ds:
            start_idx = calculate_period_differences(self._start_ds, start_ds, self._frequency)
            end_idx = calculate_period_differences(self._start_ds, end_ds, self._frequency)
        else:
            start_ds = self._start_ds
            end_ds = self._end_ds
            start_idx, end_idx = None, None
        d = self.model().order[1]
        start_idx = max(start_idx, d)
        preds_in_sample, conf_in_sample = self.model().predict_in_sample(
            X=X,
            start=start_idx,
            end=end_idx,
            return_conf_int=True)
        periods = calculate_period_differences(self._start_ds, end_ds, self._frequency) + 1
        ds_indices = self._get_ds_indices(start_ds=self._start_ds, periods=periods, frequency=self._frequency)[start_idx:]
        in_sample_pd = pd.DataFrame({'ds': ds_indices, 'yhat': preds_in_sample})
        in_sample_pd[["yhat_lower", "yhat_upper"]] = conf_in_sample
        return in_sample_pd

    def _forecast(
        self,
        horizon: int = None,
        X: pd.DataFrame = None) -> pd.DataFrame:
        horizon = horizon or self._horizon
        preds, conf = self.model().predict(
            horizon,
            X=X,
            return_conf_int=True)
        ds_indices = self._get_ds_indices(start_ds=self._end_ds, periods=horizon + 1, frequency=self._frequency)[1:]
        preds_pd = pd.DataFrame({'ds': ds_indices, 'yhat': preds})
        preds_pd[["yhat_lower", "yhat_upper"]] = conf
        return preds_pd


class MultiSeriesArimaModel(AbstractArimaModel):
    """
    ARIMA mlflow model wrapper for multivariate forecasting.
    """

    def __init__(self, pickled_model_dict: Dict[Tuple, bytes], horizon: int, frequency: str,
                 start_ds_dict: Dict[Tuple, pd.Timestamp], end_ds_dict: Dict[Tuple, pd.Timestamp],
                 time_col: str, id_cols: List[str], exogenous_cols: Optional[List[str]] = None) -> None:
        """
        Initialize the mlflow Python model wrapper for multiseries ARIMA.
        :param pickled_model_dict: the dictionary of binarized ARIMA models for different time series.
        :param horizon: int number of periods to forecast forward.
        :param frequency: the frequency of the time series
        :param start_ds_dict: the dictionary of the starting time of each time series in training data.
        :param end_ds_dict: the dictionary of the end time of each time series in training data.
        :param time_col: the column name of the time column
        :param id_cols: the column names of the identity columns for multi-series time series
        :param exogenous_cols: Optional list of column names of exogenous variables. If provided, these columns are
        used as additional features in arima model.
        """
        super().__init__()
        self._pickled_models = pickled_model_dict
        self._horizon = horizon
        self._frequency = frequency
        self._starts = start_ds_dict
        self._ends = end_ds_dict
        self._time_col = time_col
        self._id_cols = id_cols
        self._exogenous_cols = exogenous_cols

    def model(self, id_: Tuple) -> pmdarima.arima.ARIMA:
        """
        Deserialize the ARIMA model for specified time series by pickle.
        :param: id for specified time series.
        :return: ARIMA model
        """
        return pickle.loads(self._pickled_models[id_])

    def make_future_dataframe(
            self,
            horizon: Optional[int] = None,
            include_history: bool = True,
            groups: List[Tuple] = None,
    ) -> pd.DataFrame:
        """
        Generate dataframe with future timestamps for all valid identities
        :param horizon: Int number of periods in the future
        :param frequency: frequency of the history time series. It should be valid frequency
            for pd.date_range, such as "D" or "M"
        :param groups: the collection of group(s) to generate forecast predictions.
            The group definiteions must be the key value
        :return: pd.DataFrame that extends forward from history_last_date
        """
        if horizon is None:
            horizon = self._horizon
        if groups is not None:
            model_keys = set(self._pickled_models.keys())
            if not set(groups).issubset(model_keys):
                raise ValueError(f"Invalid groups: {set(groups) - model_keys}.")
        else:
            groups = list(self._pickled_models.keys())

        future_df = make_future_dataframe(
            start_time=self._starts,
            end_time=self._ends,
            horizon=horizon,
            frequency=self._frequency,
            include_history=include_history,
            groups=groups,
            identity_column_names=self._id_cols
        )
        return future_df

    def predict_timeseries(
        self,
        horizon: int = None,
        include_history: bool = True,
        df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Predict target column for given horizon_timedelta and history data.
        :param horizon: Int number of periods to forecast forward.
        :param include_history: Boolean to include the historical dates in the data
            frame for predictions.
        :param df: A pd.Dataframe containing regressors (exogenous variables), if they were used to train the model.
        :return: A pd.DataFrame with the forecast components.
        """
        horizon = horizon or self._horizon
        ids = self._pickled_models.keys()
        preds_dfs = list(map(lambda id_: self._predict_timeseries_single_id(id_, horizon, include_history, df), ids))
        return pd.concat(preds_dfs).reset_index(drop=True)

    def _predict_timeseries_single_id(
        self,
        id_: Tuple,
        horizon: int,
        include_history: bool = True,
        df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        arima_model_single_id = ArimaModel(self._pickled_models[id_], self._horizon, self._frequency,
                                           self._starts[id_], self._ends[id_], self._time_col, self._exogenous_cols)
        preds_df = arima_model_single_id.predict_timeseries(horizon, include_history, df)
        for id, col_name in zip(id_, self._id_cols):
            preds_df[col_name] = id
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
        df["ts_id"] = df[self._id_cols].apply(tuple, axis=1)
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
        preds_df = df.groupby(self._id_cols).apply(self._predict_single_id).reset_index(drop=True)
        df = df.merge(preds_df, how="left", on=[self._time_col] + self._id_cols)  # merge predictions to original order
        return df["yhat"]

    def _predict_single_id(self, df: pd.DataFrame) -> pd.DataFrame:
        id_ = df["ts_id"].to_list()[0]
        arima_model_single_id = ArimaModel(self._pickled_models[id_],
                                           self._horizon,
                                           self._frequency,
                                           self._starts[id_],
                                           self._ends[id_],
                                           self._time_col,
                                           self._exogenous_cols)
        df["yhat"] = arima_model_single_id.predict(context=None, model_input=df).to_list()
        return df


def mlflow_arima_log_model(arima_model: Union[ArimaModel, MultiSeriesArimaModel],
                           sample_input: pd.DataFrame = None) -> None:
    """
    Log the model to mlflow.
    :param arima_model: ARIMA model wrapper
    :param sample_input: sample input Dataframes for model inference
    """
    mlflow_forecast_log_model(arima_model, sample_input)
