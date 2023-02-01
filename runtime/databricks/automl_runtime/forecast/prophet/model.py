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
from typing import Dict, List, Optional, Tuple, Union

import cloudpickle
import mlflow
import pandas as pd
import prophet

from mlflow.models.signature import ModelSignature
from mlflow.utils.environment import _mlflow_conda_env

from databricks.automl_runtime.forecast import OFFSET_ALIAS_MAP, DATE_OFFSET_KEYWORD_MAP
from databricks.automl_runtime.forecast.model import ForecastModel, mlflow_forecast_log_model
from databricks.automl_runtime import version
from databricks.automl_runtime.forecast.utils import is_quaterly_alias, make_future_dataframe


PROPHET_CONDA_ENV = _mlflow_conda_env(
    additional_pip_deps=[
        f"prophet=={prophet.__version__}",
        f"cloudpickle=={cloudpickle.__version__}",
        f"databricks-automl-runtime=={version.__version__}",
    ]
)


class ProphetModel(ForecastModel):
    """
    Prophet mlflow model wrapper for univariate forecasting.
    """

    def __init__(self, model_json: Union[Dict[Tuple, str], str], horizon: int, frequency: str,
                 time_col: str) -> None:
        """
        Initialize the mlflow Python model wrapper for mlflow
        :param model_json: json string of the Prophet model or
        the dictionary of json strings of Prophet model for multi-series forecasting
        :param horizon: Int number of periods to forecast forward.
        :param frequency: the frequency of the time series
        :param time_col: the column name of the time column
        """
        self._model_json = model_json
        self._horizon = horizon
        self._frequency = frequency
        self._time_col = time_col
        self._is_quaterly = is_quaterly_alias(frequency)
        super().__init__()

    def load_context(self, context: mlflow.pyfunc.model.PythonModelContext) -> None:
        """
        Loads artifacts from the specified :class:`~PythonModelContext` that can be used
        :param context: A :class:`~PythonModelContext` instance containing artifacts that the model
                        can use to perform inference.
        """
        from prophet import Prophet  # noqa: F401
        return

    @property
    def model_env(self):
        return PROPHET_CONDA_ENV

    def model(self) -> prophet.forecaster.Prophet:
        """
        Deserialize a Prophet model from json string
        :return: Prophet model
        """
        from prophet.serialize import model_from_json
        return model_from_json(self._model_json)

    def make_future_dataframe(self, horizon: int = None, include_history: bool = True) -> pd.DataFrame:
        """
        Generate future dataframe by calling the API from prophet
        :param horizon: Int number of periods to forecast forward.
        :param include_history: Boolean to include the historical dates in the data
            frame for predictions.
        :return: pd.Dataframe that extends forward from the end of self.history for the
        requested number of periods.
        """
        offset_kwarg = DATE_OFFSET_KEYWORD_MAP[OFFSET_ALIAS_MAP[self._frequency]]
        return self.model().make_future_dataframe(periods=horizon or self._horizon,
                                                  freq=pd.DateOffset(**offset_kwarg),
                                                  include_history=include_history)

    def _predict_impl(self, horizon: int = None, include_history: bool = True) -> pd.DataFrame:
        """
        Predict using the API from prophet model.
        :param horizon: Int number of periods to forecast forward.
        :param include_history: Boolean to include the historical dates in the data
            frame for predictions.
        :return: A pd.DataFrame with the forecast components.
        """
        future_pd = self.make_future_dataframe(horizon=horizon or self._horizon, include_history=include_history)
        return self.model().predict(future_pd)

    def predict_timeseries(self, horizon: int = None, include_history: bool = True) -> pd.DataFrame:
        """
        Predict using the prophet model.
        :param horizon: Int number of periods to forecast forward.
        :param include_history: Boolean to include the historical dates in the data
            frame for predictions.
        :return: A pd.DataFrame with the forecast components.
        """
        return self._predict_impl(horizon, include_history)

    def predict(self, context: mlflow.pyfunc.model.PythonModelContext, model_input: pd.DataFrame) -> pd.Series:
        """
        Predict API from mlflow.pyfunc.PythonModel
        :param context: A :class:`~PythonModelContext` instance containing artifacts that the model
                        can use to perform inference.
        :param model_input: Input dataframe
        :return: A pd.DataFrame with the forecast components.
        """
        self._validate_cols(model_input, [self._time_col])
        test_df = pd.DataFrame({"ds": model_input[self._time_col]})
        predict_df = self.model().predict(test_df)
        return predict_df["yhat"]

    def infer_signature(self, sample_input: pd.DataFrame = None) -> ModelSignature:
        if sample_input is None:
            sample_input = self.make_future_dataframe(horizon=1)
            sample_input.rename(columns={"ds": self._time_col}, inplace=True)
        return super().infer_signature(sample_input)


class MultiSeriesProphetModel(ProphetModel):
    """
    Prophet mlflow model wrapper for multi-series forecasting.
    """

    def __init__(self, model_json: Dict[Tuple, str], timeseries_starts: Dict[Tuple, pd.Timestamp],
                 timeseries_end: str, horizon: int, frequency: str, time_col: str, id_cols: List[str],
                 ) -> None:
        """
        Initialize the mlflow Python model wrapper for mlflow
        :param model_json: the dictionary of json strings of Prophet model for multi-series forecasting
        :param timeseries_starts: the dictionary of pd.Timestamp as the starting time of each time series
        :param timeseries_end: the end time of the time series
        :param horizon: int number of periods to forecast forward
        :param frequency: the frequency of the time series
        :param time_col: the column name of the time column
        :param id_cols: the column names of the identity columns for multi-series time series
        """
        super().__init__(model_json, horizon, frequency, time_col)
        self._frequency = frequency
        self._timeseries_end = timeseries_end
        self._timeseries_starts = timeseries_starts
        self._id_cols = id_cols

    def model(self, id: Tuple) -> Optional[prophet.forecaster.Prophet]:
        """
        Deserialize one Prophet model from json string based on the id
        :param id: identity for the Prophet model
        :return: Prophet model
        """
        from prophet.serialize import model_from_json
        if id in self._model_json:
            return model_from_json(self._model_json[id])
        return None

    def make_future_dataframe(
            self,
            horizon: Optional[int] = None,
            include_history: bool = True,
            groups: List[Tuple] = None,
    ) -> pd.DataFrame:
        """
        Generate dataframe with future timestamps for all valid identities
        :param horizon: Int number of periods in the future
        :param include_history: Boolean to include the historical dates in the data
            frame for predictions.
        :param groups: the collection of group(s) to generate forecast predictions.
        :return: pd.DataFrame that extends forward
        """
        horizon=horizon or self._horizon
        if groups is not None:
            model_keys = set(self._model_json.keys())
            if not set(groups).issubset(model_keys):
                raise ValueError(f"Invalid groups: {set(groups) - model_keys}.")
        else:
            groups = list(self._model_json.keys())

        end_time = pd.Timestamp(self._timeseries_end)
        future_df = make_future_dataframe(
            start_time=self._timeseries_starts,
            end_time=end_time,
            horizon=horizon,
            frequency=self._frequency,
            include_history=include_history,
            groups=groups,
            identity_column_names=self._id_cols
        )
        return future_df

    def _predict_impl(self, df: pd.DataFrame, horizon: int = None, include_history: bool = True) -> pd.DataFrame:
        """
        Predict using the API from prophet model.
        :param df: input dataframe
        :param horizon: Int number of periods to forecast forward.
        :param include_history: Boolean to include the historical dates in the data
            frame for predictions.
        :return: A pd.DataFrame with the forecast components.
        """
        col_id = df["ts_id"].iloc[0]
        future_pd = self.model(col_id).predict(df)
        future_pd[self._id_cols] = df[self._id_cols].iloc[0]
        return future_pd

    def predict_timeseries(self, horizon: int = None, include_history: bool = True) -> pd.DataFrame:
        """
        Predict using the prophet model.
        :param horizon: Int number of periods to forecast forward.
        :param include_history: Boolean to include the historical dates in the data
            frame for predictions.
        :return: A pd.DataFrame with the forecast components.
        """
        horizon=horizon or self._horizon
        end_time = pd.Timestamp(self._timeseries_end)
        future_df = make_future_dataframe(
            start_time=self._timeseries_starts,
            end_time=end_time,
            horizon=horizon,
            frequency=self._frequency,
            include_history=include_history,
            groups=self._model_json.keys(),
            identity_column_names=self._id_cols
        )
        future_df["ts_id"] = future_df[self._id_cols].apply(tuple, axis=1)
        return future_df.groupby(self._id_cols).apply(lambda df: self._predict_impl(df, horizon, include_history)).reset_index()

    @staticmethod
    def get_reserved_cols() -> List[str]:
        """
        Get the list of reserved columns for prophet.
        :return: List of the reserved column names
        """
        reserved_names = [
            "trend", "additive_terms", "daily", "weekly", "yearly",
            "holidays", "zeros", "extra_regressors_additive", "yhat",
            "extra_regressors_multiplicative", "multiplicative_terms",
        ]
        rn_l = [n + "_lower" for n in reserved_names]
        rn_u = [n + "_upper" for n in reserved_names]
        reserved_names.extend(rn_l)
        reserved_names.extend(rn_u)
        reserved_names.extend(["y", "cap", "floor", "y_scaled", "cap_scaled"])
        return reserved_names

    def model_predict(self, df: pd.DataFrame, horizon: int = None) -> pd.DataFrame:
        """
        Predict API used for pandas UDF.
        :param df: Input dataframe.
        :param horizon: Int number of periods to forecast forward.
        :return: A pd.DataFrame with the forecast components.
        """
        df["ts_id"] = df[self._id_cols].apply(tuple, axis=1)
        forecast_df = self._predict_impl(df, horizon)
        return_cols = self.get_reserved_cols() + ["ds"] + self._id_cols
        result_df = pd.DataFrame(columns=return_cols)
        result_df = pd.concat([result_df, forecast_df])
        return result_df[return_cols]

    def predict(self, context: mlflow.pyfunc.model.PythonModelContext, model_input: pd.DataFrame) -> pd.Series:
        """
        Predict API from mlflow.pyfunc.PythonModel
        :param context: A :class:`~PythonModelContext` instance containing artifacts that the model
                        can use to perform inference.
        :param model_input: Input dataframe
        :return: A pd.DataFrame with the forecast components.
        """
        self._validate_cols(model_input, self._id_cols + [self._time_col])
        test_df = model_input.copy()
        test_df["ts_id"] = test_df[self._id_cols].apply(tuple, axis=1)
        test_df.rename(columns={self._time_col: "ds"}, inplace=True)

        def model_prediction(df):
            model = self.model(df["ts_id"].iloc[0])
            if model:
                predicts = model.predict(df)
                # We have to explicitly assign the id columns to avoid KeyError when model_input
                # only has one row. For multi-rows model_input, the ts_id will be kept as index
                # after groupby(self._id_cols).apply(...) and we can retrieve it by reset_index, but
                # for one-row model_input the id columns are missing from index.
                predicts[self._id_cols] = df.name
                return predicts
        predict_df = test_df.groupby(self._id_cols).apply(model_prediction).reset_index(drop=True)
        return_df = test_df.merge(predict_df, how="left", on=["ds"] + self._id_cols)
        return return_df["yhat"]


def mlflow_prophet_log_model(prophet_model: Union[ProphetModel, MultiSeriesProphetModel],
                             sample_input: pd.DataFrame = None) -> None:
    """
    Log the model to mlflow
    :param prophet_model: Prophet model wrapper
    :param sample_input: sample input Dataframes for model inference
    """
    mlflow_forecast_log_model(prophet_model, sample_input)
