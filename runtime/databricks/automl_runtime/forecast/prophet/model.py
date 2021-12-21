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
from typing import Dict, List, Union

import cloudpickle
import mlflow
import pandas as pd
import prophet

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

OFFSET_ALIAS_MAP = {
    "W": "W",
    "d": "D",
    "D": "D",
    "days": "D",
    "day": "D",
    "hours": "H",
    "hour": "H",
    "hr": "H",
    "h": "H",
    "H": "H",
    "m": "min",
    "minute": "min",
    "min": "min",
    "minutes": "min",
    "T": "T",
    "S": "S",
    "seconds": "S",
    "sec": "S",
    "second": "S"
}

PROPHET_CONDA_ENV = {
    "channels": ["conda-forge"],
    "dependencies": [
        {
            "pip": [
                f"prophet=={prophet.__version__}",
                f"cloudpickle=={cloudpickle.__version__}",
                f"databricks-automl-runtime==0.2.5"
            ]
        }
    ],
    "name": "fbp_env",
}


class ProphetModel(mlflow.pyfunc.PythonModel):
    """
    Prophet mlflow model wrapper for univariate forecasting.
    """
    def __init__(self, model_json: Union[Dict[str, str], str], horizon: int, frequency: str,
                 time_col: str, target_col: str) -> None:
        """
        Initialize the mlflow Python model wrapper for mlflow
        :param model_json: json string of the Prophet model or
        the dictionary of json strings of Prophet model for multi-series forecasting
        :param horizon: Int number of periods to forecast forward.
        :param frequency: the frequency of the time series
        :param time_col: the column name of the time column
        :param target_col: the column name of the predict target column
        """
        self._model_json = model_json
        self._horizon = horizon
        self._frequency = frequency
        self._time_col = time_col
        self._target_col = target_col
        super().__init__()

    def _validate_cols(self, df: pd.DataFrame, required_cols: List[str]):
        df_cols = set(df.columns)
        required_cols_set = set(required_cols)
        if not required_cols_set.issubset(df_cols):
            raise MlflowException(
                message=(
                    f"Input data columns '{list(df_cols)}' do not contain the required columns '{required_cols}'"
                ),
                error_code=INVALID_PARAMETER_VALUE,
            )

    def load_context(self, context: mlflow.pyfunc.model.PythonModelContext) -> None:
        """
        Loads artifacts from the specified :class:`~PythonModelContext` that can be used
        :param context: A :class:`~PythonModelContext` instance containing artifacts that the model
                        can use to perform inference.
        """
        from prophet import Prophet  # noqa: F401
        return

    def model(self) -> prophet.forecaster.Prophet:
        """
        Deserialize a Prophet model from json string
        :return: Prophet model
        """
        from prophet.serialize import model_from_json
        return model_from_json(self._model_json)

    def _make_future_dataframe(self, horizon: int) -> pd.DataFrame:
        """
        Generate future dataframe by calling the API from prophet
        :param horizon: Int number of periods to forecast forward.
        :return: pd.Dataframe that extends forward from the end of self.history for the
        requested number of periods.
        """
        return self.model().make_future_dataframe(periods=horizon, freq=OFFSET_ALIAS_MAP[self._frequency])

    def _predict_impl(self, horizon: int = None) -> pd.DataFrame:
        """
        Predict using the API from prophet model.
        :param horizon: Int number of periods to forecast forward.
        :return: A pd.DataFrame with the forecast components.
        """
        future_pd = self._make_future_dataframe(horizon=horizon or self._horizon)
        return self.model().predict(future_pd)

    def predict_timeseries(self, horizon: int = None) -> pd.DataFrame:
        """
        Predict using the prophet model.
        :param horizon: Int number of periods to forecast forward.
        :return: A pd.DataFrame with the forecast components.
        """
        return self._predict_impl(horizon)

    def predict(self, context: mlflow.pyfunc.model.PythonModelContext, model_input: pd.DataFrame) -> pd.DataFrame:
        """
        Predict API from mlflow.pyfunc.PythonModel
        :param context: A :class:`~PythonModelContext` instance containing artifacts that the model
                        can use to perform inference.
        :param model_input: Input dataframe
        :return: A pd.DataFrame with the forecast components.
        """
        self._validate_cols(model_input, [self._time_col])
        model_input.rename(columns={self._time_col: "ds"}, inplace=True)
        predict_df = self.model().predict(model_input)
        return predict_df["yhat"]


class MultiSeriesProphetModel(ProphetModel):
    """
    Prophet mlflow model wrapper for multi-series forecasting.
    """
    def __init__(self, model_json: Dict[str, str], timeseries_starts: Dict[str, pd.Timestamp],
                 timeseries_end: str, horizon: int, frequency: str, time_col: str,
                 target_col: str, id_cols: List[str],
                 ) -> None:
        """
        Initialize the mlflow Python model wrapper for mlflow
        :param model_json: the dictionary of json strings of Prophet model for multi-series forecasting
        :param timeseries_starts: the dictionary of pd.Timestamp as the starting time of each time series
        :param timeseries_end: the end time of the time series
        :param horizon: Int number of periods to forecast forward
        :param frequency: the frequency of the time series
        :param time_col: the column name of the time column
        :param target_col: the column name of the predict target column
        :param id_cols: the column names of the identity columns for multi-series time series
        """
        super().__init__(model_json, horizon, frequency, time_col, target_col)
        self._frequency = frequency
        self._timeseries_end = timeseries_end
        self._timeseries_starts = timeseries_starts
        self._id_cols = id_cols

    def model(self, id: str) -> prophet.forecaster.Prophet:
        """
        Deserialize one Prophet model from json string based on the id
        :param id: identity for the Prophet model
        :return: Prophet model
        """
        from prophet.serialize import model_from_json
        return model_from_json(self._model_json[id])

    def _make_future_dataframe(self, id: str, horizon: int) -> pd.DataFrame:
        """
        Generate future dataframe for one model by calling the API from prophet
        :param id: identity for the Prophet model
        :param horizon: Int number of periods to forecast forward
        :return: pd.Dataframe that extends forward from the end of self.history for the
        requested number of periods.
        """
        start_time = self._timeseries_starts[id]
        end_time = pd.Timestamp(self._timeseries_end)
        date_rng = pd.date_range(
            start=start_time,
            end=end_time + pd.Timedelta(value=horizon, unit=self._frequency),
            freq=OFFSET_ALIAS_MAP[self._frequency]
        )
        return pd.DataFrame(date_rng, columns=["ds"])

    def _predict_impl(self, df: pd.DataFrame, horizon: int = None) -> pd.DataFrame:
        """
        Predict using the API from prophet model.
        :param df: input dataframe
        :param horizon: Int number of periods to forecast forward.
        :return: A pd.DataFrame with the forecast components.
        """
        col_id = str(df["ts_id"].iloc[0])
        future_pd = self._make_future_dataframe(horizon=horizon or self._horizon, id=col_id)
        return self.model(col_id).predict(future_pd)

    def predict_timeseries(self, horizon: int = None) -> pd.DataFrame:
        """
        Predict using the prophet model.
        :param horizon: Int number of periods to forecast forward.
        :return: A pd.DataFrame with the forecast components.
        """
        ids = pd.DataFrame(self._model_json.keys(), columns=["ts_id"])
        return ids.groupby("ts_id").apply(lambda df: self._predict_impl(df, horizon)).reset_index()

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
        forecast_df = self._predict_impl(df,  horizon)
        return_cols = self.get_reserved_cols() + ["ds", "ts_id"]
        result_df = pd.DataFrame(columns=return_cols)
        result_df = pd.concat([result_df, forecast_df])
        result_df["ts_id"] = str(df["ts_id"].iloc[0])
        return result_df[return_cols]

    def predict(self, context: mlflow.pyfunc.model.PythonModelContext, model_input: pd.DataFrame) -> pd.DataFrame:
        """
        Predict API from mlflow.pyfunc.PythonModel
        :param context: A :class:`~PythonModelContext` instance containing artifacts that the model
                        can use to perform inference.
        :param model_input: Input dataframe
        :return: A pd.DataFrame with the forecast components.
        """
        self._validate_cols(model_input, self._id_cols + [self._time_col])
        model_input["ts_id"] = model_input[self._id_cols].agg('-'.join, axis=1)
        model_input.rename(columns={self._time_col: "ds"}, inplace=True)
        group_cols = ["ts_id"] + self._id_cols
        predict_df = model_input.groupby(group_cols).apply(lambda df: self.model(df.name[0]).predict(df)).reset_index()
        return_df = model_input.merge(predict_df, how="left", on=self._id_cols)
        return return_df["yhat"]


def mlflow_prophet_log_model(prophet_model: Union[ProphetModel, MultiSeriesProphetModel]) -> None:
    """
    Log the model to mlflow
    :param prophet_model: Prophet model wrapper
    """
    mlflow.pyfunc.log_model("model", conda_env=PROPHET_CONDA_ENV, python_model=prophet_model)
