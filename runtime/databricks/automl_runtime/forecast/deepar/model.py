#
# Copyright (C) 2024 Databricks, Inc.
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
from typing import List, Optional

import category_encoders
import gluonts
import mlflow
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.common import ListDataset
from gluonts.torch.model.predictor import PyTorchPredictor
from mlflow.utils.environment import _mlflow_conda_env

from databricks.automl_runtime import version
from databricks.automl_runtime.forecast.model import ForecastModel, mlflow_forecast_log_model
from databricks.automl_runtime.forecast.deepar.utils import set_index_and_fill_missing_time_steps
from databricks.automl_runtime.forecast.utils import apply_preprocess_func

DEEPAR_ADDITIONAL_PIP_DEPS = [
    f"gluonts[torch]=={gluonts.__version__}",
    f"pandas=={pd.__version__}",
    f"category_encoders=={category_encoders.__version__}",
    f"databricks-automl-runtime=={version.__version__}"
]

DEEPAR_CONDA_ENV = _mlflow_conda_env(
    additional_pip_deps=DEEPAR_ADDITIONAL_PIP_DEPS
)


class DeepARModel(ForecastModel):
    """
    DeepAR mlflow model wrapper for forecasting.
    """

    def __init__(self, model: PyTorchPredictor, horizon: int, frequency_unit: str, frequency_quantity: int,
                 num_samples: int,
                 target_col: str, time_col: str,
                 id_cols: Optional[List[str]] = None,
                 feature_cols: Optional[List[str]] = None,
                 split_col: Optional[str] = None,
                 preprocess_func: Optional[callable] = None) -> None:
        """
        Initialize the DeepAR mlflow Python model wrapper
        :param model: DeepAR model
        :param horizon: the number of periods to forecast forward
        :param frequency_unit: the frequency unit of the time series
        :param frequency_quantity: the frequency quantity of the time series
        :param num_samples: the number of samples to draw from the distribution
        :param target_col: the target column name
        :param time_col: the time column name
        :param id_cols: the column names of the identity columns for multi-series time series; None for single series
        :param feature_cols: the column names of the covariate feature columns; None if no covariates
        :param split_col: Optional column name of the split columns
        :param preprocess_func: Optional callable function for preprocessing input data
        """

        super().__init__()
        self._model = model
        self._horizon = horizon
        self._frequency_unit = frequency_unit
        self._frequency_quantity = frequency_quantity
        self._num_samples = num_samples
        self._target_col = target_col
        self._time_col = time_col
        self._id_cols = id_cols
        self._feature_cols = feature_cols
        self._split_col = split_col
        self._preprocess_func = preprocess_func

    @property
    def model_env(self):
        return DEEPAR_CONDA_ENV

    def predict(self,
                context: mlflow.pyfunc.model.PythonModelContext,
                model_input: pd.DataFrame) -> pd.DataFrame:
        """
        Predict the future dataframe given the history dataframe
        :param context: A :class:`~PythonModelContext` instance containing artifacts that the model
                        can use to perform inference.
        :param model_input: Input dataframe that contains the history data
        :return: predicted pd.DataFrame that starts after the last timestamp in the input dataframe,
                                and predicts the horizon using the mean of the samples
        """
        model_input_preprocessed = model_input.copy()
        if self._preprocess_func and self._split_col:
            if self._id_cols is not None:
                # multi-series: combine id columns
                model_input_preprocessed['ts_id'] = model_input_preprocessed[self._id_cols].astype(str).agg('-'.join,
                                                                                                            axis=1)
            # Apply the custom preprocessing function, which may use the split column
            # (e.g., to filter or transform data differently for train/test)
            model_input_preprocessed = apply_preprocess_func(model_input_preprocessed,
                                                             self._preprocess_func,
                                                             self._split_col)
            # Rejoin the original target column after preprocessing in case it was dropped or transformed.
            # This ensures the model still has the correct target values for forecasting.
            model_input_preprocessed = model_input_preprocessed.join(model_input[self._target_col])

        required_cols = [self._target_col, self._time_col]
        if self._id_cols:
            required_cols += self._id_cols
        self._validate_cols(model_input, required_cols)

        forecast_sample_list = self.predict_samples(model_input_preprocessed, num_samples=self._num_samples)

        pred_df = pd.concat(
            [
                forecast.mean_ts.rename('yhat').reset_index().assign(item_id=forecast.item_id)
                for forecast in forecast_sample_list
            ],
            ignore_index=True
        )

        pred_df = pred_df.rename(columns={'index': self._time_col})
        if self._id_cols:
            id_col_name = '-'.join(self._id_cols)
            pred_df = pred_df.rename(columns={'item_id': id_col_name})
        else:
            pred_df = pred_df.drop(columns='item_id')

        pred_df = self._period_to_timestamp(pred_df=pred_df)

        return pred_df

    def predict_samples(self,
                        model_input: pd.DataFrame,
                        num_samples: int = None) -> List[gluonts.model.forecast.SampleForecast]:
        """
        Predict the future samples given the history dataframe
        :param model_input: Input dataframe that contains the history data
        :param num_samples: the number of samples to draw from the distribution
        :return: List of SampleForecast, where each SampleForecast contains num_samples sampled forecasts
        """
        if num_samples is None:
            num_samples = self._num_samples

        # Group by the time column in case there are multiple rows for each time column,
        # for example, the user didn't provide all the identity columns for a multi-series dataset
        group_cols = [self._time_col]
        if self._id_cols:
            group_cols += self._id_cols
        # Prepare aggregation dictionary
        agg_dict = {self._target_col: "mean"}
        if self._feature_cols:
            # When grouping time series, aggregate feature columns as well.
            # - Numeric features: averaged across duplicates (e.g. multiple sensors reporting same timestamp)
            # - Categorical features: take the first value (arbitrary but consistent)
            # This ensures a single feature vector per time step before passing to GluonTS.
            for feature_col in self._feature_cols:
                if pd.api.types.is_numeric_dtype(model_input[feature_col]):
                    agg_dict[feature_col] = "mean"
                else:
                    agg_dict[feature_col] = "first"

        model_input = model_input.groupby(group_cols).agg(agg_dict).reset_index()

        # Ensure the time index is continuous and all covariates are aligned with target steps.
        # This also fills missing timestamps, which DeepAR requires for consistent sequence input.
        model_input_transformed = set_index_and_fill_missing_time_steps(model_input,
                                                                        self._time_col,
                                                                        self._frequency_unit,
                                                                        self._frequency_quantity,
                                                                        self._id_cols,
                                                                        self._feature_cols)
        if self._feature_cols:
            # GluonTS's PandasDataset does not support dynamic features.
            # Switch to ListDataset when feature columns are provided.
            list_dataset = []

            if isinstance(model_input_transformed, dict):
                # Multi-series: iterate over each series
                for ts_id, df in model_input_transformed.items():
                    # GluonTS expects dynamic real-valued features with shape (num_features, time_length).
                    # Transpose from DataFrame shape (time_length, num_features) -> (num_features, time_length).
                    # These features must align exactly with each timestamp in the target.
                    target_array = df[self._target_col].dropna().to_numpy()  # keep NaNs for horizon
                    feat_array = df[self._feature_cols].to_numpy().T  # transpose for GluonTS
                    list_dataset.append({
                        "item_id": ts_id,
                        "start": df.index[0],
                        "target": target_array,
                        "feat_dynamic_real": feat_array
                    })
            else:
                # Single-series
                df = model_input_transformed
                target_array = df[self._target_col].dropna().to_numpy()
                feat_array = df[self._feature_cols].to_numpy().T
                list_dataset.append({
                    "start": df.index[0],
                    "target": target_array,
                    "feat_dynamic_real": feat_array
                })

            freq_str = f"{self._frequency_quantity}{self._frequency_unit}"

            test_ds = ListDataset(list_dataset, freq=freq_str)
        else:
            test_ds = PandasDataset(model_input_transformed, target=self._target_col)

        forecast_iter = self._model.predict(test_ds, num_samples=num_samples)
        forecast_sample_list = list(forecast_iter)

        return forecast_sample_list

    def _period_to_timestamp(self, pred_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert the period to timestamp for the prediction dataframe
        If the frequency unit is 'W', use the end of the week,
        otherwise convert to timestamp.
        :param pred_df: prediction dataframe
        :return: prediction dataframe with timestamp
        """
        if self._frequency_unit == 'W':
            pred_df[self._time_col] = pred_df[self._time_col].dt.end_time.dt.normalize()
        else:
            pred_df[self._time_col] = pred_df[self._time_col].dt.to_timestamp()
        return pred_df


def mlflow_deepar_log_model(deepar_model: DeepARModel,
                            sample_input: pd.DataFrame = None) -> None:
    """
    Log the DeepAR model to mlflow
    :param deepar_model: DeepAR mlflow PythonModel wrapper
    :param sample_input: sample input Dataframes for model inference
    """
    mlflow_forecast_log_model(deepar_model, sample_input)
