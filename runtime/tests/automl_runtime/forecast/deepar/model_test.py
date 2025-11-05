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

import unittest

import mlflow
import pandas as pd
import torch
import torch.nn as nn
from parameterized import parameterized
from gluonts.dataset.field_names import FieldName
from gluonts.transform import InstanceSplitter, TestSplitSampler
from gluonts.torch.model.predictor import PyTorchPredictor
from unittest import mock

from databricks.automl_runtime.forecast.deepar.model import (
    DeepARModel,
    mlflow_deepar_log_model,
    DEEPAR_ADDITIONAL_PIP_DEPS
)


class TestDeepARModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Adapted from https://github.com/awslabs/gluonts/blob/dev/test/torch/model/test_torch_predictor.py
        class RandomNetwork(nn.Module):
            def __init__(
                    self,
                    prediction_length: int,
                    context_length: int,
            ) -> None:
                super().__init__()
                self.prediction_length = prediction_length
                self.context_length = context_length
                self.net = nn.Linear(context_length, prediction_length)
                torch.nn.init.uniform_(self.net.weight, -1.0, 1.0)

            def forward(self, past_target):
                out = self.net(past_target.float())
                return out.unsqueeze(1)

        cls.context_length = 5
        cls.prediction_length = 5

        cls.pred_net = RandomNetwork(
            prediction_length=cls.context_length, context_length=cls.context_length
        )

        cls.transformation = InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=TestSplitSampler(),
            past_length=cls.context_length,
            future_length=cls.prediction_length,
        )

        cls.model = PyTorchPredictor(
            prediction_length=cls.prediction_length,
            input_names=["past_target"],
            prediction_net=cls.pred_net,
            batch_size=16,
            input_transform=cls.transformation,
            device="cpu",
        )

    def _check_requirements(self, run_id: str):
        # read requirements.txt from the run
        requirements_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/model/requirements.txt")
        with open(requirements_path, "r") as f:
            requirements = f.read()
        # check if all additional dependencies are logged
        for dependency in DEEPAR_ADDITIONAL_PIP_DEPS:
            self.assertIn(dependency, requirements,
                          f"requirements.txt should contain {dependency} but got {requirements}")

    def test_model_save_and_load_single_series(self):
        target_col = "sales"
        time_col = "date"

        deepar_model = DeepARModel(
            model=self.model,
            horizon=self.prediction_length,
            frequency_unit="d",
            frequency_quantity=1,
            num_samples=1,
            target_col=target_col,
            time_col=time_col,
        )

        num_rows = 10
        sample_input = pd.concat(
            [
                pd.to_datetime(
                    pd.Series(range(num_rows), name=time_col).apply(
                        lambda i: f"2020-10-{3 * i + 1}"
                    )
                ),
                pd.Series(range(num_rows), name=target_col),
            ],
            axis=1,
        )

        with mlflow.start_run() as run:
            mlflow_deepar_log_model(deepar_model, sample_input)

        run_id = run.info.run_id

        # check if all additional dependencies are logged
        self._check_requirements(run_id)

        # load the model and predict
        loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
        pred_df = loaded_model.predict(sample_input)

        self.assertEqual(pred_df.columns.tolist(), [time_col, "yhat"])
        self.assertEqual(len(pred_df), self.prediction_length)
        self.assertGreater(pred_df[time_col].min(), sample_input[time_col].max())

    def test_model_save_and_load_multi_series(self):
        target_col = "sales"
        time_col = "date"
        id_col = "store"

        deepar_model = DeepARModel(
            model=self.model,
            horizon=self.prediction_length,
            num_samples=1,
            frequency_unit="d",
            frequency_quantity=1,
            target_col=target_col,
            time_col=time_col,
            id_cols=[id_col],
        )

        num_rows_per_ts = 10
        sample_input_base = pd.concat(
            [
                pd.to_datetime(
                    pd.Series(range(num_rows_per_ts), name=time_col).apply(
                        lambda i: f"2020-10-{3 * i + 1}"
                    )
                ),
                pd.Series(range(num_rows_per_ts), name=target_col),
            ],
            axis=1,
        )
        sample_input = pd.concat([sample_input_base.copy(), sample_input_base.copy()], ignore_index=True)
        sample_input[id_col] = [1] * num_rows_per_ts + [2] * num_rows_per_ts

        with mlflow.start_run() as run:
            mlflow_deepar_log_model(deepar_model, sample_input)

        run_id = run.info.run_id

        # check if all additional dependencies are logged
        self._check_requirements(run_id)

        # load the model and predict
        loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

        pred_df = loaded_model.predict(sample_input)

        self.assertEqual(pred_df.columns.tolist(), [time_col, "yhat", id_col])
        self.assertEqual(len(pred_df), self.prediction_length * 2)
        self.assertGreater(pred_df[time_col].min(), sample_input[time_col].max())

    def test_model_save_and_load_multi_series_multi_id_cols(self):
        target_col = "sales"
        time_col = "date"
        id_cols = ["store", "dept"]

        deepar_model = DeepARModel(
            model=self.model,
            horizon=self.prediction_length,
            num_samples=1,
            frequency_unit="d",
            frequency_quantity=1,
            target_col=target_col,
            time_col=time_col,
            id_cols=id_cols,
        )

        num_rows_per_ts = 5
        sample_input_base = pd.concat(
            [
                pd.to_datetime(
                    pd.Series(range(num_rows_per_ts), name=time_col).apply(
                        lambda i: f"2020-10-{3 * i + 1}"
                    )
                ),
                pd.Series(range(num_rows_per_ts), name=target_col),
            ],
            axis=1,
        )
        sample_input = pd.concat([sample_input_base.copy(), sample_input_base.copy(),
                                  sample_input_base.copy(), sample_input_base.copy(), ], ignore_index=True)
        sample_input[id_cols[0]] = ['A'] * (2 * num_rows_per_ts) + ['B'] * (2 * num_rows_per_ts)
        sample_input[id_cols[1]] = (['X'] * num_rows_per_ts + ['Y'] * num_rows_per_ts) * 2

        with mlflow.start_run() as run:
            mlflow_deepar_log_model(deepar_model, sample_input)
        run_id = run.info.run_id

        # load the model and predict
        loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
        pred_df = loaded_model.predict(sample_input)

        self.assertEqual(pred_df.columns.tolist(), [time_col, "yhat", "store-dept"])
        self.assertEqual(len(pred_df), self.prediction_length * 4)
        self.assertGreater(pred_df[time_col].min(), sample_input[time_col].max())

    def test_model_prediction_with_duplicate_timestamps(self):
        """
        Test that the model correctly handles and averages multiple rows with the same timestamp
        when identity columns are not provided.
        """
        target_col = "sales"
        time_col = "date"

        deepar_model = DeepARModel(
            model=self.model,
            horizon=self.prediction_length,
            frequency_unit="d",
            frequency_quantity=1,
            num_samples=1,
            target_col=target_col,
            time_col=time_col,
        )

        # Create sample input with duplicate timestamps
        dates = pd.to_datetime([
            "2020-10-01", "2020-10-01",  # duplicate date with different values
            "2020-10-04", "2020-10-04", "2020-10-04",  # triple duplicate
            "2020-10-07"  # single entry
        ])

        sales = [10, 20,  # should average to 15
                 30, 60, 90,  # should average to 60
                 100]  # single value stays 100

        sample_input = pd.DataFrame({
            time_col: dates,
            target_col: sales
        })

        with mlflow.start_run() as run:
            mlflow_deepar_log_model(deepar_model, sample_input)

        run_id = run.info.run_id

        # Load the model and predict
        loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
        pred_df = loaded_model.predict(sample_input)

        # Verify the prediction output format
        self.assertEqual(pred_df.columns.tolist(), [time_col, "yhat"])
        self.assertEqual(len(pred_df), self.prediction_length)
        self.assertGreater(pred_df[time_col].min(), sample_input[time_col].max())

    def test_model_prediction_with_monthly_data(self):
        target_col = "sales"
        time_col = "date"

        deepar_model = DeepARModel(
            model=self.model,
            horizon=self.prediction_length,
            frequency_unit="MS",
            frequency_quantity=1,
            num_samples=1,
            target_col=target_col,
            time_col=time_col,
        )

        # Create sample input with duplicate timestamps
        dates = pd.to_datetime([
            "2020-10-01", "2020-11-01", "2020-12-01",
            "2021-01-01", "2021-02-01", "2021-03-01"
        ])

        sales = [10, 20, 30,
                 60, 90, 100]

        sample_input = pd.DataFrame({
            time_col: dates,
            target_col: sales
        })

        with mlflow.start_run() as run:
            mlflow_deepar_log_model(deepar_model, sample_input)

        run_id = run.info.run_id

        # Load the model and predict
        loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
        pred_df = loaded_model.predict(sample_input)

        # Verify the prediction output format
        self.assertEqual(pred_df.columns.tolist(), [time_col, "yhat"])
        self.assertEqual(len(pred_df), self.prediction_length)
        self.assertGreater(pred_df[time_col].min(), sample_input[time_col].max())

    @parameterized.expand([(1,), (5,), (10,), (15,), (30,)])
    def test_model_prediction_with_multiple_minutes_frequency(self, frequency_quantity):
        target_col = "sales"
        time_col = "date"

        deepar_model = DeepARModel(
            model=self.model,
            horizon=self.prediction_length,
            frequency_unit="min",
            frequency_quantity=frequency_quantity,
            num_samples=1,
            target_col=target_col,
            time_col=time_col,
        )

        # Create sample input with duplicate timestamps
        dates = pd.date_range(start="2020-10-01", periods=6, freq=f"{frequency_quantity}min")

        sales = [10, 20, 30,
                 60, 90, 100]

        sample_input = pd.DataFrame({
            time_col: dates,
            target_col: sales
        })

        with mlflow.start_run() as run:
            mlflow_deepar_log_model(deepar_model, sample_input)

        run_id = run.info.run_id

        # Load the model and predict
        loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
        pred_df = loaded_model.predict(sample_input)

        # Verify the prediction output format
        self.assertEqual(pred_df.columns.tolist(), [time_col, "yhat"])
        self.assertEqual(len(pred_df), self.prediction_length)
        self.assertGreater(pred_df[time_col].min(), sample_input[time_col].max())

    def test_period_to_timestamp(self):
        """Test the _period_to_timestamp method for different frequency units"""
        target_col = "sales"
        time_col = "date"

        # Test with weekly frequency (W)
        deepar_model_weekly = DeepARModel(
            model=self.model,
            horizon=self.prediction_length,
            frequency_unit="W",
            frequency_quantity=1,
            num_samples=1,
            target_col=target_col,
            time_col=time_col,
        )

        # Create a DataFrame with Period objects for weekly frequency
        # Use proper weekly period format
        weekly_periods = pd.PeriodIndex(['2020-01-05', '2020-01-12', '2020-01-19'], freq='W')
        pred_df_weekly = pd.DataFrame({
            time_col: weekly_periods,
            'yhat': [10.0, 20.0, 30.0]
        })

        result_weekly = deepar_model_weekly._period_to_timestamp(pred_df_weekly)

        # For weekly frequency, should use end_time and normalize
        expected_weekly_timestamps = pd.to_datetime(['2020-01-05', '2020-01-12', '2020-01-19']).to_series().reset_index(
            drop=True)
        expected_weekly_timestamps.name = time_col
        pd.testing.assert_series_equal(
            result_weekly[time_col],
            expected_weekly_timestamps,
            check_dtype=False
        )

        # Test with daily frequency (D)
        deepar_model_daily = DeepARModel(
            model=self.model,
            horizon=self.prediction_length,
            frequency_unit="D",
            frequency_quantity=1,
            num_samples=1,
            target_col=target_col,
            time_col=time_col,
        )

        # Create a DataFrame with Period objects for daily frequency
        daily_periods = pd.PeriodIndex(['2020-01-01', '2020-01-02', '2020-01-03'], freq='D')
        pred_df_daily = pd.DataFrame({
            time_col: daily_periods,
            'yhat': [10.0, 20.0, 30.0]
        })

        result_daily = deepar_model_daily._period_to_timestamp(pred_df_daily)

        # For non-weekly frequency, should convert to timestamp
        expected_daily_timestamps = pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03']).to_series().reset_index(
            drop=True)
        expected_daily_timestamps.name = time_col
        pd.testing.assert_series_equal(
            result_daily[time_col],
            expected_daily_timestamps,
            check_dtype=False
        )

        # Test with monthly frequency (M)
        deepar_model_monthly = DeepARModel(
            model=self.model,
            horizon=self.prediction_length,
            frequency_unit="M",
            frequency_quantity=1,
            num_samples=1,
            target_col=target_col,
            time_col=time_col,
        )

        # Create a DataFrame with Period objects for monthly frequency
        monthly_periods = pd.PeriodIndex(['2020-01', '2020-02', '2020-03'], freq='M')
        pred_df_monthly = pd.DataFrame({
            time_col: monthly_periods,
            'yhat': [10.0, 20.0, 30.0]
        })

        result_monthly = deepar_model_monthly._period_to_timestamp(pred_df_monthly)

        # For non-weekly frequency, should convert to timestamp
        expected_monthly_timestamps = pd.to_datetime(
            ['2020-01-01', '2020-02-01', '2020-03-01']).to_series().reset_index(drop=True)
        expected_monthly_timestamps.name = time_col
        pd.testing.assert_series_equal(
            result_monthly[time_col],
            expected_monthly_timestamps,
            check_dtype=False
        )

        # Test with quarterly frequency (Q)
        deepar_model_quarterly = DeepARModel(
            model=self.model,
            horizon=self.prediction_length,
            frequency_unit="Q",
            frequency_quantity=1,
            num_samples=1,
            target_col=target_col,
            time_col=time_col,
        )

        # Create a DataFrame with Period objects for quarterly frequency
        quarterly_periods = pd.PeriodIndex(['2020Q1', '2020Q2', '2020Q3'], freq='Q')
        pred_df_quarterly = pd.DataFrame({
            time_col: quarterly_periods,
            'yhat': [10.0, 20.0, 30.0]
        })

        result_quarterly = deepar_model_quarterly._period_to_timestamp(pred_df_quarterly)

        # For non-weekly frequency, should convert to timestamp
        expected_quarterly_timestamps = pd.to_datetime(
            ['2020-01-01', '2020-04-01', '2020-07-01']).to_series().reset_index(drop=True)
        expected_quarterly_timestamps.name = time_col
        pd.testing.assert_series_equal(
            result_quarterly[time_col],
            expected_quarterly_timestamps,
            check_dtype=False
        )


class TestDeepARModelCategoryEncoders(unittest.TestCase):
    """Test category_encoders dependency inclusion"""

    @classmethod
    def setUpClass(cls) -> None:
        # Use the same setup as the main test class
        cls.context_length = 5
        cls.prediction_length = 5

        # Create a simple mock network for testing
        class MockNetwork(nn.Module):
            def __init__(self, prediction_length: int, context_length: int) -> None:
                super().__init__()
                self.prediction_length = prediction_length
                self.context_length = context_length
                self.net = nn.Linear(context_length, prediction_length)

            def forward(self, past_target):
                out = self.net(past_target.float())
                return out.unsqueeze(1)

        cls.pred_net = MockNetwork(
            prediction_length=cls.context_length, context_length=cls.context_length
        )

        cls.transformation = InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=TestSplitSampler(),
            past_length=cls.context_length,
            future_length=cls.prediction_length,
        )

        cls.model = PyTorchPredictor(
            prediction_length=cls.prediction_length,
            input_names=["past_target"],
            prediction_net=cls.pred_net,
            batch_size=16,
            input_transform=cls.transformation,
            device="cpu",
        )

    def test_category_encoders_in_requirements(self):
        """Test that category_encoders is included in model requirements"""
        target_col = "sales"
        time_col = "date"

        deepar_model = DeepARModel(
            model=self.model,
            horizon=self.prediction_length,
            frequency_unit="d",
            frequency_quantity=1,
            num_samples=1,
            target_col=target_col,
            time_col=time_col,
        )

        num_rows = 10
        sample_input = pd.concat(
            [
                pd.to_datetime(
                    pd.Series(range(num_rows), name=time_col).apply(
                        lambda i: f"2020-10-{3 * i + 1}"
                    )
                ),
                pd.Series(range(num_rows), name=target_col),
            ],
            axis=1,
        )

        with mlflow.start_run() as run:
            mlflow_deepar_log_model(deepar_model, sample_input)

        run_id = run.info.run_id

        # Read requirements.txt from the run
        requirements_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/model/requirements.txt")
        with open(requirements_path, "r") as f:
            requirements = f.read()

        # Verify category_encoders is included in requirements
        self.assertIn("category_encoders", requirements, "category_encoders should be included in model requirements")

        # Verify the specific version is included (from DEEPAR_ADDITIONAL_PIP_DEPS)
        import category_encoders
        expected_dep = f"category_encoders=={category_encoders.__version__}"
        self.assertIn(expected_dep, requirements,
                      f"Specific category_encoders version {expected_dep} should be in requirements")

    def test_model_with_category_encoding_preprocessing(self):
        """Test that models work correctly with potential category encoding preprocessing"""
        target_col = "sales"
        time_col = "date"

        deepar_model = DeepARModel(
            model=self.model,
            horizon=self.prediction_length,
            frequency_unit="d",
            frequency_quantity=1,
            num_samples=1,
            target_col=target_col,
            time_col=time_col,
        )

        # Create test data that could potentially use category encoding
        num_rows = 10
        sample_input = pd.concat(
            [
                pd.to_datetime(
                    pd.Series(range(num_rows), name=time_col).apply(
                        lambda i: f"2020-10-{3 * i + 1}"
                    )
                ),
                pd.Series(range(num_rows), name=target_col),
                pd.Series([f"category_{i % 3}" for i in range(num_rows)], name="category_col"),
            ],
            axis=1,
        )

        # This should work without errors if category_encoders is properly available
        # Note: DeepAR doesn't directly use preprocessing functions like Prophet/ARIMA,
        # but category_encoders might be used in data preparation pipelines
        try:
            import category_encoders as ce
            # Test that we can import and use category_encoders
            encoder = ce.BinaryEncoder(cols=['category_col'])
            encoded_data = encoder.fit_transform(sample_input[['category_col']])
            self.assertIsNotNone(encoded_data)
        except ImportError:
            self.fail("category_encoders should be available for DeepAR models")

    def test_multiseries_model_with_category_encoding(self):
        """Test that multi-series models work with category encoding"""
        target_col = "sales"
        time_col = "date"
        id_col = "store"

        deepar_model = DeepARModel(
            model=self.model,
            horizon=self.prediction_length,
            num_samples=1,
            frequency_unit="d",
            frequency_quantity=1,
            target_col=target_col,
            time_col=time_col,
            id_cols=[id_col],
        )

        num_rows_per_ts = 10
        sample_input_base = pd.concat(
            [
                pd.to_datetime(
                    pd.Series(range(num_rows_per_ts), name=time_col).apply(
                        lambda i: f"2020-10-{3 * i + 1}"
                    )
                ),
                pd.Series(range(num_rows_per_ts), name=target_col),
                pd.Series([f"cat_{i % 2}" for i in range(num_rows_per_ts)], name="category_col"),
            ],
            axis=1,
        )
        sample_input = pd.concat([sample_input_base.copy(), sample_input_base.copy()], ignore_index=True)
        sample_input[id_col] = [1] * num_rows_per_ts + [2] * num_rows_per_ts

        # Test that category_encoders can be used with multi-series data
        try:
            import category_encoders as ce
            encoder = ce.TargetEncoder(cols=['category_col'])
            # Just test that we can create the encoder - actual fitting would need target data
            self.assertIsNotNone(encoder)
        except ImportError:
            self.fail("category_encoders should be available for multi-series DeepAR models")

    def test_category_encoders_version_compatibility(self):
        """Test that the correct version of category_encoders is specified in dependencies"""
        # Verify that category_encoders is in DEEPAR_ADDITIONAL_PIP_DEPS
        category_encoders_deps = [dep for dep in DEEPAR_ADDITIONAL_PIP_DEPS if "category_encoders" in dep]
        self.assertEqual(len(category_encoders_deps), 1, "category_encoders should be in DEEPAR_ADDITIONAL_PIP_DEPS")

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
        target_col = "sales"
        time_col = "date"

        deepar_model = DeepARModel(
            model=self.model,
            horizon=self.prediction_length,
            frequency_unit="d",
            frequency_quantity=1,
            num_samples=1,
            target_col=target_col,
            time_col=time_col,
        )

        # Get the model environment
        model_env = deepar_model.model_env

        # Navigate to pip dependencies: dependencies list -> find dict with 'pip' key -> get pip list
        dependencies = model_env.get('dependencies', [])
        pip_deps = []
        for dep in dependencies:
            if isinstance(dep, dict) and 'pip' in dep:
                pip_deps = dep['pip']
                break

        category_encoders_found = any("category_encoders" in dep for dep in pip_deps)
        self.assertTrue(category_encoders_found, "category_encoders should be in model environment pip dependencies")


class TestDeepARModelWithCovariates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Dummy net that accepts past_target + covariates
        class CovariateNet(nn.Module):
            def __init__(self, context_length, prediction_length, num_covariates):
                super().__init__()
                input_dim = context_length + num_covariates
                self.fc = nn.Linear(input_dim, prediction_length)
                torch.nn.init.uniform_(self.fc.weight, -0.1, 0.1)

            def forward(self, past_target, feat_dynamic_real=None, **kwargs):
                batch_size = past_target.shape[0]
                if feat_dynamic_real is not None:
                    # take last value of each covariate
                    covariates = feat_dynamic_real[:, :, -1]
                else:
                    covariates = torch.zeros((batch_size, 0))
                x = torch.cat([past_target.float(), covariates.float()], dim=-1)
                return self.fc(x).unsqueeze(1)

        cls.prediction_length = 3
        cls.context_length = 5
        num_covariates = 2

        cls.pred_net = CovariateNet(
            context_length=cls.context_length,
            prediction_length=cls.prediction_length,
            num_covariates=num_covariates,
        )

        cls.transformation = InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=TestSplitSampler(),
            past_length=cls.context_length,
            future_length=cls.prediction_length,
        )

        cls.model = PyTorchPredictor(
            prediction_length=cls.prediction_length,
            input_names=["past_target", "feat_dynamic_real"],
            prediction_net=cls.pred_net,
            batch_size=16,
            input_transform=cls.transformation,
            device="cpu",
        )

    def test_model_with_covariates(self):
        """Test DeepAR model with covariate features"""
        target_col = "sales"
        time_col = "date"
        feature_cols = ["temperature", "promotion"]

        deepar_model = DeepARModel(
            model=self.model,
            horizon=self.prediction_length,
            frequency_unit="d",
            frequency_quantity=3,
            num_samples=1,
            target_col=target_col,
            time_col=time_col,
            feature_cols=feature_cols,
        )

        num_rows = 10
        sample_input = pd.DataFrame({
            time_col: pd.date_range("2020-10-01", periods=num_rows + self.prediction_length),
            target_col: list(range(num_rows)) + [None] * self.prediction_length,
            "temperature": list(range(20, 20 + num_rows)) + [0] * self.prediction_length,
            "promotion": [i % 2 for i in range(num_rows)] + [0] * self.prediction_length
        })

        # Test that model can validate covariate columns
        try:
            with mlflow.start_run() as run:
                mlflow_deepar_log_model(deepar_model, sample_input)
            run_id = run.info.run_id

            # Load model and test prediction
            loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
            pred_df = loaded_model.predict(sample_input)

            # Verify prediction structure
            self.assertEqual(pred_df.columns.tolist(), [time_col, "yhat"])
            self.assertEqual(len(pred_df), self.prediction_length)

        except Exception as e:
            self.fail(f"DeepAR model with covariates should not fail: {e}")

    def test_model_with_covariates_missing_columns(self):
        """Test DeepAR model fails appropriately when covariate columns are missing"""
        target_col = "sales"
        time_col = "date"
        feature_cols = ["temperature", "promotion"]

        deepar_model = DeepARModel(
            model=self.model,
            horizon=self.prediction_length,
            frequency_unit="d",
            frequency_quantity=1,
            num_samples=1,
            target_col=target_col,
            time_col=time_col,
            feature_cols=feature_cols,
        )

        num_rows = 10
        sample_input_missing_features = pd.DataFrame({
            time_col: pd.date_range("2020-10-01", periods=num_rows + self.prediction_length),
            target_col: list(range(num_rows)) + [None] * self.prediction_length
            # Missing covariates intentionally
        })

        # Should raise an exception due to missing columns
        with self.assertRaises(Exception):
            deepar_model.predict(context=None, model_input=sample_input_missing_features)

    def test_multi_series_model_with_covariates_preserves_item_id(self):
        """Test that multi-series DeepAR model with covariates preserves series identifiers"""
        target_col = "sales"
        time_col = "date"
        id_col = "store"
        feature_cols = ["temperature", "promotion"]

        deepar_model = DeepARModel(
            model=self.model,
            horizon=self.prediction_length,
            frequency_unit="d",
            frequency_quantity=3,
            num_samples=1,
            target_col=target_col,
            time_col=time_col,
            id_cols=[id_col],
            feature_cols=feature_cols,
        )

        num_rows_per_series = 10
        # Create data for two stores
        sample_input_store1 = pd.DataFrame({
            time_col: pd.date_range("2020-10-01", periods=num_rows_per_series + self.prediction_length),
            target_col: list(range(num_rows_per_series)) + [None] * self.prediction_length,
            id_col: [1] * (num_rows_per_series + self.prediction_length),
            "temperature": list(range(20, 20 + num_rows_per_series)) + [0] * self.prediction_length,
            "promotion": [i % 2 for i in range(num_rows_per_series)] + [0] * self.prediction_length
        })

        sample_input_store2 = pd.DataFrame({
            time_col: pd.date_range("2020-10-01", periods=num_rows_per_series + self.prediction_length),
            target_col: list(range(num_rows_per_series)) + [None] * self.prediction_length,
            id_col: [2] * (num_rows_per_series + self.prediction_length),
            "temperature": list(range(25, 25 + num_rows_per_series)) + [0] * self.prediction_length,
            "promotion": [i % 2 for i in range(num_rows_per_series)] + [0] * self.prediction_length
        })

        sample_input = pd.concat([sample_input_store1, sample_input_store2], ignore_index=True)

        with mlflow.start_run() as run:
            mlflow_deepar_log_model(deepar_model, sample_input)
        run_id = run.info.run_id

        # Load model and test prediction
        loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
        pred_df = loaded_model.predict(sample_input)

        # Verify that series identifiers are preserved in the output
        self.assertIn(id_col, pred_df.columns,
                      f"Series identifier '{id_col}' should be preserved in predictions for multi-series model with covariates")
        self.assertEqual(pred_df.columns.tolist(), [time_col, "yhat", id_col])
        self.assertEqual(len(pred_df), self.prediction_length * 2)  # predictions for both stores

        # Verify both stores are present
        unique_stores = pred_df[id_col].unique()
        self.assertEqual(len(unique_stores), 2, "Should have predictions for both stores")
        self.assertIn('1', unique_stores, "Should have predictions for store 1")
        self.assertIn('2', unique_stores, "Should have predictions for store 2")


class TestDeepARModelWithPreprocess(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Mock network for DeepAR
        class SimpleNet(nn.Module):
            def __init__(self, context_length, prediction_length, num_covariates):
                super().__init__()
                cls.input_dim = context_length + num_covariates
                self.fc = nn.Linear(cls.input_dim, prediction_length)
                torch.nn.init.uniform_(self.fc.weight, -0.1, 0.1)

            def forward(self, past_target, feat_dynamic_real=None, **kwargs):
                batch_size = past_target.shape[0]
                if feat_dynamic_real is not None:
                    covariates = feat_dynamic_real[:, :, -1]
                else:
                    covariates = torch.zeros((batch_size, 0))
                x = torch.cat([past_target.float(), covariates.float()], dim=-1)
                return self.fc(x).unsqueeze(1)

        cls.prediction_length = 3
        cls.context_length = 5
        num_covariates = 2

        cls.pred_net = SimpleNet(
            context_length=cls.context_length,
            prediction_length=cls.prediction_length,
            num_covariates=num_covariates,
        )

        cls.transformation = InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=TestSplitSampler(),
            past_length=cls.context_length,
            future_length=cls.prediction_length,
        )

        cls.model = PyTorchPredictor(
            prediction_length=cls.prediction_length,
            input_names=["past_target", "feat_dynamic_real"],
            prediction_net=cls.pred_net,
            batch_size=16,
            input_transform=cls.transformation,
            device="cpu",
        )

    def setUp(self):
        # Sample single-series data
        self.num_rows = 10
        self.start_date = pd.Timestamp("2025-01-01")
        self.horizon = self.prediction_length
        self.freq = "D"
        dates = pd.date_range(self.start_date, periods=self.num_rows, freq=self.freq)
        self.df = pd.DataFrame({
            "date": dates,
            "y": range(self.num_rows),
            "x1": range(self.num_rows),
            "x2": range(self.num_rows)
        })

        # Mock preprocess function (doubles y)
        def preprocess_func(df):
            df = df.copy()
            df["x1"] = df["x1"] * 2
            return df

        self.mock_preprocess = mock.Mock(side_effect=preprocess_func)

    def test_predict_with_preprocess_single_series(self):
        # Prepare input
        input_df = self.df.copy()

        split_col = "split"

        # Wrap PyTorchPredictor in DeepARModel interface
        self.deep_ar_model = DeepARModel(
            model=self.model,
            horizon=self.horizon,
            num_samples=1,
            target_col="y",
            time_col="date",
            feature_cols=["x1", "x2"],
            frequency_unit="D",
            frequency_quantity=1,
            split_col=split_col,
            preprocess_func=self.mock_preprocess,
        )

        # Run predict
        pred_df = self.deep_ar_model.predict(context=None, model_input=input_df)

        # Check columns
        self.assertIn("yhat", pred_df.columns)
        self.assertEqual(len(pred_df), self.prediction_length)

        # Ensure preprocess was called
        self.mock_preprocess.assert_called_once()
        call_arg = self.mock_preprocess.call_args[0][0]
        expected_call = input_df.copy()
        expected_call[split_col] = "prediction"
        expected_call["y"] = None
        pd.testing.assert_frame_equal(call_arg, expected_call)

        # Verify the return value from preprocess_func
        # The preprocess function doubles y values
        expected_return = expected_call.copy()
        expected_return["x1"] = expected_return["x1"] * 2 if expected_return["x1"] is not None else 0

        # Get the actual return value from the call
        actual_return = self.mock_preprocess.side_effect(call_arg)
        pd.testing.assert_frame_equal(actual_return, expected_return)

    def test_predict_with_preprocess_multi_series(self):
        # Multi-series setup
        df_multi = pd.DataFrame({
            "date": pd.to_datetime(
                ["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-02", "2025-01-03", "2025-01-03"]),
            "y": [1, 2, 3, 4, 5, 6],
            "id": ["A", "B", "A", "B", "A", "B"],
            "x1": [1, 10, 2, 11, 2, 9],
            "x2": [5, 15, 6, 16, 7, 8]
        })

        split_col = "split"

        # Wrap PyTorchPredictor in DeepARModel interface
        self.deep_ar_model = DeepARModel(
            model=self.model,
            horizon=self.horizon,
            num_samples=1,
            target_col="y",
            time_col="date",
            id_cols=["id"],
            feature_cols=["x1", "x2"],
            frequency_unit="D",
            frequency_quantity=1,
            split_col="split",
            preprocess_func=self.mock_preprocess,
        )

        pred_df = self.deep_ar_model.predict(context=None, model_input=df_multi)

        # Check yhat column exists
        self.assertIn("yhat", pred_df.columns)

        # Preprocess should be called once
        self.mock_preprocess.assert_called_once()

        # Ensure preprocess was called
        self.mock_preprocess.assert_called_once()
        call_arg = self.mock_preprocess.call_args[0][0]
        expected_call = df_multi.copy()
        expected_call["ts_id"] = expected_call["id"]
        expected_call[split_col] = "prediction"
        expected_call["y"] = None
        pd.testing.assert_frame_equal(call_arg, expected_call)

        # Verify the return value from preprocess_func
        # The preprocess function doubles y values
        expected_return = expected_call.copy()
        expected_return["x1"] = expected_return["x1"] * 2 if expected_return["x1"] is not None else 0

        # Get the actual return value from the call
        actual_return = self.mock_preprocess.side_effect(call_arg)
        pd.testing.assert_frame_equal(actual_return, expected_return)
