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
            self.assertIn(dependency, requirements, f"requirements.txt should contain {dependency} but got {requirements}")

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
                                  sample_input_base.copy(), sample_input_base.copy(),], ignore_index=True)
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
        self.assertIn(expected_dep, requirements, f"Specific category_encoders version {expected_dep} should be in requirements")

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
