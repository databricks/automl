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
from parameterized import parameterized

import pandas as pd

from databricks.automl_runtime.forecast.utils import \
    generate_cutoffs, get_validation_horizon, calculate_period_differences, \
    is_frequency_consistency, make_future_dataframe, make_single_future_dataframe


class TestGetValidationHorizon(unittest.TestCase):

    def test_no_truncate(self):
        # 5 day horizon is OK for dataframe with 30 days of data
        df = pd.DataFrame(pd.date_range(start="2020-08-01", end="2020-08-30", freq="D"), columns=["ds"])
        validation_horizon = get_validation_horizon(df, 5, "D")
        self.assertEqual(validation_horizon, 5)

        # 2 week horizon is OK for dataframe with ~12 weeks of data
        df = pd.DataFrame(pd.date_range(start="2020-01-01", end="2020-04-01", freq="W"), columns=["ds"])
        validation_horizon = get_validation_horizon(df, 2, "W")
        self.assertEqual(validation_horizon, 2)

    def test_truncate(self):
        # for dataframe with 19 days of data, maximum horizon is 4 days
        df = pd.DataFrame(pd.date_range(start="2020-08-01", end="2020-08-20", freq="D"), columns=["ds"])
        validation_horizon = get_validation_horizon(df, 10, "D")
        self.assertEqual(validation_horizon, 4)

        # for dataframe with 20 days of data, maximum horizon is 5 days
        df = pd.DataFrame(pd.date_range(start="2020-08-01", end="2020-08-21", freq="D"), columns=["ds"])
        validation_horizon = get_validation_horizon(df, 10, "D")
        self.assertEqual(validation_horizon, 5)

        # for dataframe with 21 days of data, maximum horizon is 5 days
        df = pd.DataFrame(pd.date_range(start="2020-08-01", end="2020-08-22", freq="D"), columns=["ds"])
        validation_horizon = get_validation_horizon(df, 10, "D")
        self.assertEqual(validation_horizon, 5)

        # for dataframe with just under one year of data, maximum horizon is 12 weeks
        df = pd.DataFrame(pd.date_range(start="2020-01-01", end="2020-12-31", freq="W"), columns=["ds"])
        validation_horizon = get_validation_horizon(df, 20, "W")
        self.assertEqual(validation_horizon, 12)

        # for dataframe with just one year of data, maximum horizon is 3 months
        df = pd.DataFrame(pd.date_range(start="2020-01-14", periods=13, 
            freq=pd.DateOffset(months=1)), columns=["ds"])
        validation_horizon = get_validation_horizon(df, 17, "MS")
        self.assertEqual(validation_horizon, 3)

        # for dataframe with 8 year of data, maximum horizon is 2 years
        df = pd.DataFrame(pd.date_range(start="2012-01-14", periods=9, freq=pd.DateOffset(years=1)), columns=["ds"])
        validation_horizon = get_validation_horizon(df, 17, "YS")
        self.assertEqual(validation_horizon, 2)

        # for dataframe with 12 quaters of data, maximum horizon is 3 quaters.
        df = pd.DataFrame(pd.date_range(start="2012-01-14", periods=13, freq=pd.DateOffset(months=3)), columns=["ds"])
        validation_horizon = get_validation_horizon(df, 17, "QS")
        self.assertEqual(validation_horizon, 3)

    def test_truncate_logs(self):
        with self.assertLogs(logger="databricks.automl_runtime.forecast", level="INFO") as cm:
            df = pd.DataFrame(pd.date_range(start="2020-08-01", end="2020-08-20", freq="D"), columns=["ds"])
            validation_horizon = get_validation_horizon(df, 10, "D")
            self.assertIn("too long relative to dataframe's timedelta. Validation horizon will be reduced to", cm.output[0])


class TestGenerateCutoffs(unittest.TestCase):

    def setUp(self) -> None:
        self.X = pd.DataFrame(
            pd.date_range(start="2020-07-01", end="2020-08-30", freq='d'), columns=["ds"]
        ).rename_axis("y").reset_index()

    def test_generate_cutoffs_success(self):
        cutoffs = generate_cutoffs(self.X, horizon=7, unit="D", num_folds=3, seasonal_period=7)
        self.assertEqual([pd.Timestamp('2020-08-16 00:00:00'), pd.Timestamp('2020-08-19 12:00:00'), pd.Timestamp('2020-08-23 00:00:00')], cutoffs)

    def test_generate_cutoffs_success_large_num_folds(self):
        cutoffs = generate_cutoffs(self.X, horizon=7, unit="D", num_folds=20, seasonal_period=1)
        self.assertEqual([pd.Timestamp('2020-07-22 12:00:00'),
                          pd.Timestamp('2020-07-26 00:00:00'),
                          pd.Timestamp('2020-07-29 12:00:00'),
                          pd.Timestamp('2020-08-02 00:00:00'),
                          pd.Timestamp('2020-08-05 12:00:00'),
                          pd.Timestamp('2020-08-09 00:00:00'),
                          pd.Timestamp('2020-08-12 12:00:00'),
                          pd.Timestamp('2020-08-16 00:00:00'),
                          pd.Timestamp('2020-08-19 12:00:00'),
                          pd.Timestamp('2020-08-23 00:00:00')], cutoffs)

    def test_generate_cutoffs_success_with_gaps(self):
        df = pd.DataFrame(
            pd.date_range(start="2020-07-01", periods=30, freq='3d'), columns=["ds"]
        ).rename_axis("y").reset_index()
        cutoffs = generate_cutoffs(df, horizon=1, unit="D", num_folds=5, seasonal_period=1)
        self.assertEqual([pd.Timestamp('2020-09-13 00:00:00'),
                          pd.Timestamp('2020-09-16 00:00:00'),
                          pd.Timestamp('2020-09-19 00:00:00'),
                          pd.Timestamp('2020-09-22 00:00:00'),
                          pd.Timestamp('2020-09-25 00:00:00')], cutoffs)

    def test_generate_cutoffs_success_hourly(self):
        df = pd.DataFrame(
            pd.date_range(start="2020-07-01", periods=168, freq='h'), columns=["ds"]
        ).rename_axis("y").reset_index()
        expected_cutoffs = [pd.Timestamp('2020-07-07 05:00:00'),
                            pd.Timestamp('2020-07-07 08:00:00'),
                            pd.Timestamp('2020-07-07 11:00:00'),
                            pd.Timestamp('2020-07-07 14:00:00'),
                            pd.Timestamp('2020-07-07 17:00:00')]
        cutoffs = generate_cutoffs(df, horizon=6, unit="H", num_folds=5, seasonal_period=24)
        self.assertEqual(expected_cutoffs, cutoffs)

        cutoffs_different_seasonal_unit = generate_cutoffs(df, horizon=6, unit="H", num_folds=5,
                                                           seasonal_period=1, seasonal_unit="D")
        self.assertEqual(expected_cutoffs, cutoffs_different_seasonal_unit)

    def test_generate_cutoffs_success_weekly(self):
        df = pd.DataFrame(
            pd.date_range(start="2020-07-01", periods=52, freq='W'), columns=["ds"]
        ).rename_axis("y").reset_index()
        cutoffs = generate_cutoffs(df, horizon=4, unit="W", num_folds=3, seasonal_period=1)
        self.assertEqual([pd.Timestamp('2021-05-02 00:00:00'), pd.Timestamp('2021-05-16 00:00:00'), pd.Timestamp('2021-05-30 00:00:00')], cutoffs)

    def test_generate_cutoffs_failure_horizon_too_large(self):
        with self.assertRaisesRegex(ValueError, "Less data than horizon after initial window. "
                                                "Make horizon shorter."):
            generate_cutoffs(self.X, horizon=20, unit="D", num_folds=3, seasonal_period=1)

    def test_generate_cutoffs_less_data(self):
        with self.assertRaisesRegex(ValueError, "Less data than horizon."):
            generate_cutoffs(self.X, horizon=100, unit="D", num_folds=3, seasonal_period=1)

    def test_generate_cutoffs_success_monthly(self):
        df = pd.DataFrame(
            pd.date_range(start="2020-01-12", periods=24, freq=pd.DateOffset(months=1)), columns=["ds"]
        ).rename_axis("y").reset_index()
        cutoffs = generate_cutoffs(df, horizon=2, unit="MS", num_folds=3, seasonal_period=1)
        self.assertEqual([pd.Timestamp('2021-08-12 00:00:00'), pd.Timestamp('2021-9-12 00:00:00'), pd.Timestamp('2021-10-12 00:00:00')], cutoffs)

    def test_generate_cutoffs_success_quaterly(self):
        df = pd.DataFrame(
            pd.date_range(start="2020-07-12", periods=9, freq=pd.DateOffset(months=3)), columns=["ds"]
        ).rename_axis("y").reset_index()
        cutoffs = generate_cutoffs(df, horizon=1, unit="QS", num_folds=3, seasonal_period=1)
        self.assertEqual([pd.Timestamp('2021-10-12 00:00:00'), pd.Timestamp('2022-01-12 00:00:00'), pd.Timestamp('2022-04-12 00:00:00')], cutoffs)

    def test_generate_cutoffs_success_annualy(self):
        df = pd.DataFrame(
            pd.date_range(start="2012-07-14", periods=10, freq=pd.DateOffset(years=1)), columns=["ds"]
        ).rename_axis("y").reset_index()
        cutoffs = generate_cutoffs(df, horizon=1, unit="YS", num_folds=3, seasonal_period=1)
        self.assertEqual([pd.Timestamp('2018-07-14 00:00:00'), pd.Timestamp('2019-07-14 00:00:00'), pd.Timestamp('2020-07-14 00:00:00')], cutoffs)


class TestCalculatePeriodsAndFrequency(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()
    
    def test_calculate_period_differences_evenly(self):
        df = pd.DataFrame({
            'start_time': pd.Series(
                ['2021-01-14', '2021-02-14', '2021-03-14']
            ),
            'end_time': pd.Series(
                ['2021-05-14', '2021-07-14', '2022-03-14']
            )
        })
        periods = df.apply(lambda x: calculate_period_differences(
            x.start_time, x.end_time, 'month'
        ), axis=1)
        self.assertTrue((periods == pd.Series([4, 5, 12])).all())
    
    def test_calculate_period_differences_unevenly(self):
        df = pd.DataFrame({
            'start_time': pd.Series(
                ['2021-01-14', '2021-02-14', '2021-03-14']
            ),
            'end_time': pd.Series(
                ['2021-05-12', '2021-07-15', '2021-03-14']
            )
        })
        periods = df.apply(lambda x: calculate_period_differences(
            x.start_time, x.end_time, 'month'
        ), axis=1)
        self.assertTrue((periods == pd.Series([4, 5, 0])).all())
        periods = df.apply(lambda x: calculate_period_differences(
            x.start_time, x.end_time, 'day'
        ), axis=1)
        self.assertTrue((periods == pd.Series([118, 151, 0])).all())

    def test_frequency_consistency(self):
        start_time = pd.Series(
            ['2021-01-14', '2021-02-14', '2021-03-14']
        )
        end_time = pd.Series(
            ['2021-05-14', '2021-07-14', '2022-03-14']
        )
        start_scalar = pd.to_datetime('2021-01-14')
        end_scalar = pd.to_datetime('2021-05-16')
        self.assertFalse(is_frequency_consistency(start_scalar, end_scalar, 'month'))
        self.assertTrue(start_time.apply(
            lambda x: is_frequency_consistency(x, end_scalar, 'day')
        ).all())
        self.assertTrue(end_time.apply(
            lambda x: is_frequency_consistency(start_scalar, x, 'month')
        ).all())


class TestMakeFutureDataFrame(unittest.TestCase):
    def test_make_single_future_dataframe(self):
        future_df = make_single_future_dataframe(
            start_time=pd.to_datetime('2022-01-01'),
            end_time=pd.to_datetime('2022-01-04'),
            horizon=1,
            frequency="d",
            include_history=False,
            column_name="test_date"
        )
        self.assertEqual(len(future_df), 1)
        expected_columns = { "test_date" }
        self.assertTrue(expected_columns.issubset(set(future_df.columns)))

        future_df = make_single_future_dataframe(
            start_time=pd.to_datetime('2022-01-01'),
            end_time=pd.to_datetime('2022-01-04'),
            horizon=1,
            frequency="d",
            include_history=True,
            column_name="test_date"
        )
        self.assertEqual(len(future_df), 5)
        expected_columns = { "test_date" }
        self.assertTrue(expected_columns.issubset(set(future_df.columns)))

    @parameterized.expand([
        (pd.to_datetime('2022-01-01'), pd.to_datetime('2022-01-05'), None, None, { "ds" }, ),
        (pd.to_datetime('2022-01-01'), pd.to_datetime('2022-01-05'), ["id"], [("1", )], { "ds", "id" }),
        (pd.to_datetime('2022-01-01'), pd.to_datetime('2022-01-05'), ["id1", "id2"],
         [("1", 1)], { "ds", "id1", "id2" }),
        ({("1", ): pd.to_datetime('2022-01-01'), ("2", ): pd.to_datetime('2022-01-01')},
         pd.to_datetime('2022-01-02'), ["id"], [("1", ), ("2", )], { "ds", "id" }),
        ({("1", ): pd.to_datetime('2022-01-01'), ("2", ): pd.to_datetime('2022-01-01')},
         {("1", ): pd.to_datetime('2022-01-02'), ("2", ): pd.to_datetime('2022-01-02')},
         ["id"], [("1", ), ("2", )], { "ds", "id" }),
    ])
    def test_make_future_dataframe(self, start_time, end_time,
                                   identity_column_names,
                                   groups, expected_columns):
        future_df = make_future_dataframe(
            start_time=start_time,
            end_time=end_time,
            horizon=1,
            frequency="d",
            groups=groups,
            identity_column_names=identity_column_names,
        )
        print(future_df)
        self.assertEqual(len(future_df), 6)
        self.assertTrue(expected_columns.issubset(set(future_df.columns)))
