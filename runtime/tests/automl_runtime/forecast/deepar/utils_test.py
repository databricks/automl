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

import pandas as pd

from databricks.automl_runtime.forecast.deepar.utils import set_index_and_fill_missing_time_steps


class TestDeepARUtils(unittest.TestCase):
    def test_single_series_filled(self):
        target_col = "sales"
        time_col = "date"
        num_rows = 10

        base_df = pd.concat(
            [
                pd.to_datetime(
                    pd.Series(range(num_rows), name=time_col).apply(
                        lambda i: f"2020-10-{i + 1}"
                    )
                ),
                pd.Series(range(num_rows), name=target_col),
            ],
            axis=1,
        )
        dropped_df = base_df.drop([4, 5]).reset_index(drop=True)

        transformed_df = set_index_and_fill_missing_time_steps(dropped_df, time_col, "D")

        expected_df = base_df.copy()
        expected_df.loc[[4, 5], target_col] = float('nan')
        expected_df = expected_df.set_index(time_col).rename_axis(None).asfreq("D")

        pd.testing.assert_frame_equal(transformed_df, expected_df)

    def test_multi_series_filled(self):
        target_col = "sales"
        time_col = "date"
        id_col = "store"

        num_rows_per_ts = 10
        base_df = pd.concat(
            [
                pd.to_datetime(
                    pd.Series(range(num_rows_per_ts), name=time_col).apply(
                        lambda i: f"2020-10-{i + 1}"
                    )
                ),
                pd.Series(range(num_rows_per_ts), name=target_col),
            ],
            axis=1,
        )
        dropped_base_df = base_df.drop([4, 5]).reset_index(drop=True)
        dropped_df = pd.concat([dropped_base_df.copy(), dropped_base_df.copy()], ignore_index=True)
        dropped_df[id_col] = [1] * (num_rows_per_ts - 2) + [2] * (num_rows_per_ts - 2)

        transformed_df_dict = set_index_and_fill_missing_time_steps(dropped_df, time_col, "D", id_cols=[id_col])
        self.assertEqual(transformed_df_dict.keys(), {"1", "2"})

        expected_first_df = base_df.copy()
        expected_first_df.loc[[4, 5], target_col] = float('nan')
        expected_first_df = expected_first_df.set_index(time_col).rename_axis(None).asfreq("D")

        pd.testing.assert_frame_equal(transformed_df_dict["1"], expected_first_df)
