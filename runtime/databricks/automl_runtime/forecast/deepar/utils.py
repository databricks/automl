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

import pandas as pd


def set_index_and_fill_missing_time_steps(df: pd.DataFrame, time_col: str,
                                          frequency: str,
                                          id_cols: Optional[List[str]] = None):
    """
    Transform the input dataframe to an acceptable format for the GluonTS library.

    - Set the time column as the index
    - Impute missing time steps between the min and max time steps

    :param df: the input dataframe that contains time_col
    :param time_col: time column name
    :param frequency: the frequency of the time series
    :param id_cols: the column names of the identity columns for multi-series time series; None for single series
    :return: single-series - transformed dataframe;
             multi-series - dictionary of transformed dataframes, each key is the (concatenated) id of the time series
    """
    total_min, total_max = df[time_col].min(), df[time_col].max()
    new_index_full = pd.date_range(total_min, total_max, freq=frequency)

    if id_cols is not None:
        df_dict = {}
        for grouped_id, grouped_df in df.groupby(id_cols):
            if isinstance(grouped_id, tuple):
                ts_id = "-".join([str(x) for x in grouped_id])
            else:
                ts_id = str(grouped_id)
            df_dict[ts_id] = (grouped_df.set_index(time_col).sort_index()
                              .reindex(new_index_full).drop(id_cols, axis=1))

        return df_dict

    df = df.set_index(time_col).sort_index()

    # Fill in missing time steps between the min and max time steps
    return df.reindex(new_index_full)
