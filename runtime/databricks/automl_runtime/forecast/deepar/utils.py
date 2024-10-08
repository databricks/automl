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
    # TODO (ML-46009): Compare with the ARIMA implementation, and fill
    #                  the missing time steps for multi-series time series too
    if id_cols is not None:
        df = df.set_index(time_col)
        return df

    df = df.set_index(time_col).sort_index()
    new_index_full = pd.date_range(df.index.min(), df.index.max(), freq=frequency)

    # Fill in missing time steps between the min and max time steps
    return df.reindex(new_index_full)