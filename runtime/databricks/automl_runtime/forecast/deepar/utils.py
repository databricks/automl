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


def validate_and_generate_index(df: pd.DataFrame, 
                                time_col: str, 
                                frequency_unit: str, 
                                frequency_quantity: int):
    """
    Generate a complete time index for the given DataFrame based on the specified frequency.
    - Ensures the time column is in datetime format.
    - Validates consistency in the day of the month if frequency is "MS" (month start).
    - Generates a new time index from the minimum to the maximum timestamp in the data.
    :param df: The input DataFrame containing the time column.
    :param time_col: The name of the time column.
    :param frequency_unit: The frequency unit of the time series.
    :param frequency_quantity: The frequency quantity of the time series.
    :return: A complete time index covering the full range of the dataset.
    :raises ValueError: If the day-of-month pattern is inconsistent for "MS" frequency.
    """
    if frequency_unit.upper() != "MS":
        return pd.date_range(df[time_col].min(), df[time_col].max(), freq=f"{frequency_quantity}{frequency_unit}")

    df[time_col] = pd.to_datetime(df[time_col])  # Ensure datetime format

    # Extract unique days
    unique_days = df[time_col].dt.day.unique()

    if len(unique_days) == 1:
        # All dates have the same day-of-month, considered consistent
        day_of_month = unique_days[0]
    else:
        # Check if all dates are last days of their respective months
        is_last_day = (df[time_col] + pd.offsets.MonthEnd(0)) == df[time_col]
        if is_last_day.all():
            day_of_month = "MonthEnd"
        else:
            raise ValueError("Inconsistent day of the month found in time column.")

    # Generate new index based on detected pattern
    total_min, total_max = df[time_col].min(), df[time_col].max()
    month_starts = pd.date_range(start=total_min.to_period("M").to_timestamp(),
                                 end=total_max.to_period("M").to_timestamp(),
                                 freq="MS")

    if day_of_month == "MonthEnd":
        new_index_full = month_starts + pd.offsets.MonthEnd(0)
    else:
        new_index_full = month_starts.map(lambda d: d.replace(day=day_of_month))

    return new_index_full

def set_index_and_fill_missing_time_steps(
    df: pd.DataFrame,
    time_col: str,
    frequency_unit: str,
    frequency_quantity: int,
    id_cols: Optional[List[str]] = None,
    feature_cols: Optional[List[str]] = None
):
    """
    Transform the input dataframe to an acceptable format for the GluonTS library.

    - Set the time column as the index
    - Impute missing time steps between the min and max time steps
    - Forward/backward fill covariates if provided

    :param df: the input dataframe that contains time_col
    :param time_col: time column name
    :param frequency_unit: the frequency unit of the time series
    :param frequency_quantity: the frequency quantity of the time series
    :param id_cols: the column names of the identity columns for multi-series time series; None for single series
    :param feature_cols: covariate columns to preserve (optional)
    :return: single-series - transformed dataframe;
             multi-series - dictionary of transformed dataframes, each key is the (concatenated) id of the time series
    """
    total_min, total_max = df[time_col].min(), df[time_col].max()
    # We need to adjust the frequency_unit for pd.date_range if it is weekly,
    # otherwise it would always be "W-SUN"
    if frequency_unit.upper() == "W":
        weekday_name = total_min.strftime("%a").upper() # e.g., "FRI"
        frequency_unit = f"W-{weekday_name}"

    valid_index = validate_and_generate_index(df=df, time_col=time_col, frequency_unit=frequency_unit, frequency_quantity=frequency_quantity)

    cols_to_keep = [c for c in df.columns if c not in id_cols] if id_cols else df.columns.tolist()
    if feature_cols:
        # ensure feature_cols are included in the reindexing
        cols_to_keep = list(set(cols_to_keep) | set(feature_cols))

    if id_cols is not None:
        df_dict = {}
        for grouped_id, grouped_df in df.groupby(id_cols):
            if isinstance(grouped_id, tuple):
                ts_id = "-".join([str(x) for x in grouped_id])
            else:
                ts_id = str(grouped_id)

            temp_df = grouped_df[cols_to_keep].set_index(time_col).sort_index()
            temp_df = temp_df.reindex(valid_index)  # add missing timestamps

            if feature_cols:
                # Only forward/backward fill covariates.
                # Do not fill the target column: DeepAR handles missing targets differently.
                # This ensures covariate arrays have no NaNs and align with the full timestamp index.
                covars = [c for c in feature_cols if c in temp_df.columns]
                temp_df[covars] = temp_df[covars].ffill().bfill()

            # Apply MS conversion if needed
            if frequency_unit.upper() == "MS":
                # Truncate the day of month to avoid issues with pandas frequency check
                temp_df = temp_df.to_period("M")

            df_dict[ts_id] = temp_df

        return df_dict
    else:
        df = df[cols_to_keep].set_index(time_col).sort_index()
        df = df.reindex(valid_index)

        if feature_cols:
            # Only forward/backward fill covariates.
            # Do not fill the target column: DeepAR handles missing targets differently.
            # This ensures covariate arrays have no NaNs and align with the full timestamp index.
            covars = [c for c in feature_cols if c in df.columns]
            df[covars] = df[covars].ffill().bfill()

        if frequency_unit.upper() == "MS":
            # Truncate the day of month to avoid issues with pandas frequency check
            df = df.to_period("M")

        return df
