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

import pandas as pd

from databricks.automl_runtime.forecast.pmdarima.utils import plot


class TestPlot(unittest.TestCase):

    def test_plot_success(self):
        num_rows, horizon = 20, 4
        yhat = [i + 0.5 for i in range(num_rows)]
        history_pd = pd.DataFrame({
            "ds":
            pd.date_range(start="2020-10-01",
                          periods=num_rows - horizon,
                          freq='d'),
            "y":
            range(num_rows - horizon)
        })
        forecast_pd = pd.DataFrame({
            "ds":
            pd.date_range(start="2020-10-01", periods=num_rows, freq='d'),
            "yhat":
            yhat,
            "yhat_lower": [i - 0.5 for i in yhat],
            "yhat_upper": [i + 1 for i in yhat]
        })
        plot(history_pd, forecast_pd)
