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

import numpy as np
import pandas as pd
import unittest

from pandas.testing import assert_frame_equal
from sklearn.pipeline import Pipeline


from databricks.automl_runtime.sklearn import InterpolationImputer

class TestInterpolationImputer(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame([(0.0, np.nan, -1.0, 1.0),
                                (np.nan, 2.0, np.nan, np.nan),
                                (2.0, 3.0, np.nan, 9.0),
                                (np.nan, 4.0, -4.0, 16.0)],
                               columns=list('abcd'))
        self.expected_df = pd.DataFrame([(0.0, 2.0, -1.0, 1.0),
                                         (1.0, 2.0, -2.0, 5.0),
                                         (2.0, 3.0, -3.0, 9.0),
                                         (2.0, 4.0, -4.0, 16.0)],
                                        columns=list('abcd'))
    def test_imputer(self):
        assert_frame_equal(
            InterpolationImputer().transform(self.df), self.expected_df)

    def test_imputer_with_custom_method(self):
        expected_df = pd.DataFrame([(0.0, 2.0, -1.0, 1.0),
                                    (0.0, 2.0, -1.0, 1.0),
                                    (2.0, 3.0, -1.0, 9.0),
                                    (2.0, 4.0, -4.0, 16.0)],
                                   columns=list('abcd'))
        interpolate_param = {"method": 'pad', "limit": 2}
        imputed_df = InterpolationImputer(impute_params=interpolate_param).transform(self.df)
        assert_frame_equal(imputed_df, expected_df)

    def test_pipeline(self):
        imputed_df = Pipeline([("imputer", InterpolationImputer())]).fit_transform(self.df)
        assert_frame_equal(imputed_df, self.expected_df)
