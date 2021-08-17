import unittest

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

from databricks.automl_runtime.feature.sklearn.timestamp_transformer import TimestampTransformer


class TestTimestampTransformer(unittest.TestCase):

    def setUp(self) -> None:
        num_rows = 4
        self.X = pd.concat([
            pd.Series(range(num_rows), name='int1'),
            pd.Series(range(num_rows), name='timestamp1').apply(
                lambda i: pd.Timestamp(2017, 2, 1, i + 5) if i < 3 else pd.Timestamp(pd.NaT)),
        ], axis=1)
        self.timestamp_expected = np.array([
            [1485925200, False, 0.9165622558699762, -0.39989202431974097,
             0.2424681428783799, 0.97015936819118, 0.5251794996758523,
             0.8509914765261879, 0.9659258262890683, 0.25881904510252074, 5, False, False],
            [1485928800, False, 0.9009688679024191, -0.43388373911755806,
             0.2506525322587205, 0.9680771188662043, 0.5257880785139989,
             0.8506155985476382, 1.0, 6.123233995736766e-17, 6, False, False],
            [1485932400, False, 0.8841153935046098,
             -0.46726862827306204, 0.25881904510252074, 0.9659258262890683,
             0.5263963883313831, 0.8502392853495278, 0.9659258262890683,
             -0.25881904510252063, 7, False, False],
            [0, False, 0.43388373911755823, -0.900968867902419,
             0.20129852008866006, 0.9795299412524945, 0.01716632975470737,
             0.9998526477050269, 0.0, 1.0, 0, True, True],
        ])
        self.transformer = TimestampTransformer()

    def test_transform(self):
        timestamp_transformed = self.transformer.transform(self.X[['timestamp1']])
        np.testing.assert_array_almost_equal(timestamp_transformed.to_numpy(), self.timestamp_expected, decimal=5,
                                             err_msg="Actual: {}\nExpected: {}\nEquality: {}".format(
                                                 timestamp_transformed,
                                                 self.timestamp_expected,
                                                 timestamp_transformed == self.timestamp_expected))

    def test_with_pipeline(self):
        pipeline = Pipeline([("ts_transformer", self.transformer)])
        timestamp_transformed = pipeline.fit_transform(self.X[['timestamp1']])
        np.testing.assert_array_almost_equal(timestamp_transformed.to_numpy(), self.timestamp_expected, decimal=5,
                                             err_msg="Actual: {}\nExpected: {}\nEquality: {}".format(
                                                 timestamp_transformed,
                                                 self.timestamp_expected,
                                                 timestamp_transformed == self.timestamp_expected))
