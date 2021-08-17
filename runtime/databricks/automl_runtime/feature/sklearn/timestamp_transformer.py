import pandas as pd

from databricks.automl_runtime.feature.sklearn.base_datetime_transformer import BaseDateTimeTransformer


class TimestampTransformer(BaseDateTimeTransformer):
    columns_to_ohe = [10]  # index of hour column

    def transform(self, X):
        X = X.fillna(pd.Timestamp(self.EPOCH))  # Fill NaT with the Unix epoch

        return self._generate_features(X, include_timestamp=True)
