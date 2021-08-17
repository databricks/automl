import pandas as pd

from databricks.automl_runtime.feature.sklearn.base_datetime_transformer import BaseDateTimeTransformer


class TimestampTransformer(BaseDateTimeTransformer):
    columns_to_ohe = [10]  # index of hour column

    def fit(self, X, y=None):
        """
        Do nothing and return the estimator unchanged. This method is just there to implement
        the usual API and hence work in pipelines.
        """
        return self

    def transform(self, X):
        X = X.fillna(pd.Timestamp(self.epoch))  # Fill NaT with the Unix epoch

        return self._generate_features(X, include_timestamp=True)
