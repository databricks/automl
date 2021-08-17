import pandas as pd

from databricks.automl_runtime.feature.sklearn.base_datetime_transformer import BaseDateTimeTransformer


class DateTransformer(BaseDateTimeTransformer):

    def fit(self, X, y=None):
        """
        Do nothing and return the estimator unchanged. This method is just there to implement
        the usual API and hence work in pipelines.
        """
        return self

    def transform(self, X):
        X.iloc[:, 0] = X.iloc[:, 0].apply(pd.to_datetime, errors="coerce")
        X = X.fillna(pd.Timestamp(self.epoch))  # Fill NaT with the Unix epoch

        return self._generate_features(X, include_timestamp=False)
