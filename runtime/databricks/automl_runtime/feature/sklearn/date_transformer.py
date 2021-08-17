import pandas as pd

from databricks.automl_runtime.feature.sklearn.base_datetime_transformer import BaseDateTimeTransformer


class DateTransformer(BaseDateTimeTransformer):

    def transform(self, X):
        X.iloc[:, 0] = X.iloc[:, 0].apply(pd.to_datetime, errors="coerce")
        X = X.fillna(pd.Timestamp(self.EPOCH))  # Fill NaT with the Unix epoch

        return self._generate_features(X, include_timestamp=False)
