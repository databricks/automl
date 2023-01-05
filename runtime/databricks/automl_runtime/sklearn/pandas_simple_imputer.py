#
# Copyright (C) 2023 Databricks, Inc.
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
import pandas as pd
from sklearn.impute import SimpleImputer

class PandasSimpleImputer(SimpleImputer):
    """
    A wrapper of `SimpleImputer` with the support for pandas dataframe output.
    """

    def fit(self, X, y=None):
        """Fits the imputer on X

        Parameters
        ----------

        X : pd.DataFrame of shape = [n_samples, n_features]
            Training dataframe, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        """
        self.columns = X.columns
        return super().fit(X, y)

    def transform(self, X):
        """Impute all missing values in `X`.

        Parameters
        ----------

        X : pd.DataFrame of shape = [n_samples, n_features]
            Training dataframe, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------

        X_tr : pd.DataFrame of shape (n_samples, n_features_encoded)
            Transformed features.
        """
        return pd.DataFrame(super().transform(X), columns=self.columns)
