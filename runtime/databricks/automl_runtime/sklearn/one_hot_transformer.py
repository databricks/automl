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
from sklearn.base import TransformerMixin, BaseEstimator

from scipy import sparse
from category_encoders import one_hot

class OneHotEncoder(TransformerMixin, BaseEstimator):
    """
    A wrapper around the category_encoder's `OneHotEncoder` with additional support for sparse output.
    """

    def __init__(self, sparse=True, **kwargs):
        """Creates a wrapper around category_encoder's `OneHotEncoder`

        Parameters
        ----------
        Same parameters as category_encoder's OneHotEncoder, but with sparse as a new addition

        sparse: boolean describing whether you want a sparse outut or not
        """
        self.sparse = sparse
        self.base_one_hot_encoder = one_hot.OneHotEncoder(**kwargs)

    def fit_transform(self, X, y=None, **fit_params):
        """Fits the encoder according to X and y (and additional fit_params) and then transforms X

        Parameters
        ----------

        X : pd.DataFrame of shape = [n_samples, n_features]
            Training dataframe, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------

        X_tr : pd.DataFrame of shape (n_samples, n_features_encoded)
            Transformed features.
        """
        self.base_one_hot_encoder.fit(X, y, **fit_params)
        X_updated = self.base_one_hot_encoder.transform(X)
        if self.sparse:
            X_updated = sparse.csr_matrix(X_updated)
        return X_updated

    def fit(self, X, y=None, **kwargs):
        """Fits the encoder according to X and y (and additional fit_params)

        Parameters
        ----------

        X : pd.DataFrame of shape = [n_samples, n_features]
            Training dataframe, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        """
        self.base_one_hot_encoder.fit(X, y, **kwargs)

    def transform(self, X):
        """Transforms the input by adding additional columns for OneHotEncoding

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
        X_updated = self.base_one_hot_encoder.transform(X)
        if self.sparse:
            X_updated = sparse.csr_matrix(X_updated)
        return X_updated
