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

from sklearn.base import ClassifierMixin, BaseEstimator, clone
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import FunctionTransformer


class TransformedTargetClassifier(ClassifierMixin, BaseEstimator):
    """Meta-estimator to classify on a transformed target.

    Useful when the underlying classifier only accepts specified format of targets.
    For example, xgboost requires the target values to be in {0, 1, ..., num_class -1}
    since 1.6.0.

    The implementation is similar to sklearn.compose.TransformedTargetRegressor.

    Parameters
    ----------
    classifier : object
        Classifier object such as derived from sklearn.base.ClassifierMixin.
        This classifier will automatically be cloned each time prior to fitting.
    transformer : object, default=None
        Estimator object that is designed to transform targets/labels, such as
        transformers defined in sklearn.preprocessing._label.
        If transformer is None, the transformer will be an identity
        transformer. Note that the transformer will be cloned during fitting.
        Notice that the transformer should work for the specific `y` to be
        passed in `fit`, otherwise the `fit` call can fail.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.preprocessing import LabelEncoder
    >>> from databricks.automl_runtime.sklearn import TransformedTargetClassifier
    >>> tt = TransformedTargetClassifier(classifier=LogisticRegression(), transformer=LabelEncoder())
    >>> X = np.arange(4).reshape(-1, 1)
    >>> y = ["A", "C", "B", "C"]
    >>> tt.fit(X, y)
    """

    def __init__(self, classifier, *, transformer=None):
        self.classifier_ = clone(classifier)
        self.transformer_ = FunctionTransformer() if transformer is None else clone(transformer)

    @property
    def classifier(self):
        return self.classifier_

    @property
    def transformer(self):
        return self.transformer_

    def fit(self, X, y, **fit_params):
        """Transform the target values and then fit the model with given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Target values.
        **fit_params : dict
            Parameters passed to the `fit` method of the underlying
            classifier.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        y_trans = self.transformer_.fit_transform(y)

        self.classifier_.fit(X, y_trans, **fit_params)
        classes_in_classifier = self.classifier_.classes_
        self.classes_ = self.transformer_.inverse_transform(classes_in_classifier)
        return self

    def predict(self, X, **predict_params):
        """Predict using the base classifier, applying inverse transform to target values.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Samples.

        **predict_params : dict of str -> object
            Parameters passed to the `predict` method of the underlying
            classifier.

        Returns
        -------
        y_hat : ndarray of shape (n_samples,)
            Predicted values.
        """
        check_is_fitted(self)

        pred = self.classifier_.predict(X, **predict_params)
        pred_trans = self.transformer_.inverse_transform(pred)

        return pred_trans

    def predict_proba(self, X, **predict_params):
        """Predict class probabilities using the base classifier.

        If the base classifier does not implement a `predict_prob` method,
        throw `NotImplementedError`.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Samples.

        **predict_params : dict of str -> object
            Parameters passed to the `predict_proba` method of the underlying
            classifier.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute classes_.
        """
        if not hasattr(self.classifier_, "predict_proba"):
            raise NotImplementedError(
                f"`predict_proba` is not implemented in {self.classifier_.__class__.__name__}"
            )
        return self.classifier_.predict_proba(X, **predict_params)

    def decision_function(self, X, **additional_params):
        """Compute the decision function of X using the base classifier.

        If the base classifier does not implement a `decision_function` method,
        throw `NotImplementedError`.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Samples.

        **additional_params : dict of str -> object
            Parameters passed to the `decision_function` method of the underlying
            classifier.

        Returns
        -------
        score: The decision function of the input samples, whose shape is determined
            by the `decision_function` of the underlying classifier.
        """
        if not hasattr(self.classifier_, "decision_function"):
            raise NotImplementedError(
                f"`decision_function` is not implemented in {self.classifier_.__class__.__name__}"
            )
        return self.classifier_.decision_function(X, **additional_params)
