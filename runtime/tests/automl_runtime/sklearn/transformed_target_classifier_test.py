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
import unittest
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

from databricks.automl_runtime.sklearn import TransformedTargetClassifier


class TestTransformedTargetClassifier(unittest.TestCase):
    def setUp(self):
        iris = datasets.load_iris(as_frame=True)
        self.X = iris.data
        self.y = iris.target.apply(lambda x: iris.target_names[x])

    def test_fit(self):
        model = TransformedTargetClassifier(classifier=LogisticRegression(), transformer=LabelEncoder())
        model.fit(self.X, self.y)
        y_trans = model.transformer_.transform(self.y)
        y_trans_inversed = model.transformer_.inverse_transform(y_trans)

        le = LabelEncoder()
        y_encoded = le.fit_transform(self.y)
        np.testing.assert_array_almost_equal(y_encoded, y_trans)
        self.assertTrue((self.y == y_trans_inversed).all())

    def test_predict(self):
        model = TransformedTargetClassifier(classifier=LogisticRegression(), transformer=LabelEncoder())
        model.fit(self.X, self.y)
        y_pred = model.predict(self.X)

        lr = LogisticRegression()
        le = LabelEncoder()
        y_encoded = le.fit_transform(self.y)
        lr.fit(self.X, y_encoded)
        y_pred_lr = lr.predict(self.X)
        self.assertTrue((y_pred == le.inverse_transform(y_pred_lr)).all())

    def test_identity_transformation_when_no_given_transformer(self):
        model = TransformedTargetClassifier(classifier=LogisticRegression())
        model.fit(self.X, self.y)
        y_trans = model.transformer_.transform(self.y)
        self.assertTrue((self.y == y_trans).all())

    def test_predict_prob(self):
        model = TransformedTargetClassifier(classifier=LogisticRegression(), transformer=LabelEncoder())
        model.fit(self.X, self.y)
        proba = model.predict_proba(self.X)

        lr = LogisticRegression()
        le = LabelEncoder()
        y_encoded = le.fit_transform(self.y)
        lr.fit(self.X, y_encoded)
        proba_lr = lr.predict_proba(self.X)
        np.testing.assert_array_almost_equal(proba, proba_lr)

    def test_predict_prob_not_implemented_error(self):
        model = TransformedTargetClassifier(classifier=LinearSVC(), transformer=LabelEncoder())
        model.fit(self.X, self.y)
        with self.assertRaises(NotImplementedError):
            model.predict_proba(self.X)

    def test_decision_function(self):
        model = TransformedTargetClassifier(classifier=LogisticRegression(), transformer=LabelEncoder())
        model.fit(self.X, self.y)
        decision_function = model.decision_function(self.X)

        lr = LogisticRegression()
        le = LabelEncoder()
        y_encoded = le.fit_transform(self.y)
        lr.fit(self.X, y_encoded)
        decision_function_lr = lr.decision_function(self.X)
        np.testing.assert_array_almost_equal(decision_function, decision_function_lr)

    def test_decision_function_not_implemented_error(self):
        model = TransformedTargetClassifier(classifier=DecisionTreeClassifier(), transformer=LabelEncoder())
        model.fit(self.X, self.y)
        with self.assertRaises(NotImplementedError):
            model.decision_function(self.X)
