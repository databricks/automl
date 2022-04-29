#
# Copyright (C) 2021 Databricks, Inc.
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

import unittest
from unittest import mock

from hyperopt import STATUS_OK

from databricks.automl_runtime.hyperopt.early_stop import get_early_stop_fn


early_stop_fn = get_early_stop_fn(no_early_stop_threshold=40, no_progress_stop_threshold=20)


def mock_trial(tid, ok=False, loss=None):
    mock_trial = {"tid": tid, "result": {}}
    if ok:
        mock_trial["result"]["status"] = STATUS_OK
    else:
        mock_trial["result"]["status"] = "not_ok"

    if loss:
        mock_trial["result"]["loss"] = loss

    return mock_trial


class TestEarlyStopping(unittest.TestCase):
    # Used by unittest.TestCase, removes the limit on the size of the objects our assertions compare
    maxDiff = None

    def test_early_stop_no_early_stop_threshold(self):
        # don't early stop if NO_EARLY_STOP_THRESHOLD has not been reached
        mock_trials = mock.MagicMock()
        stop, _ = early_stop_fn(mock_trials, best_loss=100, no_progress_iters=50, completed_trial_ids=set(range(39)))
        self.assertFalse(stop)

    def test_early_stop_no_progress_stop_threshold(self):
        # don't early stop if NO_PROGRESS_STOP_THRESHOLD has not been reached
        mock_trials = mock.MagicMock()
        stop, _ = early_stop_fn(mock_trials, best_loss=100, no_progress_iters=19, completed_trial_ids=set(range(100)))
        self.assertFalse(stop)

    def test_early_stop(self):
        # early stop if both conditions reached
        mock_trials = mock.MagicMock()
        stop, _ = early_stop_fn(mock_trials, best_loss=100, no_progress_iters=20, completed_trial_ids=set(range(40)))
        self.assertTrue(stop)
        stop, _ = early_stop_fn(mock_trials, best_loss=100, no_progress_iters=200, completed_trial_ids=set(range(400)))
        self.assertTrue(stop)

        mock_trials.__iter__.return_value = [mock_trial(39, ok=True, loss=100)]
        stop, _ = early_stop_fn(mock_trials, best_loss=100, no_progress_iters=19, completed_trial_ids=set(range(39)))
        self.assertTrue(stop)

    def test_early_stop_one_trial(self):
        # new_args are set from initial state, after one completed trial
        mock_trials = mock.MagicMock()
        mock_trials.__iter__.return_value = [
            mock_trial(0, ok=True, loss=100),
            mock_trial(1),
        ]
        stop, new_args = early_stop_fn(mock_trials)
        best_loss, no_progress_iters, completed_trial_ids = new_args
        self.assertFalse(stop)
        self.assertEqual(best_loss, 100)
        self.assertEqual(no_progress_iters, 0)
        self.assertEqual(completed_trial_ids, set(range(1)))

    def test_early_stop_multiple_trials(self):
        # new_args are set from initial state, after multiple completed trials
        mock_trials = mock.MagicMock()
        mock_trials.__iter__.return_value = [
            mock_trial(0, ok=True, loss=100),
            mock_trial(1, ok=True, loss=50),
            mock_trial(2),
            mock_trial(3),
        ]
        stop, new_args = early_stop_fn(mock_trials)
        best_loss, no_progress_iters, completed_trial_ids = new_args
        self.assertFalse(stop)
        self.assertEqual(best_loss, 50)
        self.assertEqual(no_progress_iters, 0)
        self.assertEqual(completed_trial_ids, set(range(2)))

        # use output of previous call as arguments to early_stop_fn
        mock_trials = mock.MagicMock()
        mock_trials.__iter__.return_value = [
            mock_trial(0, ok=True, loss=100),
            mock_trial(1, ok=True, loss=50),
            mock_trial(2, ok=True, loss=25),
            mock_trial(3),
        ]
        stop, new_args = early_stop_fn(mock_trials, best_loss, no_progress_iters, completed_trial_ids)
        best_loss, no_progress_iters, completed_trial_ids = new_args
        self.assertFalse(stop)
        self.assertEqual(best_loss, 25)
        self.assertEqual(no_progress_iters, 0)
        self.assertEqual(completed_trial_ids, set(range(3)))

    def test_early_stop_no_progress_iters(self):
        mock_trials = mock.MagicMock()
        mock_trials.__iter__.return_value = [
            mock_trial(0, ok=True, loss=100),
            mock_trial(1, ok=True, loss=25),
            mock_trial(2),
            mock_trial(3),
            mock_trial(4),
        ]
        stop, new_args = early_stop_fn(mock_trials)
        best_loss, no_progress_iters, completed_trial_ids = new_args
        self.assertFalse(stop)
        self.assertEqual(best_loss, 25)
        self.assertEqual(no_progress_iters, 0)
        self.assertEqual(completed_trial_ids, set(range(2)))

        # no_progress_iters is updated
        mock_trials = mock.MagicMock()
        mock_trials.__iter__.return_value = [
            mock_trial(0, ok=True, loss=100),
            mock_trial(1, ok=True, loss=25),
            mock_trial(2, ok=True, loss=25),
            mock_trial(3, ok=True, loss=25),
            mock_trial(4, ok=True, loss=25),
        ]
        stop, new_args = early_stop_fn(mock_trials, best_loss, no_progress_iters, completed_trial_ids)
        best_loss, no_progress_iters, completed_trial_ids = new_args
        self.assertFalse(stop)
        self.assertEqual(best_loss, 25)
        self.assertEqual(no_progress_iters, 3)
        self.assertEqual(completed_trial_ids, set(range(5)))
