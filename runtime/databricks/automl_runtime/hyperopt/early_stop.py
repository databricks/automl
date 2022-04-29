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

import logging

from hyperopt import STATUS_OK

_logger = logging.getLogger(__name__)
_logger_hyperopt = logging.getLogger("hyperopt-spark")
_logger_hyperopt.setLevel(logging.ERROR)


def get_early_stop_fn(no_early_stop_threshold=40, no_progress_stop_threshold=20):
    """
    Returns an `early_stop_fn` used by hyperopt.

    :param no_early_stop_threshold: minimum number of trials to run before
        early stopping is considered
    :param no_progress_stop_threshold: hyperopt search will stop early if the
        loss doesn't improve after this number of iterations
    """

    def early_stop_fn(trials, best_loss=None, no_progress_iters=0, completed_trial_ids=set()):
        """
        The function checks if hyperopt should stop searching given results from the runs. 
        See hyperopt documentation for more details on the API of this function:
        https://github.com/hyperopt/hyperopt/blob/master/hyperopt/fmin.py#L487

        :param trials: SparkTrials
        :param best_loss: best validation loss so far
        :param no_progress_iters: number of trials/iterations where the loss has not improved
            compared to best_loss
        :param completed_trial_ids: set of trial ids for all trials that finished running.
            This is used to determine which trials are newly completed.
        :return: (stop, [best_loss, no_progress_iters, completed_trial_ids]),
            where stop (bool) indicates whether hyperopt should early stop, and the other three
            parameters are updated inputs to the next call of early_stop_fn
        """
        new_completed_trial_ids = set()
        new_best_loss = float("inf")
        for trial in trials:
            if trial["result"]["status"] == STATUS_OK and trial["tid"] not in completed_trial_ids:
                new_completed_trial_ids.add(trial["tid"])
                new_loss = trial["result"]["loss"]
                if new_loss < new_best_loss:
                    new_best_loss = new_loss

        completed_trial_ids = completed_trial_ids.union(new_completed_trial_ids)
        if best_loss is None:
            return False, [new_best_loss, 0, completed_trial_ids]
        if new_best_loss < best_loss:
            best_loss = new_best_loss
            no_progress_iters = 0
        else:
            no_progress_iters += len(new_completed_trial_ids)
            _logger.info(
                f"No hyperparameter tuning progress made for {no_progress_iters} iterations."
                f"Will early stop after {no_progress_stop_threshold} iterations."
                f"best_loss={best_loss}, new_best_loss={new_best_loss}"
            )

        return (
            no_progress_iters >= no_progress_stop_threshold and len(completed_trial_ids) >= no_early_stop_threshold,
            [best_loss, no_progress_iters, completed_trial_ids],
        )

    return early_stop_fn
