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

from __future__ import absolute_import, division, print_function
from prophet.diagnostics import single_cutoff_forecast

import logging
from tqdm.auto import tqdm
import concurrent.futures

import pandas as pd


logger = logging.getLogger('prophet')

def cross_validation(model, horizon, period=None, initial=None, parallel=None, cutoffs=None, disable_tqdm=False):
    """Cross-Validation for time series.

    This function is same as prophet's native function
    https://github.com/facebook/prophet/blob/main/python/prophet/diagnostics.py#L61
    except that we support user to pass in horizon as pandas.DateOffset
    object directly, this would support the monthly/quarterly/annually horizon
    forecast since pd.TimeDelta does not support those types.

    Computes forecasts from historical cutoff points, which user can input.
    If not provided, begins from (end - horizon) and works backwards, making
    cutoffs with a spacing of period until initial is reached.

    When period is equal to the time interval of the data, this is the
    technique described in https://robjhyndman.com/hyndsight/tscv/ .

    Parameters
    ----------
    model: Prophet class object. Fitted Prophet model.
    horizon: Can be a pd.DateOffset object or a string with pd.Timedelta
        compatible style, e.g., '5 days', '3 hours', '10 seconds'.
    period: string with pd.Timedelta compatible style. Simulated forecast will
        be done at every this period. If not provided, 0.5 * horizon is used.
    initial: string with pd.Timedelta compatible style. The first training
        period will include at least this much data. If not provided,
        3 * horizon is used.
    cutoffs: list of pd.Timestamp specifying cutoffs to be used during
        cross validation. If not provided, they are generated as described
        above.
    parallel : {None, 'processes', 'threads', 'dask', object}
    disable_tqdm: if True it disables the progress bar that would otherwise show up when parallel=None

        How to parallelize the forecast computation. By default no parallelism
        is used.

        * None : No parallelism.
        * 'processes' : Parallelize with concurrent.futures.ProcessPoolExectuor.
        * 'threads' : Parallelize with concurrent.futures.ThreadPoolExecutor.
            Note that some operations currently hold Python's Global Interpreter
            Lock, so parallelizing with threads may be slower than training
            sequentially.
        * 'dask': Parallelize with Dask.
           This requires that a dask.distributed Client be created.
        * object : Any instance with a `.map` method. This method will
          be called with :func:`single_cutoff_forecast` and a sequence of
          iterables where each element is the tuple of arguments to pass to
          :func:`single_cutoff_forecast`

          .. code-block::

             class MyBackend:
                 def map(self, func, *iterables):
                     results = [
                        func(*args)
                        for args in zip(*iterables)
                     ]
                     return results

    Returns
    -------
    A pd.DataFrame with the forecast, actual value and cutoff.
    """

    df = model.history.copy().reset_index(drop=True)
    if not isinstance(horizon, pd.DateOffset):
        horizon = pd.Timedelta(horizon)

    predict_columns = ['ds', 'yhat']
    if model.uncertainty_samples:
        predict_columns.extend(['yhat_lower', 'yhat_upper'])
        
    # Identify largest seasonality period
    period_max = 0.
    for s in model.seasonalities.values():
        period_max = max(period_max, s['period'])
    seasonality_dt = pd.Timedelta(str(period_max) + ' days')    

    # add validation of the cutoff to make sure that the min cutoff is strictly greater than the min date in the history
    if min(cutoffs) <= df['ds'].min(): 
        raise ValueError("Minimum cutoff value is not strictly greater than min date in history")
    # max value of cutoffs is <= (end date minus horizon)
    end_date_minus_horizon = df['ds'].max() - horizon 
    if max(cutoffs) > end_date_minus_horizon: 
        raise ValueError("Maximum cutoff value is greater than end date minus horizon, no value for cross-validation remaining")
    initial = cutoffs[0] - df['ds'].min()
        
    # Check if the initial window 
    # (that is, the amount of time between the start of the history and the first cutoff)
    # is less than the maximum seasonality period
    if initial < seasonality_dt:
            msg = 'Seasonality has period of {} days '.format(period_max)
            msg += 'which is larger than initial window. '
            msg += 'Consider increasing initial.'
            logger.warning(msg)

    if parallel:
        valid = {"threads", "processes", "dask"}

        if parallel == "threads":
            pool = concurrent.futures.ThreadPoolExecutor()
        elif parallel == "processes":
            pool = concurrent.futures.ProcessPoolExecutor()
        elif parallel == "dask":
            try:
                from dask.distributed import get_client
            except ImportError as e:
                raise ImportError("parallel='dask' requies the optional "
                                  "dependency dask.") from e
            pool = get_client()
            # delay df and model to avoid large objects in task graph.
            df, model = pool.scatter([df, model])
        elif hasattr(parallel, "map"):
            pool = parallel
        else:
            msg = ("'parallel' should be one of {} for an instance with a "
                   "'map' method".format(', '.join(valid)))
            raise ValueError(msg)

        iterables = ((df, model, cutoff, horizon, predict_columns)
                     for cutoff in cutoffs)
        iterables = zip(*iterables)

        logger.info("Applying in parallel with %s", pool)
        predicts = pool.map(single_cutoff_forecast, *iterables)
        if parallel == "dask":
            # convert Futures to DataFrames
            predicts = pool.gather(predicts)

    else:
        predicts = [
            single_cutoff_forecast(df, model, cutoff, horizon, predict_columns) 
            for cutoff in (tqdm(cutoffs) if not disable_tqdm else cutoffs)
        ]
    # Combine all predicted pd.DataFrame into one pd.DataFrame
    return pd.concat(predicts, axis=0).reset_index(drop=True)
