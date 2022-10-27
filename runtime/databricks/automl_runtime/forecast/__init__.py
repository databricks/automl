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

# Offset Alias reference: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
OFFSET_ALIAS_MAP = {
    "W": "W",
    "d": "D",
    "D": "D",
    "days": "D",
    "day": "D",
    "hours": "H",
    "hour": "H",
    "hr": "H",
    "h": "H",
    "H": "H",
    "m": "min",
    "minute": "min",
    "min": "min",
    "minutes": "min",
    "T": "min",
    "S": "S",
    "seconds": "S",
    "sec": "S",
    "second": "S",
    'M': 'MS',
    'MS': 'MS',
    'months': 'MS',
    'month': 'MS',
    'Q': 'QS',
    'QS': 'QS',
<<<<<<< HEAD
    'quarters': 'QS',
    'quarter': 'QS',
=======
    'quaters': 'QS',
    'quater': 'QS',
>>>>>>> 3a0460e (Use DateOffset in foreast utils to support calculation based on months quaters and years.)
    'Y': 'YS',
    'YS': 'YS',
    'years': 'YS',
    'year': 'YS',
}

<<<<<<< HEAD
# Reference: https://pandas.pydata.org/docs/reference/api/pandas.tseries.offsets.DateOffset.html
=======
>>>>>>> 3a0460e (Use DateOffset in foreast utils to support calculation based on months quaters and years.)
DATE_OFFSET_KEYWORD_MAP = {
    'YS': 'years',
    'QS': 'months',
    'MS': 'months',
    'W': 'weeks',
    'D': 'days',
    'H': 'hours',
    'min': 'minutes',
    'S': 'seconds',
}

QUATERLY_OFFSET_ALIAS = [
    'Q', 'QS', 'BQ', 'BQS'
]
