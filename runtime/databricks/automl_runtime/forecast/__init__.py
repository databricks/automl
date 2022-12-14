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
    'quarters': 'QS',
    'quarter': 'QS',
    'Y': 'YS',
    'YS': 'YS',
    'years': 'YS',
    'year': 'YS',
}

# Reference: https://pandas.pydata.org/docs/reference/api/pandas.tseries.offsets.DateOffset.html
DATE_OFFSET_KEYWORD_MAP = {
    'YS': {
        'years': 1
    },
    'QS': {
        'months': 3
    },
    'MS': {
        'months': 1
    },
    'W': {
        'weeks': 1
    },
    'D': {
        'days': 1
    },
    'H': {
        'hours': 1
    },
    'min': {
        'minutes': 1
    },
    'S': {
        'seconds': 1
    }
}

QUATERLY_OFFSET_ALIAS = [
    'Q', 'QS', 'BQ', 'BQS'
]

NON_DAILY_OFFSET_ALIAS = [
    'M', 'MS', 'Q', 'QS', 'Y', 'YS'
]

# 
PERIOD_ALIAS_MAP = {
    "W": "W",
    "D": "D",
    "H": "H",
    "min": "min",
    "S": "S",
    'MS': 'M',
    'QS': 'Q',
    'YS': 'Y',
}