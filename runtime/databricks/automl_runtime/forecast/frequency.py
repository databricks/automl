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
from dataclasses import dataclass
from typing import ClassVar, Set

@dataclass(frozen=True)
class Frequency:
    """
    Represents the frequency of a time series.

    Attributes:
        frequency_unit (str): The unit of time for the frequency.
        frequency_quantity (int): The number of frequency_units in the period.
    
    Valid frequency units: source of truth is OFFSET_ALIAS_MAP in forecast.__init__.py
        - Weeks: "W", "W-SUN", "W-MON", "W-TUE", "W-WED", "W-THU", "W-FRI", "W-SAT" These are aliases for "W", used for DeepAR only
        - Days: "d", "D", "days", "day"
        - Hours: "hours", "hour", "hr", "h", "H
        - Minutes: "m", "minute", "min", "minutes", "T"
        - Seconds: "S", "seconds", "sec", "second"
        - Months: "M", "MS", "month", "months"
        - Quarters: "Q", "QS", "quarter", "quarters"
        - Years: "Y", "YS", "year", "years"

    Valid frequency quantities:
        - For minutes: {1, 5, 10, 15, 30}
        - For all other units: {1}
    """

    VALID_FREQUENCY_UNITS: ClassVar[Set[str]] = {
        "W", "W-SUN", "W-MON", "W-TUE", "W-WED", "W-THU", "W-FRI", "W-SAT",
        "d", "D", "days", "day", "hours", "hour", "hr", "h", "H",
        "m", "minute", "min", "minutes", "T", "S", "seconds",
        "sec", "second", "M", "MS", "month", "months", "Q", "QS", "quarter",
        "quarters", "Y", "YS", "year", "years"
    }
    
    VALID_MINUTE_QUANTITIES: ClassVar[Set[int]] = {1, 5, 10, 15, 30}
    DEFAULT_QUANTITY: ClassVar[int] = 1  # Default for non-minute units

    frequency_unit: str
    frequency_quantity: int

    def __str__(self):
        return f"{self.frequency_quantity}{self.frequency_unit}"

    def __post_init__(self):
        if self.frequency_unit not in self.VALID_FREQUENCY_UNITS:
            raise ValueError(f"Invalid frequency unit: {self.frequency_unit}")

        if self.frequency_unit in {"m", "minute", "min", "minutes", "T"}:
            if self.frequency_quantity not in self.VALID_MINUTE_QUANTITIES:
                raise ValueError(
                    f"Invalid frequency quantity {self.frequency_quantity} for minutes. "
                    f"Allowed values: {sorted(self.VALID_MINUTE_QUANTITIES)}"
                )
        else:
            if self.frequency_quantity != self.DEFAULT_QUANTITY:
                raise ValueError(
                    f"Invalid frequency quantity {self.frequency_quantity} for {self.frequency_unit}. "
                    "Only 1 is allowed for this unit."
                )

    def is_second(self) -> bool:
        return self.frequency_unit in {"S", "seconds", "sec", "second"}        

    def is_minute(self) -> bool:
        return self.frequency_unit in {"m", "minute", "min", "minutes", "T"}

    def is_hourly(self) -> bool:
        return self.frequency_unit in {"hours", "hour", "hr", "h", "H"}        
    
    def is_daily(self) -> bool:
        return self.frequency_unit in {"d", "D", "days", "day"}

    def is_weekly(self) -> bool:
        return self.frequency_unit in {"W", "W-SUN", "W-MON", "W-TUE", "W-WED", "W-THU", "W-FRI", "W-SAT"}
    
    def is_monthly(self) -> bool:
        return self.frequency_unit in {"M", "MS", "month", "months"}
    
    def is_quarterly(self) -> bool:
        return self.frequency_unit in {"Q", "QS", "quarter", "quarters"}
    
    def is_yearly(self) -> bool:
        return self.frequency_unit in {"Y", "YS", "year", "years"}
