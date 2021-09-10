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
import wrapt


def fail_safe_with_default(default_result):
    """
    Decorator to ensure that individual failures don't fail training
    """
    @wrapt.decorator
    def fail_safe(func, self, args, kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Encountered an exception: {repr(e)}")
            return default_result
    return fail_safe
