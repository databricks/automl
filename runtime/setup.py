#!/usr/bin/env python

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

from os import path
from setuptools import setup
import sys

DESCRIPTION = "Databricks AutoML Runtime Package"
# Read the contents of README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

# Read the requirements
with open(path.join(this_directory, "requirements.txt"), encoding="utf-8") as f:
    requirements = f.readlines()

# Load the version from `version.py` in the package
try:
    exec(open("databricks/automl_runtime/version.py").read())
except IOError:
    print("Failed to load AutoML runtime package version file for packaging.",
          file=sys.stderr)
    sys.exit(-1)
VERSION = __version__  # noqa

setup(
    name="databricks-automl-runtime",
    version=VERSION,
    author="Databricks",
    packages=[
        "databricks",
        "databricks.automl_runtime",
        "databricks.automl_runtime.forecast",
        "databricks.automl_runtime.forecast.pmdarima",
        "databricks.automl_runtime.forecast.prophet",
        "databricks.automl_runtime.hyperopt",
        "databricks.automl_runtime.sklearn"],
    license="http://www.apache.org/licenses/LICENSE-2.0",
    url="https://github.com/databricks/automl",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    options={"bdist_wheel": {"universal": True}},
)
