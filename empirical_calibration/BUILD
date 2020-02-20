# Copyright 2019 The Empirical Calibration Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(["LICENSE"])

pytype_library(
    name = "core",
    srcs = [
        "__init__.py",
        "core.py",
    ],
    srcs_version = "PY3",
    deps = [
        # absl:app dep,
        "//empirical_calibration/data:kang_schafer",
        "//empirical_calibration/data:lalonde",
        # enum dep,
        # numpy dep,
        # pandas dep,
        # patsy dep,
        # scipy dep,
        # six dep,
        # sklearn dep,
        # typing dep,
    ],
)

py_test(
    name = "core_test",
    size = "small",
    srcs = [
        "core_test.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":core",
        "//testing/pybase:parameterized",
        # mock dep,
        # numpy dep,
        # scipy dep,
        # six dep,
    ],
)
