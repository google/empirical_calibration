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

"""LaLonde dataset to evaluate the effect of a job training program.
"""

import os
import pandas as pd


_URL_ROOT = "http://www.nber.org/~rdehejia/data"

_COLUMNS = [
    "treatment",
    "age",
    "education",
    "black",
    "hispanic",
    "married",
    "nodegree",
    "earnings1974",
    "earnings1975",
    "earnings1978",
]


def _read_txt(txt_file):
  return pd.read_csv(
      txt_file, delim_whitespace=True, header=None, names=_COLUMNS)


def experimental_treated():
  """Returns the experimental treated group.

  Returns:
    A pandas.DataFrame with shape (185, 10).
  """
  data = _read_txt(os.path.join(_URL_ROOT, "nswre74_treated.txt"))
  assert data.shape == (185, 10)
  return data


def experimental_control():
  """Returns the experimental control group.

  Returns:
    A pandas.DataFrame with shape (260, 10).
  """
  data = _read_txt(os.path.join(_URL_ROOT, "nswre74_control.txt"))
  assert data.shape == (260, 10)
  return data


def observational_control():
  """Returns the observational control group.

  Returns:
    A pandas.DataFrame with shape (15992, 10).
  """
  data = _read_txt(os.path.join(_URL_ROOT, "cps_controls.txt"))
  assert data.shape == (15992, 10)
  return data
