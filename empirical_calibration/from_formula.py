# Copyright 2019 Google LLC
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
"""Formular API for empirical balancing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import google_type_annotations
from __future__ import print_function

from empirical_calibration import empirical_calibration as ec
import numpy as np
import pandas as pd
import patsy
from typing import Text, Tuple

def _dmatrix_from_formula(formula: Text,
                          df: pd.DataFrame) -> pd.DataFrame:
  """Generates dmatrix from formula and dataframe.

  This is a wrapper around patsy's dmatrix function.

  Args:
    formula: Formula to be fed into patsy. No outcome variable allowed.
    df: Df containing the raw data.

  Returns:
    Design matrix.
  """
  dmatrix = patsy.highlevel.dmatrix(formula, df, return_type="dataframe")
  if "Intercept" in dmatrix.columns:
    dmatrix.drop(columns="Intercept", inplace=True)
  return dmatrix


def from_formula(formula: Text,
                 df: pd.DataFrame,
                 target_df: pd.DataFrame,
                 target_weights: np.ndarray = None,
                 autoscale: bool = False,
                 objective: ec.Objective = ec.Objective.ENTROPY,
                 max_weight: float = 1.0,
                 increment: float = 0.001) -> Tuple[np.ndarray, float]:
  """"Runs empirical calibration function from formula.

  This is the formula API of the maybe_exact_calibrate function.

  Args:
    formula: Formula used to generate design matrix.
      No outcome variable allowed.
    df: Data to be calibrated.
    target_df: Data containing the target.
    target_weights: Weights for target_df.
      If None, equal weights will be used.
    autoscale: Whether to scale  covariates to [0, 1] and apply the same
      scaling to target covariates. Setting it to True can help improve
      numerical stability.
    objective: The objective of the convex optimization problem.
    max_weight: The upper bound on weights. Must be between uniform weight
      (1 / _size) and 1.0.
    increment: The increment of the search sequence.

  Returns:
    A tuple of (weights, l2_norm) where
      weights: The weights for the  subjects. They should sum up to 1.
      l2_norm: The L2 norm of the covariate balance constraint.
  """
  target_covariates = _dmatrix_from_formula(formula=formula,
                                            df=target_df)
  covariates = _dmatrix_from_formula(formula=formula,
                                     df=df)

  return ec.maybe_exact_calibrate(
      covariates=covariates,
      target_covariates=target_covariates,
      target_weights=target_weights,
      autoscale=autoscale,
      objective=objective,
      max_weight=max_weight,
      increment=increment)
