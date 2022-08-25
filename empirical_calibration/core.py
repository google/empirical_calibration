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

"""Empirical calibration.

References:

  Hainmueller, J. (2012). Entropy balancing for causal effects: A multivariate
  reweighting method to produce balanced samples in observational studies.
  Political Analysis, 20(1), 25-46.
  https://web.stanford.edu/~jhain/Paper/PA2012.pdf

  Wang, X., Miao, J., & Sun, Y. (2019). A Python library For empirical
  calibration.
  https://arxiv.org/abs/1906.11920

  Zhao, Q., & Percival, D. (2017). Entropy balancing is doubly robust. Journal
  of Causal Inference, 5(1).
  https://arxiv.org/abs/1501.03571
"""

import enum
from typing import Tuple

from absl import logging
import numpy as np
import pandas as pd
import patsy
from scipy import optimize
from sklearn import preprocessing


_MAX_EXPONENT = 20


@enum.unique
class Objective(enum.Enum):
  """Objective of the convex optimization problem."""

  # Minimizes the negative entropy: `\sum_i w_i * log(w_i)`. It effectively
  # minimizes the Kullback-Leibler divergence between the balancing weights and
  # uniform weights.
  ENTROPY = 0

  # Minimizes the sum of squares: `\sum_i w_i^2`. It effectively minimizes the
  # Euclidean distance between the balancing weight and uniform weight, or
  # maximizes the effective sample size.
  QUADRATIC = 1

  
class ConvergenceError(Exception):
  pass


def calibrate(covariates: np.ndarray,
              target_covariates: np.ndarray,
              baseline_weights: np.ndarray = None,
              target_weights: np.ndarray = None,
              autoscale: bool = False,
              objective: Objective = Objective.QUADRATIC,
              min_weight: float = 0.0,
              max_weight: float = 1.0,
              l2_norm: float = 0) -> Tuple[np.ndarray, bool]:
  """Calibrates covariates toward target.
  
  It solves a constrained convex optimization problem that minimizes the
  variation of weights for units while achieving direct covariate
  balance. The weighted mean of covariates would match the simple mean
  of target covariates up to a prespecified L2 norm. If there is no feasible
  solution of weights to satisfy the covariate balance constraint, it would
  exit with the boolean optimization status set false, in which case, one needs
  to relax the covariate constraint by increasing the `l2_norm`, or use the
  `maybe_exact_calibrate` function to find a feasible solution with the smallest
  `l2_norm`.
  Only the first moment is matched between the input covariates and
  targert_covariates. To match higher moments or some transformation of the raw
  covariates, one needs to manually include them in the input. Similarly, to
  match categorical variables, one needs to convert them into dummy/indicator
  variables first. Zhao (2017) shows that the covariates to be balanced serve as
  the linear predictors in the induced propensity score model and the outcome
  regression model.
  There are two choices of the optimization objective: entropy of the weights
  (entropy balancing, or EB) and effective sample size implied by the weights
  (quadratic balancing, or QB). EB can be viewed as minimizing the
  Kullback-Leibler divergence between the optimal weights and equal weights;
  while QB effectively minimizes the Euclidean distance between the optimal
  weights and equal weights. The two objectives correspond to different link
  functions for the weights (or the odds of propensity scores) - `exp(x)` for EB
  and `max(x, 0)` for QB. Therefore, EB weights are strictly positive; while QB
  weights can be zero and induce sparsity.
  #### Examples
  ```python
  # Entropy balancing, exactly match treatment and weighted control.
  weights, success = empirical_calibration.calibrate(
      covariates=control_covariates,
      target_convariates=treatment_covariates)
  # Estimate the average target effect on the treated (ATT) for an univariate
  # outcome.
  att = np.mean(treatment_outcome) - np.sum(control_outcome * weights)
  # Estimate ATT for a multivariate outcome.
  att = np.mean(
      treatment_outcome, axis=0) - control_outcome.T @ weights
  ```
  Args:
    covariates: covariates to be calibrated. All values must be numeric.
      For categorical values, the `from_formula` function is often more
      convenient.
    target_covariates: covariates to be used as target in calibration. The
      number of columns should match `covariates`. All values must be numeric.
      For categorical values, the `from_formula` function is often more
      convenient.
    baseline_weights: baseline weights. For example, survey data may come
      with weights already. If None, uniform baseline weights will be used.
      If given, instead of coming up with weights close to uniform, the
      function will try to come up with weights close to the baseline weights.
    target_weights: Weights for target_covariates. These are needed when the
      target_covariates themselves have weights. Its length must equal the
      number of rows in target_covariates. If None, equal weights will be used.
    autoscale: Whether to scale `covariates` to [0, 1] and apply the same
      scaling to `target_covariates`. Setting it to True might help improve
      numerical stability.
    objective: The objective of the convex optimization problem. Supported
      values are Objective.ENTROPY and Objective.QUADRATIC.
    min_weight: The lower bound on weights. Must be between 0.0 and the uniform
      weight (1 / number of rows in `covariates`).
    max_weight: The upper bound on weights. Must be between the uniform weight
      (1 / number of rows in `covariates`) and 1.0.
    l2_norm: The L2 norm of the covaraite balance constraint, i.e., the
      Euclidean distance between the weighted mean of covariates and the simple
      mean of target covaraites after balancing.
  Returns:
    A tuple of (weights, success) where
      weights: The weights for the subjects. They should sum up to 1.
      success: Whether the constrained optimization succeeds.
  Raises:
    ValueError: incompatible target and covariates or invalid balancing options.
  """

  if target_covariates.shape[1] != covariates.shape[1]:
    raise ValueError(
        "unequal number of columns in target covariates (%d) and "
        " covariates (%d)" % (target_covariates.shape[1], covariates.shape[1]))
  if autoscale:
    scaler = preprocessing.MinMaxScaler()
    # For each covariate, find its min and max, and rescale to [0, 1].
    covariates = scaler.fit_transform(covariates)
    # For each target covariate, apply the same rescaling.
    target_covariates = scaler.transform(target_covariates)

  if l2_norm < 0:
    raise ValueError("l2_norm %f is negative" % l2_norm)

  num_samples, num_covariates = covariates.shape
  uniform_weight = 1.0 / num_samples
  if max_weight < uniform_weight:
    raise ValueError("max_weight %f cannot be smaller than uniform weight %f" %
                     (max_weight, uniform_weight))
  if min_weight > uniform_weight:
    raise ValueError("min_weight %f cannot be larger than uniform weight %f" %
                     (min_weight, uniform_weight))

  baseline_weights_is_none = baseline_weights is None
  if baseline_weights is not None:
    baseline_weights = baseline_weights / sum(baseline_weights)

  z = np.hstack(
      (np.expand_dims(np.ones(num_samples), 1),
       covariates - np.average(target_covariates,
                               weights=target_weights, axis=0)))

  if objective == Objective.ENTROPY:
    # Running entropy balancing directly with a weight bound may fail due to bad
    # initial value for beta, so we first run it with a huge bound (1e8) to get
    # a good guess of beta.
    if baseline_weights_is_none:
      weight_link = lambda x: np.exp(np.minimum(x, np.log(1e8)))
    else:
      weight_link = lambda x: np.exp(
          np.minimum(np.log(baseline_weights) + (x - 1), np.log(1e8)))
    beta_init = np.zeros(num_covariates + 1)
  elif objective == Objective.QUADRATIC:
    if baseline_weights_is_none:
      weight_link = lambda x: np.clip(x, min_weight, max_weight)
      # Solution of the dual problem without the non-negative weight constraint.
      # Use pseudoinverse in case z is not full rank.
      beta_init = np.linalg.pinv(z.T @ z) @ np.concatenate(
          (np.ones(1), np.zeros(num_covariates)))
    else:
      weight_link = lambda x: np.clip(x * baseline_weights + baseline_weights,
          min_weight, max_weight)
      # Solution of the dual problem without the non-negative weight constraint.
      # Use pseudoinverse in case z is not full rank.
      beta_init = np.linalg.pinv(z.T @ np.diag(baseline_weights) @ z) @ (
          np.concatenate(
              (np.ones(1), np.zeros(num_covariates))) - z.T @ baseline_weights)
  else:
    raise ValueError("unknown objective %s" % objective)

  def estimating_equation(beta):
    weights = weight_link(np.dot(z, beta))

    norm = np.linalg.norm(beta[1:])
    if norm == 0.0:
      slack = np.zeros(len(beta[1:]))
    else:
      slack = l2_norm * beta[1:] / norm
    return np.dot(z.T, weights) + np.concatenate((-np.ones(1), slack))

  logging.info(
      "Running calibration with objective=%s, autoscale=%s, l2_norm=%s, "
      "max_weight=%s, min_weight=%s:", objective.name, autoscale, l2_norm,
      (1.0 if objective == Objective.ENTROPY else max_weight),
      (0.0 if objective == Objective.ENTROPY else min_weight))
  beta, info_dict, status, msg = optimize.fsolve(
      estimating_equation, x0=beta_init, full_output=True)
  weights = weight_link(np.dot(z, beta))
  logging.info(msg)
  logging.info("Number of function calls: %d", info_dict["nfev"])

  if objective == Objective.ENTROPY and ((np.max(weights) > max_weight) or
                                         (np.min(weights) < min_weight)):
    if baseline_weights_is_none:
      if min_weight == 0.0:
        weight_link = lambda x: np.exp(np.minimum(x, np.log(max_weight)))
      else:
        weight_link = lambda x: np.exp(
            np.clip(x, np.log(min_weight), np.log(max_weight)))
    else:
      if min_weight == 0.0:
        weight_link = lambda x: np.exp(
            np.minimum(np.log(baseline_weights) + (x - 1), np.log(max_weight)))
      else:
        weight_link = lambda x: np.exp(
            np.clip(
                np.log(baseline_weights) +
                (x - 1), np.log(min_weight), np.log(max_weight)))

    logging.info(
        "Running calibration with objective=%s, autoscale=%s, l2_norm=%s, "
        "max_weight=%s, min_weight=%s:", objective.name, autoscale, l2_norm,
        max_weight, min_weight)
    beta, info_dict, status, msg = optimize.fsolve(
        estimating_equation, x0=beta, full_output=True)
    weights = weight_link(np.dot(z, beta))
    logging.info(msg)
    logging.info("Number of function calls: %d", info_dict["nfev"])

  if np.abs(np.sum(weights) - 1.0) > 1e-3:
    return weights, False

  success = status == 1
  return weights, success


def maybe_exact_calibrate(covariates: np.ndarray,
                          target_covariates: np.ndarray,
                          baseline_weights: np.ndarray = None,
                          target_weights: np.ndarray = None,
                          autoscale: bool = False,
                          objective: Objective = Objective.QUADRATIC,
                          min_weight: float = 0.0,
                          max_weight: float = 1.0,
                          increment: float = 0.001) -> Tuple[np.ndarray, float]:
  """Finds feasible weights with the tightest covariate balance constraint.
  
  It is possible that there is no feasible solution for the weighted mean of
  covariates to exactly match the mean of target covariates. In such case, one
  needs to relax the covariate balance constraint. This function searches on a
  sequence of constraints for the tightest one that yields a feasible solution.
  The search is efficiently implemented with a bracket phase followed by a zoom
  phase.
  Args:
    covariates: covariates to be calibrated. All values must be numeric.
      For categorical values, the `from_formula` function is often more
      convenient.
    target_covariates: covariates to be used as target in calibration. The
      number of columns should match `covariates`. All values must be numeric.
      For categorical values, the `from_formula` function is often more
      convenient.
    baseline_weights: baseline weights. For example, survey data may come
      with weights already. If None, uniform baseline weights will be used.
      If given, instead of coming up with weights close to uniform, the
      function will try to come up with weights close to the baseline weights.
    target_weights: Weights for target_covariates. These are needed when the
      target_covariates itself has weights. Its length must equal the number of
      rows in target_covariates. If None, equal weights will be used.
    autoscale: Whether to scale `covariates` to [0, 1] and apply the same
      scaling to `target_covariates`. Setting it to True can help improve
      numerical stability.
    objective: The objective of the convex optimization problem.
    min_weight: The lower bound on weights. Must be between 0.0 and the uniform
      weight (1 / number of rows in `covariates`).
    max_weight: The upper bound on weights. Must be between the uniform weight
      (1 / number of rows in `covariates`) and 1.0.
    increment: The increment of the search sequence.
  Returns:
    A tuple of (weights, l2_norm) where
      weights: The weights for the subjects. They should sum up to 1.
      l2_norm: The L2 norm of the covariate balance constraint.
  Raises:
    ConvergenceError: If the calibration could not converge for any l2_norm.
  """
  if autoscale:
    scaler = preprocessing.MinMaxScaler()
    covariates = scaler.fit_transform(covariates)
    target_covariates = scaler.transform(target_covariates)

  def bracket():
    """Brackets the tightest covariate balance constraint by a power series."""
    for j in range(_MAX_EXPONENT):
      right = np.power(2, j, dtype=int) - 1
      l2_norm = right * increment
      weights, success = calibrate(
          covariates,
          target_covariates,
          baseline_weights,
          target_weights,
          autoscale=False,
          objective=objective,
          min_weight=min_weight,
          max_weight=max_weight,
          l2_norm=l2_norm)
      logging.info("Bracketing %s with l2_norm = %d*increment = %f",
                   ("succeeded" if success else "failed"), right, l2_norm)
      if success:
        break

    return weights, right, success

  def zoom(left, right, weights):
    """Recursively halves the interval until its length becomes `increment`."""
    if left + 1 == right:
      return weights, right * increment
    mid = (left + right) // 2
    l2_norm = mid * increment
    mid_weights, success = calibrate(
        covariates,
        target_covariates,
        baseline_weights,
        target_weights,
        autoscale=False,
        objective=objective,
        min_weight=min_weight,
        max_weight=max_weight,
        l2_norm=l2_norm)
    logging.info("Zooming %s with l2_norm = %d*increment = %f",
                 ("succeeded" if success else "failed"), mid, l2_norm)
    if success:
      return zoom(left, mid, mid_weights)

    return zoom(mid, right, weights)

  weights, right, success = bracket()
  if not success:
    raise ConvergenceError(
        "None of the attempted l2_norm error bounds gave a solution.")
  if right == 0:
    return weights, 0

  left = (right + 1) // 2 - 1
  return zoom(left, right, weights)


def dmatrix_from_formula(formula: str, df: pd.DataFrame) -> pd.DataFrame:
  """Generates dmatrix from formula and dataframe.

  Builds a design matrix by applying patsy's dmatrix function one variable at a
  time to ensure that reference categories of categorical variables are not
  dropped.
  Args:
    formula: patsy-style formula. No outcome variable allowed. Interactions
      between categorical variables of the form dim1*dim2 should not be used
      (use dim1:dim2 instead).
    df: Df containing the raw data.
  Returns:
    Design matrix.
  """

  # Parse the formula into a list of dimensions
  dimensions = formula.replace(" ", "").replace("~-1+", "").replace(
      "~0+", "").replace("~1+", "").replace("~", "").split("+")
  return pd.concat([
      patsy.highlevel.dmatrix(
          f"~ 0 + {dimension}", df,
          return_type="dataframe") for dimension in dimensions
      ], axis=1)


def from_formula(formula: str,
                 df: pd.DataFrame,
                 target_df: pd.DataFrame,
                 baseline_weights: np.ndarray = None,
                 target_weights: np.ndarray = None,
                 autoscale: bool = False,
                 objective: Objective = Objective.QUADRATIC,
                 min_weight: float = 0.0,
                 max_weight: float = 1.0,
                 increment: float = 0.001) -> Tuple[np.ndarray, float]:
  """"Runs empirical calibration function from formula.
  
  This is the formula API of the maybe_exact_calibrate function.
  Args:
    formula: Formula used to generate design matrix.
      No outcome variable allowed.
      Interactions between categorical variables of the form dim1*dim2 should
        not be used (use dim1:dim2 instead).
    df: Data to be calibrated.
    target_df: Data containing the target.
    baseline_weights: baseline weights. For example, survey data may come
      with weights already. If None, uniform baseline weights will be used.
      If given, instead of coming up with weights close to uniform, the
      function will try to come up with weights close to the baseline weights.
    target_weights: Weights for target_df. These are needed when the
      target_df itself has weights. Its length must equal the number of
      rows in target_df. If None, equal weights will be used.
    autoscale: Whether to scale `covariates` to [0, 1] and apply the same
      scaling to `target_covariate`. Setting it to True can help improve
      numerical stability.
    objective: The objective of the convex optimization problem.
    min_weight: The lower bound on weights. Must be between 0.0 and the uniform
      weight (1 / number of rows in `df`).
    max_weight: The upper bound on weights. Must be between the uniform weight
      (1 / number of rows in `df`) and 1.0.
    increment: The increment of the search sequence.
  Returns:
    A tuple of (weights, l2_norm) where
      weights: The weights for the subjects. They should sum up to 1.
      l2_norm: The L2 norm of the covariate balance constraint.
  Raises:
    ConvergenceError: If the calibration could not converge for any l2_norm.
  """
  target_covariates = dmatrix_from_formula(formula=formula, df=target_df)
  covariates = dmatrix_from_formula(formula=formula, df=df)

  return maybe_exact_calibrate(
      covariates=covariates,
      target_covariates=target_covariates,
      baseline_weights=baseline_weights,
      target_weights=target_weights,
      autoscale=autoscale,
      objective=objective,
      min_weight=min_weight,
      max_weight=max_weight,
      increment=increment)
