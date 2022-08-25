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

import empirical_calibration as ec
from empirical_calibration.data import kang_schafer as ks
import mock
import numpy as np
import pandas as pd
import pandas.util.testing as pd_testing
import patsy
import scipy
import unittest
from absl.testing import parameterized


# Total number of rows in both covariates and target_covariates.
_SIZE = 2000


def _mock_calibrate(covariates, target_covariates, baseline_weights,
                    target_weights, autoscale,
                    objective, min_weight, max_weight, l2_norm):
  """Mocks the `calibrate` to return success only when `l2_norm` is large."""
  del target_covariates, covariates, target_weights, baseline_weights
  del autoscale, objective, min_weight, max_weight
  if l2_norm < _mock_calibrate.min_feasible_l2_norm:
    return None, False
  return None, True


class EmpiricalCalibrationTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(123)
    simulation = ks.Simulation(_SIZE)
    self.covariates = simulation.covariates[simulation.treatment == 0]
    self.target_covariates = simulation.covariates[simulation.treatment == 1]

  def assert_weights_constraints(self, weights, min_weight=0.0, max_weight=1.0):
    self.assertAlmostEqual(1.0, weights.sum())
    self.assertTrue(all(weights >= 0))
    # 1.00001 leaves a buffer for floating-point representation error.
    self.assertTrue(all(weights >= min_weight * (1.0-0.00001)))
    self.assertTrue(all(weights <= max_weight * 1.00001))

  def assert_balancing_constraint(self, weights, l2_norm=0.0):
    self.assertAlmostEqual(
        l2_norm,
        np.linalg.norm(
            np.mean(self.target_covariates, axis=0) -
            self.covariates.T @ weights))

  @parameterized.parameters(
      (ec.Objective.ENTROPY, 0.0, 1.0, 0.0),
      (ec.Objective.ENTROPY, 0.0, 1.0, 0.1),
      (ec.Objective.ENTROPY, 0.0, 0.005, 0.0),
      (ec.Objective.ENTROPY, 0.0, 0.005, 0.1),
      (ec.Objective.QUADRATIC, 0.0, 1.0, 0.0),
      (ec.Objective.QUADRATIC, 0.0, 1.0, 0.1),
      (ec.Objective.QUADRATIC, 0.0, 0.005, 0.0),
      (ec.Objective.QUADRATIC, 0.0, 0.005, 0.1),
      (ec.Objective.ENTROPY, 0.00005, 1.0, 0.0),
      (ec.Objective.ENTROPY, 0.00005, 1.0, 0.1),
      (ec.Objective.ENTROPY, 0.00005, 0.005, 0.0),
      (ec.Objective.ENTROPY, 0.00005, 0.005, 0.1),
      (ec.Objective.QUADRATIC, 0.00005, 1.0, 0.0),
      (ec.Objective.QUADRATIC, 0.00005, 1.0, 0.1),
      (ec.Objective.QUADRATIC, 0.00005, 0.005, 0.0),
      (ec.Objective.QUADRATIC, 0.00005, 0.005, 0.1),
  )
  def test_calibrate(self, objective, min_weight, max_weight, l2_norm):
    weights, success = ec.calibrate(
        covariates=self.covariates,
        target_covariates=self.target_covariates,
        objective=objective,
        min_weight=min_weight,
        max_weight=max_weight,
        l2_norm=l2_norm)
    self.assertTrue(success)
    self.assert_weights_constraints(weights, min_weight, max_weight)
    self.assert_balancing_constraint(weights, l2_norm)

  def test_entropy_balancing_bounded(self):
    weights_unbounded = ec.calibrate(
        covariates=self.covariates,
        target_covariates=self.target_covariates,
        objective=ec.Objective.ENTROPY,
        max_weight=1.0)[0]

    weights_bounded = ec.calibrate(
        covariates=self.covariates,
        target_covariates=self.target_covariates,
        objective=ec.Objective.ENTROPY,
        max_weight=0.005)[0]

    # Imposing an upper bound on weights should lead to a worse objective.
    self.assertLessEqual(-scipy.stats.entropy(weights_unbounded),
                         -scipy.stats.entropy(weights_bounded))

  def test_entropy_balancing_approximate(self):
    weights_exact = ec.calibrate(
        covariates=self.covariates,
        target_covariates=self.target_covariates,
        objective=ec.Objective.ENTROPY,
        l2_norm=0.0)[0]

    weights_approximate = ec.calibrate(
        covariates=self.covariates,
        target_covariates=self.target_covariates,
        objective=ec.Objective.ENTROPY,
        l2_norm=0.1)[0]

    # Approximate match should lead to a better objective.
    self.assertGreaterEqual(-scipy.stats.entropy(weights_exact),
                            -scipy.stats.entropy(weights_approximate))

  def test_quadratic_balancing_bounded(self):
    weights_unbounded = ec.calibrate(
        covariates=self.covariates,
        target_covariates=self.target_covariates,
        objective=ec.Objective.QUADRATIC,
        max_weight=1.0)[0]

    weights_bounded = ec.calibrate(
        covariates=self.covariates,
        target_covariates=self.target_covariates,
        objective=ec.Objective.QUADRATIC,
        max_weight=0.005)[0]

    # Imposing an upper bound on weights should lead to a worse objective.
    self.assertLessEqual(
        np.sum(np.square(weights_unbounded)), np.sum(
            np.square(weights_bounded)))

  def test_quadratic_balancing_approximate(self):
    weights_exact = ec.calibrate(
        covariates=self.covariates,
        target_covariates=self.target_covariates,
        objective=ec.Objective.QUADRATIC,
        l2_norm=0.0)[0]

    weights_approximate = ec.calibrate(
        covariates=self.covariates,
        target_covariates=self.target_covariates,
        objective=ec.Objective.QUADRATIC,
        l2_norm=0.1)[0]

    # Approximate match should lead to a better objective.
    self.assertGreaterEqual(
        np.sum(np.square(weights_exact)), np.sum(
            np.square(weights_approximate)))

  @parameterized.parameters((ec.Objective.ENTROPY), (ec.Objective.QUADRATIC))
  def test_autoscale(self, objective):
    weights_raw = ec.calibrate(
        covariates=self.covariates,
        target_covariates=self.target_covariates,
        autoscale=False,
        objective=objective)[0]

    weights_scaled = ec.calibrate(
        covariates=self.covariates,
        target_covariates=self.target_covariates,
        autoscale=True,
        objective=objective)[0]

    # With and without autoscale should give almost identical weight.
    self.assertAlmostEqual(0, np.linalg.norm(weights_raw - weights_scaled))

  def test_invalid_l2_norm(self):
    # Should fail with negative l2_norm.
    with self.assertRaises(ValueError):
      ec.calibrate(
          covariates=self.covariates,
          target_covariates=self.target_covariates,
          l2_norm=-0.01)

  def test_invalid_max_weight(self):
    # Should fail when the max_weight is smaller than the uniform weight.
    with self.assertRaises(ValueError):
      ec.calibrate(
          covariates=self.covariates,
          target_covariates=self.target_covariates,
          max_weight=1.0 / (_SIZE + 1000))

  def test_invalid_min_weight(self):
    # Should fail when the min_weight is larger than the uniform weight.
    with self.assertRaises(ValueError):
      ec.calibrate(
          covariates=self.covariates,
          target_covariates=self.target_covariates,
          min_weight=1.0 / (_SIZE - 1000))

  @parameterized.parameters((0.0), (0.07), (0.12))
  @mock.patch("__main__.ec.core.calibrate",
              side_effect=_mock_calibrate)
  def test_maybe_exact_calibrate(
      self,
      min_feasible_l2_norm,
      mock_maybe_exact_calibrate):  # pylint: disable=unused-argument
    _mock_calibrate.min_feasible_l2_norm = min_feasible_l2_norm
    self.assertEqual(
        ec.maybe_exact_calibrate(
            covariates=None,
            target_covariates=None,
            target_weights=None,
            autoscale=None,
            objective=None,
            min_weight=None,
            max_weight=None,
            increment=0.01)[1],
        min_feasible_l2_norm)

  def test_target_weights(self):
    # Replicating the first 10 rows of self.target_covariates should be
    # equivalent to
    # assigning a weight of 2 to each of the first 10 rows and 1 for others.
    n = len(self.target_covariates)
    index = list(range(10)) + list(range(n))
    weights = [2] * 10 + [1] * (n - 10)
    duplicated_weights, duplicated_l2 = ec.maybe_exact_calibrate(
        covariates=self.covariates,
        target_covariates=self.target_covariates[index])
    weighted_weights, weighted_l2 = ec.maybe_exact_calibrate(
        covariates=self.covariates,
        target_covariates=self.target_covariates,
        target_weights=weights)
    self.assertAlmostEqual(duplicated_l2, weighted_l2)
    self.assertAlmostEqual(
        0.0, np.linalg.norm(duplicated_weights - weighted_weights))
 
  @parameterized.named_parameters(
      ("entropy", ec.Objective.ENTROPY),
      ("quadratic", ec.Objective.QUADRATIC),
  )
  def test_uniform_weights(self, objective):
    # when covariates == target_covariates, uniform weights should be returned.
    covariates = target_covariates = np.random.normal(size=(100, 4))
    weights, success = ec.calibrate(
        covariates=covariates,
        target_covariates=target_covariates,
        objective=objective
    )
    self.assertTrue(all(np.isclose(weights, 1.0 / 100)))

  @parameterized.named_parameters(
      ("entropy", ec.Objective.ENTROPY),
      ("quadratic", ec.Objective.QUADRATIC),
  )
  def test_baseline_weights(self, objective):
    covariates = target_covariates = np.random.normal(size=(100, 2))
    repeats = np.repeat([2, 1], 50)
    weights, success = ec.calibrate(
        covariates=covariates,
        target_covariates=target_covariates,
        baseline_weights=repeats,
        target_weights=repeats,
        objective=ec.Objective.QUADRATIC
    )
    # First 50 rows should have 2 / 3 of the weights.
    expected = repeats * 2 / 3 / 100
    self.assertTrue(all(np.isclose(weights, expected)))


class FromFormulaTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Toy df and dmatrix for tests.
    self.seed_df = pd.DataFrame({
        "x": ["a", "b", "c"],
        "y": [1, 2, 3]
    }, columns=["x", "y"])
    self.seed_dmatrix = pd.DataFrame({
        "x[a]": [1.0, 0.0, 0.0],
        "x[b]": [0.0, 1.0, 0.0],
        "x[c]": [0.0, 0.0, 1.0],
        "y": [1.0, 2.0, 3.0],
        "x[a]:y": [1.0, 0.0, 0.0],
        "x[b]:y": [0.0, 2.0, 0.0],
        "x[c]:y": [0.0, 0.0, 3.0]
    }, columns=["x[a]", "x[b]", "x[c]",
                "y", "x[a]:y", "x[b]:y", "x[c]:y"])
    # Generate df's and dmatrix's with more rows.
    self.idx = [0, 0,
                1, 1, 1, 1,
                2, 2, 2, 2, 2, 2, 2, 2]
    self.target_idx = [0, 0, 0, 0,
                       1, 1, 1, 1,
                       2, 2, 2, 2]

    self.df = self.seed_df.iloc[self.idx]
    self.target_df = self.seed_df.iloc[self.target_idx]

    # Drop interaction terms.
    self.dmatrix = self.seed_dmatrix.iloc[self.idx]
    self.target_dmatrix = self.seed_dmatrix.iloc[self.target_idx]

    self.columns = ["x[a]", "x[b]", "x[c]", "y"]
    self.dmatrix = self.dmatrix[self.columns]
    self.target_dmatrix = self.target_dmatrix[self.columns]

  def test_dmatrix_from_formula_drop_intercept(self):
    # Intercept is dropped even when formula asks for Intercept.
    self.assertNotIn(
        "Intercept",
        ec.dmatrix_from_formula(formula="~ 1 + x + y + x:y",
                                df=self.seed_df).columns)

  def test_dmatrix_from_formula_with_interaction(self):
    # x:y in formula asks for interaction.
    pd_testing.assert_frame_equal(
        self.seed_dmatrix,
        ec.dmatrix_from_formula(formula="~ x + y + x:y", df=self.seed_df))

  def test_dmatrix_from_formula_no_interaction(self):
    # No interaction, so only main effect terms are expected.
    pd_testing.assert_frame_equal(
        self.seed_dmatrix[["x[a]", "x[b]", "x[c]", "y"]],
        ec.dmatrix_from_formula(formula="~ x + y", df=self.seed_df))

  def test_dmatrix_from_formula_y_raises_error(self):
    # Error should be raised if dependent variable is specified in formula.
    with self.assertRaises(patsy.PatsyError):
      ec.dmatrix_from_formula(formula="y ~ x", df=self.seed_df)

  @parameterized.named_parameters(
      ("entropy, same weights", ec.Objective.ENTROPY, [1] * 12),
      ("quadratic, same weights", ec.Objective.QUADRATIC, [1] * 12),
      ("entropy, varying weights", ec.Objective.ENTROPY, [1] * 4 + [2] * 8),
      ("quadratic, varying weights", ec.Objective.QUADRATIC, [1] * 4 + [2] * 8),
  )
  def test_from_formula(self, objective, target_weights):
    # Two api should give the same results.
    # _ec indicates the original empirical_calibration API.
    weights_ec, l2_norm_ec = ec.maybe_exact_calibrate(
        covariates=self.dmatrix,
        target_covariates=self.target_dmatrix,
        target_weights=target_weights,
        objective=objective)

    # _fec indicates empirical_calibration's formula API.
    formula = "~ x + y"
    weights_fec, l2_norm_fec = ec.from_formula(
        formula=formula,
        df=self.df,
        target_df=self.target_df,
        target_weights=target_weights,
        objective=objective)

    np.testing.assert_almost_equal(weights_ec, weights_fec, decimal=3)
    self.assertAlmostEqual(l2_norm_ec, l2_norm_fec, places=2)


if __name__ == "__main__":
  unittest.main()
