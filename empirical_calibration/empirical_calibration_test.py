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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from empirical_calibration import empirical_calibration as ec
from empirical_calibration.data import kang_schafer as ks
import mock
import numpy as np
import scipy
from six.moves import range
import unittest
from absl.testing import parameterized


# Total number of rows in both covariates and target_covariates.
_SIZE = 2000


def _mock_calibrate(covariates, target_covariates, target_weights, autoscale,
                    objective, max_weight, l2_norm):
  """Mocks the `calibrate` to return success only when `l2_norm` is large."""
  del target_covariates, covariates, target_weights,
  del autoscale, objective, max_weight
  if l2_norm < _mock_calibrate.min_feasible_l2_norm:
    return None, False
  return None, True


class EmpiricalCalibrationTest(parameterized.TestCase):

  def setUp(self):
    super(EmpiricalCalibrationTest, self).setUp()
    np.random.seed(123)
    simulation = ks.Simulation(_SIZE)
    self.covariates = simulation.covariates[simulation.treatment == 0]
    self.target_covariates = simulation.covariates[simulation.treatment == 1]

  def assert_weights_constraints(self, weights, max_weight=1.0):
    self.assertAlmostEqual(1.0, weights.sum())
    self.assertTrue(all(weights >= 0))
    self.assertTrue(all(weights <= max_weight))

  def assert_balancing_constraint(self, weights, l2_norm=0.0):
    self.assertAlmostEqual(
        l2_norm,
        np.linalg.norm(
            np.mean(self.target_covariates, axis=0) -
            np.matmul(self.covariates.T, weights)))

  @parameterized.parameters(
      (ec.Objective.ENTROPY, 1.0, 0.0),
      (ec.Objective.ENTROPY, 1.0, 0.1),
      (ec.Objective.ENTROPY, 0.005, 0.0),
      (ec.Objective.ENTROPY, 0.005, 0.1),
      (ec.Objective.QUADRATIC, 1.0, 0.0),
      (ec.Objective.QUADRATIC, 1.0, 0.1),
      (ec.Objective.QUADRATIC, 0.005, 0.0),
      (ec.Objective.QUADRATIC, 0.005, 0.1),
  )
  def test_calibrate(self, objective, max_weight, l2_norm):
    weights, success = ec.calibrate(
        covariates=self.covariates,
        target_covariates=self.target_covariates,
        objective=objective,
        max_weight=max_weight,
        l2_norm=l2_norm)
    self.assertTrue(success)
    self.assert_weights_constraints(weights, max_weight)
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

  @parameterized.parameters((0.0), (0.07), (0.12))
  @mock.patch("__main__.ec.calibrate", side_effect=_mock_calibrate)
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
            max_weight=None,
            increment=0.01)[1],
        min_feasible_l2_norm)

  def test_target_weights(self):
    # Replicating the first 10 rows of self.target_covariates should be
    # equivalent to
    # assigning a weight of 2 to each of the first 10 rows and 0 for others.
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


if __name__ == "__main__":
  googletest.main()
