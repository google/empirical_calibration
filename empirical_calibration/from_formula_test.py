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
"""Tests for from_formula."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from empirical_calibration import empirical_calibration as ec
from empirical_calibration import from_formula as fec

import numpy as np
import pandas as pd
import pandas.util.testing as pd_testing
import patsy
import unittest
from absl.testing import parameterized



class FromFormulaTest(parameterized.TestCase):

  def setUp(self):
    # Toy df and dmatrix for tests.
    self.seed_df = pd.DataFrame({
        "x": ["a", "b", "c"],
        "y": [1, 2, 3]
    }, columns=["x", "y"])
    self.seed_dmatrix = pd.DataFrame({
        "x[T.b]": [0.0, 1.0, 0.0],
        "x[T.c]": [0.0, 0.0, 1.0],
        "y": [1.0, 2.0, 3.0],
        "x[T.b]:y": [0.0, 2.0, 0.0],
        "x[T.c]:y": [0.0, 0.0, 3.0]
    }, columns=["x[T.b]", "x[T.c]",
                "y", "x[T.b]:y", "x[T.c]:y"])
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

    self.columns = ["x[T.b]", "x[T.c]", "y"]
    self.dmatrix = self.dmatrix[self.columns]
    self.target_dmatrix = self.target_dmatrix[self.columns]

  def test_dmatrix_from_formula_drop_intercept(self):
    # Intercept is dropped even when formula asks for Intercept.
    self.assertNotIn(
        "Intercept",
        fec._dmatrix_from_formula(formula="~ 1 + x + y + x:y",
                                  df=self.seed_df).columns)

  def test_dmatrix_from_formula_with_interaction(self):
    # x:y in formula asks for interaction.
    pd_testing.assert_frame_equal(
        self.seed_dmatrix,
        fec._dmatrix_from_formula(formula="~ x + y + x:y", df=self.seed_df))

  def test_dmatrix_from_formula_no_interaction(self):
    # No interaction, so only main effect terms are expected.
    pd_testing.assert_frame_equal(
        self.seed_dmatrix[["x[T.b]", "x[T.c]", "y"]],
        fec._dmatrix_from_formula(formula="~ x + y", df=self.seed_df))

  def test_dmatrix_from_formula_y_raises_error(self):
    # Error should be raised if dependent variable is specified in formula.
    with self.assertRaises(patsy.PatsyError):
      fec._dmatrix_from_formula(formula="y ~ x", df=self.seed_df)

  @parameterized.named_parameters(
      ("entropy, same weights", ec.Objective.ENTROPY, [1] * 12),
      ("quadratic, same weights", ec.Objective.QUADRATIC, [1] * 12),
      ("entropy,  varying weights", ec.Objective.ENTROPY, [1] * 4 + [1] * 8),
      ("quadratic, varying weights", ec.Objective.QUADRATIC, [1] * 4 + [1] * 8),
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
    weights_fec, l2_norm_fec = fec.from_formula(
        formula=formula,
        df=self.df,
        target_df=self.target_df,
        target_weights=target_weights,
        objective=objective)

    np.testing.assert_almost_equal(weights_ec, weights_fec, decimal=3)
    self.assertAlmostEqual(l2_norm_ec, l2_norm_fec, places=2)


if __name__ == "__main__":
  googletest.main()
