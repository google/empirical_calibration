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

"""Kang-Schafer simulation to evaluate observational causal methods.
"""

import numpy as np


class Simulation:
  """Kang-Schafer simulation.

  The simulation can be used to illustrate selection bias of outcome under
  informative nonresponse. Due to the selection bias, the outcome mean for the
  treated group (200) is lower than the control group (220), but the difference
  is not attributed to the treatment. In fact, given the covariates, the outcome
  is generated independent of treatment, i.e., the true average treatment effect
  on the treated (ATT) is zero.

  #### Examples

  ```python
  simulation = kang_schafer.Simulation(200)

  # Simple difference in means.
  diff = np.mean(simulation.outcome[simulation.treatment == 1]) -
    np.mean(simulation.outcome[simulation.treatment == 0])

  # Estimate the ATT using empirical calibration.
  weights = empirical_calibration.balance(
    simulation.covariates[simulation.treatment == 1],
    simulation.covariates[simulation.treatment == 0])[0]

  att = np.mean(simulation.outcome[simulation.treatment == 1]) -
    np.mean(simulation.outcome[simulation.treatment == 0] * weights)
  ```

  Attributes:
    size: The total number of treated and control units.
    covariates: Raw covariates, generated as i.i.d. standard normal samples.
    treatment: Unit-level treatment assignments, generated from a logistic
      regression model of the covariates.
    outcome: Unit-level outcomes, generated from a linear regression model of
      the covariates.
  """

  def __init__(self, size: int = 2000):
    """Constructs a `Simulation` instance.

    Args:
      size: The total number of treated and control units.
    """
    self.size = size

    self.covariates = np.random.randn(size, 4)

    propensity_score = 1.0 / (1.0 + np.exp(
        -np.dot(self.covariates, np.array([-1.0, 0.5, -0.25, -0.1]))))
    self.treatment = np.random.binomial(1, propensity_score)

    self.outcome = 210.0 + np.dot(self.covariates, np.array(
        [27.4, 13.7, 13.7, 13.7])) + np.random.randn(size)  # pyformat: disable

  @property
  def transformed_covariates(self) -> np.ndarray:
    """Returns nonlinear transformations of the raw covariates.

    When the transformed covariates are observed in place of the true
    covariates, both propensity score and outcome regression models become
    misspecified.
    """
    x1, x2, x3, x4 = np.hsplit(self.covariates, 4)
    return np.hstack([
        np.exp(x1 / 2.0),
        x2 / (1 + np.exp(x1)) + 10.0,
        np.power(x1 * x3 / 25 + 0.6, 3),
        np.square(x2 + x4 + 20.0),
    ])
