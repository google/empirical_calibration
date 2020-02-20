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


from empirical_calibration.data import kang_schafer as ks
import numpy as np
import unittest


class KangSchaferTest(googletest.TestCase):

  def test_simulation(self):
    np.random.seed(123)
    size = 1000
    simulation = ks.Simulation(size)

    self.assertEqual((size, 4), simulation.covariates.shape)
    self.assertEqual((size, 4), simulation.transformed_covariates.shape)
    self.assertLen(simulation.treatment, size)
    self.assertEqual([0.0, 1.0], np.unique(simulation.treatment).tolist())
    self.assertLen(simulation.outcome, size)
    self.assertTrue(all(simulation.outcome < 400))
    self.assertTrue(all(simulation.outcome > 0))


if __name__ == '__main__':
  googletest.main()
