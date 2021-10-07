import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import torch
import unittest
import numpy as np
from unittest.mock import MagicMock
from src.utils.integralapproximation import riemann_sum, analytical_squared_euclidean
from src.models.intensityfunctions.commonbias import CommonBias

class TestIntegralApproximation(unittest.TestCase):

    def test_riemann_zero_integral(self):
        i = 0
        j = 1
        t0 = 0
        tn = 6
        n_samples = 10
        intensity_func = CommonBias(1.)
        intensity_func.result = MagicMock(return_value=0)
        self.assertEqual(riemann_sum(t0, tn, n_samples, zt=lambda t: None, i=None, j=None, func=intensity_func), torch.tensor([0]))

    def test_riemann_given_tn_6_and_constant_func_5_expects_30(self):
        i = 0
        j = 1
        t0 = 0
        tn = 6
        n_samples = 10
        intensity_func = CommonBias(1.)
        intensity_func.result = MagicMock(return_value=5)
        self.assertEqual(riemann_sum(t0, tn, n_samples, lambda t: None, None, None, intensity_func), torch.tensor([30]))

    def test_riemann_given_tn_6_and_dynamic_func_expects_36(self):
        i = 0
        j = 1
        t0 = 0
        tn = 6
        n_samples = 10
        intensity_func = CommonBias(1.)
        intensity_func.result = lambda t, i, j: 2*t
        self.assertEqual(riemann_sum(t0, tn, n_samples, lambda t: t, i, j, intensity_func), torch.tensor([36]))

    def test_analytical_positive_positions_velocity_integral(self):
        i = 0
        j = 1
        t0 = 0
        tn = 6
        intensity_func = CommonBias(1.)
        z = torch.Tensor(np.asarray([[1,1], [2,2]]))
        v = torch.Tensor(np.asarray([[1,0], [0,1]]))
        self.assertAlmostEqual(analytical_squared_euclidean(t0, tn, lambda t: z, v, i, j, intensity_func).item(), torch.tensor([0.8915]).item(), places=2)

    def test_analytical_negative_positions_integral(self):
        i = 0
        j = 1
        t0 = 0
        tn = 6
        intensity_func = CommonBias(1.)
        z = torch.Tensor(np.asarray([[-1,1], [-2,2]]))
        v = torch.Tensor(np.asarray([[1,0], [0,1]]))
        self.assertAlmostEqual(analytical_squared_euclidean(t0, tn, lambda t: z, v, i, j, intensity_func).item(), torch.tensor([0.1206]).item(), places=2)

    def test_analytical_negative_velocities_integral(self):
        i = 0
        j = 1
        t0 = 0
        tn = 6
        intensity_func = CommonBias(1.)
        z = torch.Tensor(np.asarray([[1,1], [2,2]]))
        v = torch.Tensor(np.asarray([[-1,0], [0,-1]]))
        self.assertAlmostEqual(analytical_squared_euclidean(t0, tn, lambda t: z, v, i, j, intensity_func).item(), torch.tensor([4.725]).item(), places=2)

    def test_analytical_large_bias_integral(self):
        i = 0
        j = 1
        t0 = 0
        tn = 6
        intensity_func = CommonBias(10.)
        z = torch.Tensor(np.asarray([[1,1], [2,2]]))
        v = torch.Tensor(np.asarray([[-1,0], [0,-1]]))
        self.assertAlmostEqual(analytical_squared_euclidean(t0, tn, lambda t: z, v, i, j, intensity_func).item(), torch.tensor([38293.58138]).item(), places=2)

    def test_analytical_zero_integral(self):
        i = 0
        j = 1
        t0 = 0
        tn = 0
        intensity_func = CommonBias(1.)
        z = torch.Tensor(np.asarray([[1,1], [2,2]]))
        v = torch.Tensor(np.asarray([[-1,0], [0,-1]]))
        self.assertAlmostEqual(analytical_squared_euclidean(t0, tn, lambda t: z, v, i, j, intensity_func).item(), torch.tensor([0.]).item(), places=2)



if __name__ == '__main__':
    unittest.main(verbosity=2)