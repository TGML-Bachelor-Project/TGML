import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import torch
import unittest
import numpy as np
from utils.nodes.distances import get_squared_euclidean_dist
from src.utils.integrals.riemann import riemann_sum
from src.utils.integrals.analytical import analytical_integral
torch.pi = torch.tensor(torch.acos(torch.zeros(1)).item()*2)

class TestIntegralApproximation(unittest.TestCase):

    def test_riemann_zero_integral(self):
        i = 0
        j = 1
        t0 = 0
        tn = 6
        n_samples = 10
        z = lambda t: 0
        f = lambda z, i, j: 0
        self.assertEqual(riemann_sum(t0, tn, n_samples, z, i=None, j=None, func=f), torch.tensor([0]))

    def test_riemann_given_tn_6_and_constant_func_5_expects_30(self):
        i = 0
        j = 1
        t0 = 0
        tn = 6
        n_samples = 10
        z = lambda t: 0
        f = lambda z, i, j: 5
        self.assertEqual(riemann_sum(t0, tn, n_samples, z=z, i=None, j=None, func=f), torch.tensor([30]))

    def test_riemann_given_tn_6_and_dynamic_func_expects_36(self):
        i = 0
        j = 1
        t0 = 0
        tn = 6
        n_samples = 10
        z = lambda t: 2*t
        f = lambda z, i, j: z
        self.assertEqual(riemann_sum(t0, tn, n_samples, z, i, j, func=f), torch.tensor([36]))

    def test_analytical_positive_positions_velocity_integral(self):
        i = 0
        j = 1
        t0 = 0
        tn = 6
        z = torch.Tensor(np.asarray([[1,1], [2,2]]))
        v = torch.Tensor(np.asarray([[1,0], [0,1]]))
        self.assertAlmostEqual(analytical_integral(t0, tn, z, v, i, j, beta=1.).item(), torch.tensor([0.8915]).item(), places=2)

    def test_analytical_negative_positions_integral(self):
        i = 0
        j = 1
        t0 = 0
        tn = 6
        z = torch.Tensor(np.asarray([[-1,1], [-2,2]]))
        v = torch.Tensor(np.asarray([[1,0], [0,1]]))
        self.assertAlmostEqual(analytical_integral(t0, tn, z, v, i, j, beta=1.).item(), torch.tensor([0.1206]).item(), places=2)

    def test_analytical_negative_velocities_integral(self):
        i = 0
        j = 1
        t0 = 0
        tn = 6
        z = torch.Tensor(np.asarray([[1,1], [2,2]]))
        v = torch.Tensor(np.asarray([[-1,0], [0,-1]]))
        self.assertAlmostEqual(analytical_integral(t0, tn, z, v, i, j, beta=1.).item(), torch.tensor([4.725]).item(), places=2)

    def test_analytical_large_bias_integral(self):
        i = 0
        j = 1
        t0 = 0
        tn = 6
        z = torch.Tensor(np.asarray([[1,1], [2,2]]))
        v = torch.Tensor(np.asarray([[-1,0], [0,-1]]))
        self.assertAlmostEqual(analytical_integral(t0, tn, z, v, i, j, beta=10.).item(), torch.tensor([38293.58138]).item(), places=2)

    def test_analytical_zero_integral(self):
        i = 0
        j = 1
        t0 = 0
        tn = 0
        z = torch.Tensor(np.asarray([[1,1], [2,2]]))
        v = torch.Tensor(np.asarray([[-1,0], [0,-1]]))
        self.assertAlmostEqual(analytical_integral(t0, tn, z, v, i, j, beta=1.).item(), torch.tensor([0.]).item(), places=2)


if __name__ == '__main__':
    unittest.main(verbosity=2)