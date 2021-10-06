import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import unittest
from unittest.mock import MagicMock
from src.utils.integralapproximation import riemann_sum, analytical_squared_euclidean
from src.models.intensityfunctions.commonbias import CommonBias

class TestIntegralApproximation(unittest.TestCase):

    def test_riemann_zero_integral(self):
        i = torch.tensor([0])
        j = torch.tensor([1])
        t0 = torch.tensor([0])
        tn = torch.tensor([6])
        n_samples = 10
        intensity_func = CommonBias()
        intensity_func.result = MagicMock(return_value=0)
        self.assertEqual(riemann_sum(i, j, t0, tn, n_samples, intensity_func), torch.tensor([0]))

    def test_riemann_given_tn_6_and_constant_func_5_expects_30(self):
        i = torch.tensor([0])
        j = torch.tensor([1])
        t0 = torch.tensor([0])
        tn = torch.tensor([6])
        n_samples = 10
        intensity_func = CommonBias()
        intensity_func.result = MagicMock(return_value=0)
        self.assertEqual(riemann_sum(i, j, t0, tn, n_samples, intensity_func), torch.tensor([30]))

    def test_riemann_given_tn_6_and_dynamic_func_expects_36(self):
        i = torch.tensor([0])
        j = torch.tensor([1])
        t0 = torch.tensor([0])
        tn = torch.tensor([6])
        n_samples = 10
        intensity_func = CommonBias()
        intensity_func.result = MagicMock(return_value=0)
        self.assertEqual(riemann_sum(i, j, t0, tn, n_samples, intensity_func), torch.tensor([36]))

    def test_analytical_zero_integral(self):
        i = torch.tensor([0])
        j = torch.tensor([1])
        t0 = torch.tensor([0])
        tn = torch.tensor([6])
        n_samples = 10
        intensity_func = CommonBias()
        intensity_func.result = MagicMock(return_value=0)
        self.assertEqual(analytical_squared_euclidean(i, j, t0, tn, n_samples, intensity_func), torch.tensor([0]))



if __name__ == '__main__':
    unittest.main(verbosity=2)