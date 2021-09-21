import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import unittest
from src.utils.integralapproximation import riemann_sum

class TestIntegralApproximation(unittest.TestCase):

    def test_zero_integral(self):
        i = torch.tensor([0])
        j = torch.tensor([1])
        t0 = torch.tensor([0])
        tn = torch.tensor([6])
        n_samples = 10
        func = lambda x_i, id1, id2: 0 #just return 0, so the reimann_sum should be 0 
        self.assertEqual(riemann_sum(i, j, t0, tn, n_samples, func), torch.tensor([0]))

    def test_given_tn_6_and_constant_func_5_expects_30(self):
        i = torch.tensor([0])
        j = torch.tensor([1])
        t0 = torch.tensor([0])
        tn = torch.tensor([6])
        n_samples = 10
        func = lambda x_i, id1, id2: 5  
        self.assertEqual(riemann_sum(i, j, t0, tn, n_samples, func), torch.tensor([30]))

    def test_given_tn_6_and_dynamic_func_expects_36(self):
        i = torch.tensor([0])
        j = torch.tensor([1])
        t0 = torch.tensor([0])
        tn = torch.tensor([6])
        n_samples = 10
        func = lambda x_i, id1, id2: 2*x_i 
        self.assertEqual(riemann_sum(i, j, t0, tn, n_samples, func), torch.tensor([36]))


if __name__ == '__main__':
    unittest.main(verbosity=2)