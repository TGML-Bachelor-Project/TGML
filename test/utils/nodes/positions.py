import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import unittest
import numpy as np
from src.utils.nodes.positions import *

class TestIntegralApproximation(unittest.TestCase):

    def test_get_position_single_node(self):
        z = np.array([[3,0]])
        v = np.array([[1, 0]])
        i = 0 # node index in z and v
        t = 5
        self.assertListEqual(list(get_current_position(z, v, i, t)), list(np.array([8, 0])))

    def test_get_position_multiple_nodes(self):
        z = np.array([[3,0], [2, 1]])
        v = np.array([[1, 0], [2, 4]])
        i = 1 # node index in z and v
        t = 5
        self.assertListEqual(list(get_current_position(z, v, i, t)), list(np.array([12, 21])))

    def test_floating_point_velocity(self):
        z = np.array([[3,0], [2, 1]])
        v = np.array([[1.5, 0], [2, 4]])
        i = 0 # node index in z and v
        t = 5
        self.assertListEqual(list(get_current_position(z, v, i, t)), list(np.array([10.5, 0])))

if __name__ == '__main__':
    unittest.main(verbosity=2)

