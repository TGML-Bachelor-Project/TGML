import unittest
from unittest.mock import MagicMock

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import kl_div
from src.data.builder import build_dataset
from src.data.synthetic.simulators.constantvelocity import ConstantVelocitySimulator

class TestBuildDataset(unittest.TestCase):
    def test_build_dataset(self):
        seed = 2
        max_time = 50
        true_beta = .001
        z0 = np.asarray([[-5, 0], [4, 0], [0, 3], [0, -2]])
        v0 = np.asarray([[0.02, 0], [-0.02, 0], [0, -0.02], [0, 0.02]])

        event_simulator = ConstantVelocitySimulator(starting_positions=z0, 
                                                    velocities=v0, 
                                                    T=max_time, 
                                                    beta=true_beta, 
                                                    seed=seed)
        event_simulator.__intensity_function = MagicMock(return_value=3)
        ## Compute events
        events = event_simulator.sample_interaction_times_for_all_node_pairs()
        
        time_column_idx=2
        data_set = build_dataset(num_of_nodes=z0.shape[0], events=events, time_column_idx=2)

        real_poisson = np.random.poisson(lam=3, size=len(data_set))

        plt.plot(data_set[:,2], label='Simulated Poisson Events')
        plt.plot(real_poisson, label='Numpy Poisson Distribution')
        plt.legend()
        plt.show()

        self.assertLess(kl_div(data_set[:,time_column_idx], real_poisson).sum(), 0.5)


if __name__ == '__main__':
    unittest.main(verbosity=2)