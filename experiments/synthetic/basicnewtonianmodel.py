import os, sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models.basicnewtonian import BasicNewtonianModel

from data.synthetic import nhpp
from models import basicnewtonian

if __name__ == '__main__':
    # A simple example for 2 nodes
    seed = 2

    # Set the initial position and velocity
    x0 = np.asarray([[-3, 0], [3, 0]])
    v0 = np.asarray([[1, 0], [-1, 0]])

    # Get the number of nodes and dimension size
    numOfNodes = x0.shape[0]
    dim = x0.shape[1]

    # Set the max time
    maxTime = 6

    # Bias values for nodes
    gamma = 0.5 * np.ones(shape=(numOfNodes, ))

    bnm = BasicNewtonianModel(starting_positions=x0, velocities=v0, T=maxTime, gamma=gamma, seed=seed)
    events = bnm.sample_interaction_times_for_all_node_pairs()
    dataset = []

    # Build dataset of node interactions
    for i in range(numOfNodes):
        for j in range(i+1, numOfNodes):
            print("Events for node pair ({}-{}): {}".format(i, j, events[i][j]))
        # TODO Create dataset of node interactions by [i,j,events[i][j]]

    
    # TODO Define model

    # TODO Split dataset into training and testing

    # TODO Train and evaluate model visualize training metrics with plots

    # TODO Visualize model predictions of network evolution over time

    