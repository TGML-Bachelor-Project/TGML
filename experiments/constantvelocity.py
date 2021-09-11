# Add necessary folders/files to path
import os, sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Imports
from models.basiceuclideandist import BasicEuclideanDistModel
from data.synthetic.simulators.constantvelocity import ConstantVelocitySimulator

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

    # Simulate events from a non-homogeneous Poisson distribution
    event_simulator = ConstantVelocitySimulator(starting_positions=x0, velocities=v0, T=maxTime, gamma=gamma, seed=seed)
    events = event_simulator.sample_interaction_times_for_all_node_pairs()

    # Build dataset of node pair interactions
    dataset = []
    for i in range(numOfNodes):
        for j in range(i+1, numOfNodes):
            nodepair_events = events[i][j]
            print("Events for node pair ({}-{}): {}".format(i, j, nodepair_events))
            for np_event in nodepair_events:
                dataset.append([i,j, np_event])

    # Make sure dataset is numpy array
    dataset = np.asarray(dataset)
    # Make sure dataset is sorted according to increasing event times in column index 2
    dataset = dataset[dataset[:, 2].argsort()]

    print(dataset)
    
    # TODO Define model
    model = BasicEuclideanDistModel()

    # Model training and evaluation