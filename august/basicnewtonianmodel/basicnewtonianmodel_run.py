import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from datasets.synthetic.basicnewtonianmodel import *

# A simple example for 2 nodes
seed = 1000

# Set the initial position and velocity
x0 = np.asarray([[-3, 0], [3, 0]])
v0 = np.asarray([[1, 0], [-1, 0]])
# x0 = np.asarray([[-3, 0], [3, 0], [0, 3]])
# v0 = np.asarray([[1, 0], [-1, 0], [0, -1]])

# Get the number of nodes and dimension size
numOfNodes = x0.shape[0]
dim = x0.shape[1]

# Set the max time
maxTime = 6

# Bias values for nodes
gamma = 0.5 * np.ones(shape=(numOfNodes, ))

bnm = BasicNewtonianModel(x0=x0, v0=v0, maxTime=maxTime, gamma=gamma, seed=seed)
events = bnm.sampleEventsForAllNodePairs()

for i in range(numOfNodes):
    for j in range(i+1, numOfNodes):
        print("Events for node pair ({}-{}): {}".format(i, j, events[i][j]))