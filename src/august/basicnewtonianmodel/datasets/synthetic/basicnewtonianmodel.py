import numpy as np
from lib.nhpp import *


class BasicNewtonianModel:

    def __init__(self, x0, v0, maxTime, gamma, seed=0):

        # Initialize the models parameters
        self.__t0 = 0
        self.__x0 = np.asarray(x0, dtype=np.float)
        self.__v0 = np.asarray(v0, dtype=np.float)
        self.__maxTime = maxTime

        # Get the number of nodes and dimension size
        self.__numOfNodes = self.__x0[0].shape[0]
        self.__dim = self.__x0.shape[1]

        # Set the node specific biases
        self.__gamma = gamma

        # Set the seed value
        self.__seed = seed

        # Get the node indices
        self._nodePairs = np.triu_indices(n=self.__numOfNodes, k=1)

        # Set the seed value
        np.random.seed(self.__seed)

    def getPosition(self, i, t):

        return self.__x0[i, :] + ( self.__v0[i, :] * t )

    def getVelocity(self, i, t):

        return self.__v0[i, :] * t

    def __getDistanceBetween(self, i, j, t, order=2):

        xi = self.getPosition(i=i, t=t)
        xj = self.getPosition(i=j, t=t)

        dx = xi - xj

        return np.sqrt(np.dot(dx , dx))

    # Find the critical points
    def __findCriticalPoints(self, i, j):
        # Assumption: Euclidean distance

        # Get the differences
        deltaX = self.__x0[i, :] - self.__x0[j, :]
        deltaV = self.__v0[i, :] - self.__v0[j, :]

        # Add the initial time point
        criticalPoints = [0]

        # For the model containing only position and velocity
        # Find the point in which the derivative equal to 0
        t = - np.dot(deltaX, deltaV) / np.dot(deltaV, deltaV)
        criticalPoints.append(t)

        # Add the last time point
        criticalPoints.append(self.__maxTime)

        return criticalPoints

    def __computeIntensityForPair(self, i, j, t):

        return np.exp( self.__gamma[i] + self.__gamma[j] - self.__getDistanceBetween(i=i, j=j, t=t) )

    def sampleEventsForAllNodePairs(self):

        networkEvents = [[[] for _ in range(i, self.__numOfNodes)] for i in range(self.__numOfNodes)]

        for i, j in zip(self._nodePairs[0], self._nodePairs[1]):
            # Define the intensity function for each node pair (i,j)
            intensityFunc = lambda t: self.__computeIntensityForPair(i=i, j=j, t=t)
            # Get the critical points
            criticalPoints = self.__findCriticalPoints(i=i, j=j)
            # Simulate the models
            nhppij = NHPP(maxTime=self.__maxTime, intensityFunc=intensityFunc, timeBins=criticalPoints, seed=self.__seed)
            eventTimes = nhppij.simulate()
            # Add the event times
            networkEvents[i][j].extend(eventTimes)

        return networkEvents