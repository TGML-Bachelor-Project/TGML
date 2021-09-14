import numpy as np
from data.synthetic.distributions.nhpp import NHPP

class ConstantVelocitySimulator:
    '''
    Model using Newtonian dynamics in the form of constant velocities
    to model node pair interactions based on Euclidean distance in a latent space.
    '''
    def __init__(self, starting_positions:list, velocities:list, T:int, beta:list, seed:int=42):
        '''
        :param starting_positions:     The 2d coordinates of each node starting position in the latent space
        :param velocities:             Velocities for each node. The velocities are constant over time
        :param T:                      The end of the time interval which the modelling runs over
        :param gamma:                  The gamma parameters used in the intensity function
        :param seed:                   The seed used to pseudo randomness of the code
        '''
        # Model parameters
        self.z0 = np.asarray(starting_positions)
        self.__v0 = np.asarray(velocities)
        self.__max_time = T
        self.__beta = beta
        self.__seed = seed
        self.__num_of_nodes = self.z0.shape[0]

        self.__node_pair_indices = np.tril_indices(n=self.__num_of_nodes)
        np.random.seed(seed)

    def __get_position(self, i, t):
        return self.z0[i, :] + self.__v0[i, :] * t

    def __calculate_distance(self, i, j, t):
        '''
        Calculates the Eucledian distance between node i and j at time t
        '''
        xi, xj = self.__get_position(i, t), self.__get_position(j, t)

        deltaX = xi - xj

        # Euclediean distance
        return np.sqrt(np.dot(deltaX, deltaX))

    def __critical_time_points(self, i, j):
        # Assumption: Euclidean distance

        # Get the differences
        deltaX = self.z0[i, :] - self.z0[j, :]
        deltaV = self.__v0[i, :] - self.__v0[j, :]

        # Add the initial time point
        criticalPoints = [0]

        # For the model containing only position and velocity
        # Find the point in which the derivative equal to 0
        t = - np.dot(deltaX, deltaV) / np.dot(deltaV, deltaV)
        criticalPoints.append(t)

        # Add the last time point
        criticalPoints.append(self.__max_time)

        return criticalPoints

    def __intensity_function(self, i, j, t):
        '''
        The intensity function used to calculate the event frequencies at time t in the
        simulation of the Non-homogeneous Poisson process
        '''
        return np.exp(self.__beta[i] - self.__calculate_distance(i,j,t))

    def sample_interaction_times_for_all_node_pairs(self):
        # Lower triangular matrix of lists
        networkEvents = [[[] for _ in range(i, self.__num_of_nodes)] for i in reversed(range(self.__num_of_nodes))]

        for i, j in zip(self.__node_pair_indices[0], self.__node_pair_indices[1]):
            # Define the intensity function for each node pair (i,j)
            intensityFunc = lambda t: self.__intensity_function(i=i, j=j, t=t)
            # Get the critical points
            criticalPoints = self.__critical_time_points(i=i, j=j)
            # Simulate the models
            nhppij = NHPP(T=self.__max_time, intensity_func=intensityFunc, time_bins=criticalPoints, seed=self.__seed)
            eventTimes = nhppij.generate_time_units()
            # Add the event times
            networkEvents[i][j].extend(eventTimes)
        
        return networkEvents