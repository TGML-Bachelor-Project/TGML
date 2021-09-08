import numpy as np
from data.synthetic.nhpp import NHPP

class BasicNewtonianModel:
    '''
    Model using Newtonian dynamics in the form of constant velocities
    to model node pair interactions based on Euclidean distance in a latent space.
    '''
    def __init__(self, starting_positions, velocities, T, gamma, seed=42):
        '''
        starting_positions:     The 2d coordinates of each node starting position in the latent space
        velocities:             Velocities for each node. The velocities are constant over time
        T:                      The end of the time interval which the modelling runs over
        gamma:                  The gamma parameters used in the intensity function
        seed:                   The seed used to pseudo randomness of the code
        '''
        # Model parameters
        self.__x = np.asarray(starting_positions)
        self.__v = np.asarray(velocities)
        self.__max_time = T
        self.__gamma = gamma
        self.__seed = seed
        self.__num_of_nodes = self.__x.shape[0]

        self.__node_pair_indices = np.triu_indices(n=self.__num_of_nodes, k=1)
        np.random.seed(seed)

    def __get_position(self, i, t):
        return self.__x[i, :] + self.__v[i, :] * t

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
        deltaX = self.__x[i, :] - self.__x[j, :]
        deltaV = self.__v[i, :] - self.__v[j, :]

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
        return np.exp(self.__gamma[i] + self.__gamma[j] - self.__calculate_distance(i,j,t))

    def sample_interaction_times_for_all_node_pairs(self):
        networkEvents = [[[] for _ in range(i, self.__num_of_nodes)] for i in range(self.__num_of_nodes)]

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