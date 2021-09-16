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

    def __get_position(self, i:int, t:int) -> np.ndarray:
        '''
        Calculates position of node i at time t.
        With the assumption of constant velocity of node i.

        :param i:   Index of the node to get the position of
        :param t:   The current time

        :returns:   The current position of node i based on 
                    its starting position and velocity
        '''
        return self.z0[i, :] + self.__v0[i, :] * t

    def __calculate_distance(self, i:int, j:int, t:int) -> np.float64:
        '''
        Calculates the Eucledian distance between node i and j at time t

        :param i:   Index of node i
        :param j:   Index of node j
        :param t:   The time at which the distance of node i and j
                    is calculated
        
        :returns:   The Euclidean distance of node i and j at time t
        '''
        xi, xj = self.__get_position(i, t), self.__get_position(j, t)

        # Euclediean distance
        return np.linalg.norm(xi-xj)

    def __critical_time_points(self, i:int, j:int) -> list:
        '''
        Creates a list of critical time points for the development
        of the dynamic temporal graph network i.e. points in time
        where the derivative of the intensity function for the
        two nodes in question is 0.
        The space of the nodes is assumed to be Euclidean.

        :param i:   Index of node i
        :param j:   Index of node j

        :returns:   A list of critical time points as floating point values
        '''
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

    def __intensity_function(self, i:int, j:int, t:int) -> np.float64:
        '''
        The intensity function used to calculate the event frequencies at time t in the
        simulation of the Non-homogeneous Poisson process

        :param i:   Index of node i
        :param j:   Index of node j
        :param t:   The time at which to compute the intensity function

        :returns:   The intensity between node i and j at time t i.e.
                    a measure of the likelihood of the two nodes interacting
        '''
        return np.exp(self.__beta - self.__calculate_distance(i,j,t))

    def sample_interaction_times_for_all_node_pairs(self) -> list:
        '''
        Samples interactions between nodes in a dynamic temporal graph network
        based on a Non-homogeneous Poisson Process.
        The interactions are stored in a lower triangular matrix with rows
        and columns corresponding to node indecies e.g. networkEvents[3][0]
        would be a collection of floating point numbers indicating the time points 
        of node 3 and node 0 interacting.

        :returns:   A lower triangular matrix with rows and colums being node indecies
                    and entries [i][j] being a collection of time points indicating
                    the times where node j and node i interacts.
        '''
        # Lower triangular matrix of lists
        networkEvents = [[[] for _ in range(i, self.__num_of_nodes)] for i in reversed(range(self.__num_of_nodes))]
        distinct_node_pairs = [pair for pair in 
                                zip(self.__node_pair_indices[0], self.__node_pair_indices[1])
                                if pair[0] != pair[1]]

        for i, j in distinct_node_pairs:
            # Define the intensity function for each node pair (i,j)
            intensityFunc = lambda t: self.__intensity_function(i=i, j=j, t=t)
            # Get the critical points
            criticalPoints = self.__critical_time_points(i=i, j=j)
            print(f'Critical time points {i}-{j}: {criticalPoints}')
            # Simulate the models
            nhppij = NHPP(T=self.__max_time, intensity_func=intensityFunc, time_bins=criticalPoints, seed=self.__seed)
            eventTimes = nhppij.generate_time_units()
            # Add the event times
            networkEvents[i][j].extend(eventTimes)
        
        print('Network Events Lower Triangular Matrix:')
        print(np.asarray(networkEvents))
        return networkEvents