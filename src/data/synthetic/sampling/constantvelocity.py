import numpy as np
from data.synthetic.nhpp_starttime_zero import NHPP
from utils.nodes.positions import get_current_position

class ConstantVelocitySimulator:
    '''
    Model using Newtonian dynamics in the form of constant velocities
    to model node pair interactions based on Euclidean distance in a latent space.
    '''
    def __init__(self, starting_positions:list, velocities:list, T:int, beta:list, seed:int=42, t_start=0):
        '''
        :param starting_positions:     The 2d coordinates of each node starting position in the latent space
        :param velocities:             Velocities for each node. The velocities are constant over time
        :param T:                      The end of the time interval which the modelling runs over
        :param gamma:                  The gamma parameters used in the intensity function
        :param seed:                   The seed used to pseudo randomness of the code
        '''
        # Model parameters
        self.z0 = np.asarray(starting_positions)
        self.v0 = np.asarray(velocities)
        self.__t_start = t_start
        self.__max_time = T
        self.__beta = beta
        self.__num_of_nodes = self.z0.shape[0]

        self.__node_pair_indices = np.triu_indices(n=self.__num_of_nodes, k=1)
        np.random.seed(seed)
        self.eps = np.finfo(float).eps

    def __squared_euclidean_distance(self, i:int, j:int, t:int) -> np.float64:
        '''
        Calculates the squared Eucledian distance between node i and j at time t

        :param i:   Index of node i
        :param j:   Index of node j
        :param t:   The time at which the distance of node i and j
                    is calculated
        
        :returns:   The squared Euclidean distance of node i and j at time t
        '''
        p, q = get_current_position(self.z0, self.v0, i, t), get_current_position(self.z0, self.v0, j, t)

        # Squared Euclidean distance
        return (p[0]-q[0])**2 + (p[1]-q[1])**2

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
        deltaZ = self.z0[i, :] - self.z0[j, :]
        deltaV = self.v0[i, :] - self.v0[j, :]

        # Add the initial time point
        criticalPoints = [self.__t_start]

        # For the model containing only position and velocity
        # Find the point in which the derivative equal to 0
        t = - np.dot(deltaZ, deltaV) / (np.dot(deltaV, deltaV) + self.eps)
        if self.__t_start <= t <= self.__max_time:
            criticalPoints.append(t)
            # print(i, j)
            # print(t)

        # Add the last time point
        criticalPoints.append(self.__max_time)

        return criticalPoints

    def get_end_positions(self):
        return self.z0 + self.v0*(self.__max_time - self.__t_start)

    def intensity_function(self, i:int, j:int, t:float) -> np.float64:
        '''
        The intensity function used to calculate the event frequencies at time t in the
        simulation of the Non-homogeneous Poisson process

        :param i:   Index of node i
        :param j:   Index of node j
        :param t:   The time at which to compute the intensity function

        :returns:   The intensity between node i and j at time t i.e.
                    a measure of the likelihood of the two nodes interacting
        '''
        log_intensity = self.__beta - self.__squared_euclidean_distance(i,j,t)
        if log_intensity > -700:
            return np.exp(log_intensity)
        else:
            return 0. + self.eps

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
        # Upper triangular matrix of lists
        network_events = [[[] for _ in range(self.__num_of_nodes)] for _ in range(self.__num_of_nodes)]

        for i, j in zip(self.__node_pair_indices[0], self.__node_pair_indices[1]):
            print("Generating data for node", i,j)
            # Define the intensity function for each node pair (i,j)
            intensity_func = lambda t: self.intensity_function(i=i, j=j, t=t)
            # Get the critical points
            critical_points = self.__critical_time_points(i=i, j=j)
            # Simulate the models
            nhppij = NHPP(max_time=self.__max_time, intensity_func=intensity_func, time_bins=critical_points, 
                            seed=np.random.randint(100000), t_start=self.__t_start)
            event_times = nhppij.generate_time_units()
            # Add the event times
            network_events[i][j].extend(event_times)

        return network_events