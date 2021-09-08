import numpy as np

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
        self.__maxTime = T
        self.__gamma = gamma
        self.__seed = seed

        self.__node_pair_indices = np.triu_indices(self.__x.shape[0], k=1)
        np.random.seed(seed)

    def __get_position(self, i, t):
        return self.__x[i, :] + self.__v[i, :] * t

    def __get_velocity(self, i, t):
        return self.__v[i, :] * t

    def __calculate_distance(self, i, j, t):
        '''
        Calculates the Eucledian distance between node i and j at time t
        '''
        xi, xj = self.__get_position(i, t), self.__get_position(j, t)

        deltaX = xi - xj

        # Euclediean distance
        return np.sqrt(np.dot(deltaX, deltaX))

    def sample_interaction_times_for_all_node_pairs(self):