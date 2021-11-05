import torch
import numpy as np
from data.synthetic.nhpp import NHPP
from utils.nodes.positions import stepwise_get_current_position as get_current_position

class StepwiseConstantVelocitySimulator:
    '''
    Model using Newtonian dynamics in the form of constant velocities
    to model node pair interactions based on Euclidean distance in a latent space.
    '''
    def __init__(self, starting_positions, velocities, max_time:int, beta:int, device='cpu', seed:int=42):
        '''
        :param starting_positions:     The 2d coordinates of each node starting position in the latent space
        :param velocities:             Velocities for each node. The velocities are constant over time
        :param T:                      The end of the time interval which the modelling runs over
        :param gamma:                  The gamma parameters used in the intensity function
        :param seed:                   The seed used to pseudo randomness of the code
        '''
        # Model parameters
        starting_positions = starting_positions.astype(np.float32)
        self.z0 = torch.tensor(starting_positions) if not isinstance(starting_positions, torch.Tensor) else starting_positions
        self.v0 = torch.tensor(velocities) if not isinstance(velocities, torch.Tensor) else velocities
        self.__beta = beta
        self.__num_of_nodes = self.z0.shape[0]

        # Calulate the step size in time
        self.__max_time = max_time
        steps = self.v0.shape[2]
        end_times = torch.linspace(0, max_time, steps+1)
        start_times = end_times[:-1]
        end_times = end_times[1:]
        self.time_deltas = end_times-start_times

        self.__node_pair_indices = np.triu_indices(n=self.__num_of_nodes, k=1)
        np.random.seed(seed)
        self.eps = torch.tensor(np.finfo(float).eps).to(device) #Adding eps to avoid devision by 0 

    def __squared_euclidean_distance(self, i:int, j:int, t:int) -> np.float64:
        '''
        Calculates the squared Eucledian distance between node i and j at time t

        :param i:   Index of node i
        :param j:   Index of node j
        :param t:   The time at which the distance of node i and j
                    is calculated
        
        :returns:   The squared Euclidean distance of node i and j at time t
        '''
        p = get_current_position(self.z0, self.v0, i, t, self.time_deltas)
        q = get_current_position(self.z0, self.v0, j, t, self.time_deltas)

        # Squared Euclidean distance
        return ((p[0]-q[0])**2 + (p[1]-q[1])**2).item()

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
        delta_z = (self.z0[i] - self.z0[j]).unsqueeze(1)
        delta_v = self.v0[i] - self.v0[j]

        # Add the initial time point
        critical_points = [0]

        # For the model containing only position and velocity
        # Use division of two dot products to find the point in which the derivative equal to 0
        t = -torch.sum(delta_z*delta_v,dim=0) / (torch.sum(delta_v*delta_v,dim=0) + self.eps)
        t, _ = torch.sort(torch.unique(torch.abs(t)))
        critical_points.extend(t.tolist())
            # print(i, j)
            # print(t)

        # Add the last time point
        critical_points.append(self.__max_time)

        return list(set(critical_points)) #get unique critical time points

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
                            seed=np.random.randint(100000), t_start=0)
            event_times = nhppij.generate_time_units()
            # Add the event times
            network_events[i][j].extend(event_times)
        
        return network_events