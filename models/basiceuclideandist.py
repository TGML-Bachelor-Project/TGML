import torch
import numpy as np
import torch.nn as nn
from utils.integralapproximation import riemann_sum


class BasicEuclideanDistModel(nn.Module):
    '''
    Model for predicting initial conditions of a temporal dynamic graph network.
    The model predicts starting postion z0, starting velocities v0, and starting background node intensity beta
    using a Euclidean distance measure in latent space for the intensity function.
    '''
    def __init__(self, n_points:int, init_beta:float, riemann_samples:int, non_intensity_weight:int=1):
        '''
        :param n_points:                Number of nodes in the temporal dynamics graph network
        :param init_beta:               Initialization value of model background intensity measure 
        :param riemann_samples:         Number of time splits to use when calculating the integral
                                        as part of the model log liklihood function
        :param non_intensity_weight:    Weight factor for the non_event part of the log likelihood function
        '''
        super().__init__()

        self.beta = nn.Parameter(torch.tensor([[init_beta]]), requires_grad=True)
        self.z0 = nn.Parameter(self.init_parameter(torch.zeros(size=(n_points,2))), requires_grad=True)
        self.v0 = nn.Parameter(self.init_parameter(torch.zeros(size=(n_points,2))), requires_grad=True)

        self.n_points = n_points
        self.n_node_pairs = n_points*(n_points-1) // 2

        self.node_pair_idxs = torch.triu_indices(row=self.n_points, col=self.n_points, offset=1)
        self.integral_samples = riemann_samples
        self.integral_approximator = riemann_sum
        self.non_event_weight = non_intensity_weight

    def init_parameter(self, tensor:torch.Tensor) -> torch.Tensor:
        '''
        Fills the input Tensor with values drawn from the uniform distribution from a to b

        :returns:   The given tensor filled with the values from the uniform distribution from a to b
        '''
        return torch.nn.init.uniform_(tensor, a=-0.5, b=0.5)

    def step(self, t:torch.Tensor) -> torch.Tensor:
        '''
        Increments the model's time by t by
        updating the latent node position vector z
        based on a constant velocity dynamic.

        :param t:   The time to update the latent position vector z with

        :returns:   The updated latent position vector z
        '''
        self.z = self.z0[:,:] + self.v0[:,:]*t
        return self.z

    def get_euclidean_dist(self, t:torch.Tensor, i:torch.Tensor, j:torch.Tensor) -> torch.Tensor:
        '''
        Calculates the Euclidean distance between node i and j at time t

        :param t:   The time at which to calculate distance between i and j
                    The time point is a floating point number
        :param i:   Index of node i
        :param j:   Index of node j

        :returns:   The Euclidean distance between node i and j at time t
        '''
        z = self.step(t)
        z_i = torch.reshape(z[i], shape=(1,2))
        z_j = torch.reshape(z[j], shape=(1,2))

        # Euclediean distance
        return torch.cdist(z_i, z_j, p=2)

    def intensity_fun(self, t:torch.Tensor, i:torch.Tensor, j:torch.Tensor) -> torch.Tensor:
        '''
        The model intensity function between node i and j at time t.
        The intensity function measures the likelihood of node i and j
        interacting at time t

        :param t:   The time for which the intensity between i and j is computed
                    The time point is a floating point number.
        :param i:   Index of node i
        :param j:   Index of node j

        :returns:   The intensity between i and j at time t as a measure of
                    the two nodes' likelihood of interacting.
        '''
        d = self.get_euclidean_dist(t, i, j)
        return torch.exp(self.beta - d)

    def forward(self, data:torch.Tensor, t0:torch.Tensor, tn:torch.Tensor) -> torch.Tensor:
        '''
        Standard torch method for training of the model.

        :param data:    Node pair interaction data with columns [node_i, node_j, time_point]
        :param t0:      Start of the interaction period
        :param tn:      End of the interaction period

        :returns:       Log liklihood of the model based on the given data
        '''
        event_intensity = 0.
        non_event_intensity = 0.
        for u, v, event_time in data:
            u, v = u.long(), v.long() # cast to int for indexing
            event_intensity += self.beta - self.get_euclidean_dist(event_time, u, v)

        for u, v in zip(self.node_pair_idxs[0], self.node_pair_idxs[1]):
            non_event_intensity += self.integral_approximator(u, v, t0, tn, self.integral_samples, func=self.intensity_fun)

        log_likelihood = event_intensity - self.non_event_weight*non_event_intensity

        return log_likelihood