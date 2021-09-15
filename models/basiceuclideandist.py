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
    def __init__(self, n_points, init_beta, riemann_samples, non_intensity_weight=1):
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
        self.integral_samples=riemann_samples
        self.non_event_weight = non_intensity_weight

    def init_parameter(self, tensor):
        return torch.nn.init.uniform_(tensor, a=-0.5, b=0.5)

    def step(self, t):
        self.z = self.z0[:,:] + self.v0[:,:]*t
        return self.z

    def get_euclidean_dist(self, t, u, v):
        z = self.step(t)
        z_u = torch.reshape(z[u], shape=(1,2))
        z_v = torch.reshape(z[v], shape=(1,2))

        # Euclediean distance
        return torch.cdist(z_u, z_v, p=2)

    def intensity_fun(self, t, u, v):
        d = self.get_euclidean_dist(t, u, v)
        return torch.exp(self.beta - d)

    def forward(self, data, t0, tn):
        event_intensity = 0.
        non_event_intensity = 0.
        for u, v, event_time in data:
            u, v = u.long(), v.long() # cast to int for indexing
            event_intensity += self.beta - self.get_euclidean_dist(event_time, u, v)

        for u, v in zip(self.node_pair_idxs[0], self.node_pair_idxs[1]):
            non_event_intensity += riemann_sum(u, v, t0, tn, self.integral_samples, func=self.intensity_fun)

        log_likelihood = event_intensity - self.non_event_weight*non_event_intensity

        return log_likelihood