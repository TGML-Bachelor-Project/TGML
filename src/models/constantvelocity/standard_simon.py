import numpy as np
import torch
import torch.nn as nn
from utils.nodes.distances import get_squared_euclidean_dist
from utils.integrals.analytical import analytical_integral as evaluate_integral



class SimonConstantVelocityModel(nn.Module):

    def __init__(self, n_points, init_beta):
        super().__init__()

        self.beta = nn.Parameter(torch.tensor([[init_beta]]), requires_grad=True)
        self.z0 = nn.Parameter(torch.rand(size=(n_points,2))*0.5, requires_grad=True) 
        self.v0 = nn.Parameter(torch.rand(size=(n_points,2))*0.5, requires_grad=True)

        self.n_points = n_points
        self.ind = torch.triu_indices(row=self.n_points, col=self.n_points, offset=1)
        self.pdist = nn.PairwiseDistance(p=2) # euclidean

    def step(self, t):
        self.z = self.z0[:,:] + self.v0[:,:]*t 
        return self.z

    def get_sq_dist(self, t, u, v):
        z = self.step(t)
        z_u = torch.reshape(z[u], shape=(1,2))
        z_v = torch.reshape(z[v], shape=(1,2))
        d = self.pdist(z_u, z_v)
        return d**2

    def evaluate_integral(self, i, j, t0, tn, z, v, beta):
        a = z[i,0] - z[j,0]
        b = z[i,1] - z[j,1]
        m = v[i,0] - v[j,0]
        n = v[i,1] - v[j,1]
        return -torch.sqrt(torch.pi)*torch.exp(((-b**2 + beta)*m**2 + 2*a*b*m*n - n**2*(a**2 - beta))/(m**2 + n**2))*(torch.erf(((m**2 + n**2)*t0 + a*m + b*n)/torch.sqrt(m**2 + n**2)) - torch.erf(((m**2 + n**2)*tn + a*m + b*n)/torch.sqrt(m**2 + n**2)))/(2*torch.sqrt(m**2 + n**2))

    def log_intensity_function(self, i, j, t):
        '''
        The log version of the  model intensity function between node i and j at time t.
        The intensity function measures the likelihood of node i and j
        interacting at time t using a common bias term beta

        :param i:   Index of node i
        :param j:   Index of node j
        :param t:   The time to update the latent position vector z with

        :returns:   The log of the intensity between i and j at time t as a measure of
                    the two nodes' log-likelihood of interacting.
        '''
        z = self.step(t)
        d = get_squared_euclidean_dist(z, i, j)
        return self.beta - d

    def forward(self, data, t0, tn):

        event_intensity = 0.
        for u, v, event_time in data:
            u, v = int(u), int(v) # cast to int for indexing
            event_intensity += self.beta - self.get_sq_dist(event_time, u, v)
        
        non_event_intensity = 0.
        for u, v in zip(self.ind[0], self.ind[1]):
            non_event_intensity += self.evaluate_integral(u, v, t0, tn, self.z0, self.v0, beta=self.beta)

        log_likelihood = event_intensity - non_event_intensity
        ratio = event_intensity / non_event_intensity

        return log_likelihood#, ratio