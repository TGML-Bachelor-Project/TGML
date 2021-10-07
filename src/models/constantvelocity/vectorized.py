import torch
import torch.nn as nn
from utils.nodes.distances import get_squared_euclidean_dist


class ConstantVelocityModel(nn.Module):
    '''
    Model for predicting initial conditions of a temporal dynamic graph network.
    The model predicts starting postion z0, starting velocities v0, and starting background node intensity beta
    using a Euclidean distance measure in latent space for the intensity function.
    '''
    def __init__(self, n_points:int, beta:int):
            '''
            :param n_points:                Number of nodes in the temporal dynamics graph network
            :param intensity_func:          The intensity function of the model
            :param integral_approximator:   The function used to approximate the non-event intensity integral
            '''
            super().__init__()
    
            self.beta = nn.Parameter(torch.tensor([[beta]]), requires_grad=True)
            self.z0 = nn.Parameter(self.__init_vector(torch.zeros(size=(n_points,2))), requires_grad=True)
            self.v0 = nn.Parameter(self.__init_vector(torch.zeros(size=(n_points,2))), requires_grad=True)
    
            self.n_points = n_points
            self.n_node_pairs = n_points*(n_points-1) // 2
            self.node_pair_idxs = torch.tril_indices(row=self.n_points, col=self.n_points, offset=-1)


    def __init_vector(self, tensor:torch.Tensor) -> torch.Tensor:
        '''
        Fills the input Tensor with values drawn from the uniform distribution from a to b

        :returns:   The given tensor filled with the values from the uniform distribution from a to b
        '''
        return torch.nn.init.uniform_(tensor, a=-0.25, b=0.25)

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

    def intensity_function(self, i, j, t):
        '''
        The model intensity function between node i and j at time t.
        The intensity function measures the likelihood of node i and j
        interacting at time t using a common bias term beta

        :param i:   Index of node i
        :param j:   Index of node j
        :param t:   The time to update the latent position vector z with

        :returns:   The intensity between i and j at time t as a measure of
                    the two nodes' likelihood of interacting.
        '''
        z = self.step(t)
        d = get_squared_euclidean_dist(z, i, j)
        return torch.exp(self.beta - d)

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

    def forward(self, data:torch.Tensor, t0:torch.Tensor, tn:torch.Tensor) -> torch.Tensor:
        '''
        Standard torch method for training of the model.

        :param data:    Node pair interaction data with columns [node_i, node_j, time_point]
        :param t0:      Start of the interaction period
        :param tn:      End of the interaction period

        :returns:       Log liklihood of the model based on the given data
        '''
        raise Exception('Not implemented')