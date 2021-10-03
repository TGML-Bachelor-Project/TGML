import torch
import torch.nn as nn
from utils.nodes import get_squared_euclidean_dist

class CommonBias(nn.Module):
    '''
    The model intensity function between node i and j at time t.
    The intensity function measures the likelihood of node i and j
    interacting at time t using a common bias term beta
    '''
    def __init__(self, beta:float):
        '''
        :param beta:    The value of the common bias term to use 
        '''
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([[beta]]), requires_grad=True)

    def result(self, z:torch.Tensor, i:torch.Tensor, j:torch.Tensor) -> torch.Tensor:
        '''
        The model intensity function between node i and j at time t.
        The intensity function measures the likelihood of node i and j
        interacting at time t using a common bias term beta

        :param z:   The latent space vector representation
        :param i:   Index of node i
        :param j:   Index of node j

        :returns:   The intensity between i and j at time t as a measure of
                    the two nodes' likelihood of interacting.
        '''
        d = get_squared_euclidean_dist(z, i, j)
        return torch.exp(self.beta - d)
    
    def log_result(self, z:torch.Tensor, i:torch.Tensor, j:torch.Tensor):
        '''
        The log version of the  model intensity function between node i and j at time t.
        The intensity function measures the likelihood of node i and j
        interacting at time t using a common bias term beta

        :param z:   The latent space vector representation
        :param i:   Index of node i
        :param j:   Index of node j

        :returns:   The log of the intensity between i and j at time t as a measure of
                    the two nodes' log-likelihood of interacting.
        '''
        d = get_squared_euclidean_dist(z, i, j)
        return self.beta - d