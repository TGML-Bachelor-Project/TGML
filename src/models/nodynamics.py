import torch
import torch.nn as nn
from utils.nodes.distances import vec_squared_euclidean_dist


class NoDynamicsModel(nn.Module):
    '''
    This is a baseline model with no dynamics i.e. the velocity changes over time are removed.
    The model is only used as baseline for evaluation purposes.
    '''

    def __init__(self, n_points:int, beta:float, z0=None):
            '''
            :param n_points:    Number of nodes in the temporal dynamics graph network
            :param beta:        Common bias term
            :param z0:          An optional z0 matrix to initialize the model with specific starting positions
            '''
            super().__init__()
    
            self.beta = nn.Parameter(torch.tensor([[beta]]), requires_grad=True)
            self.z0 = z0 if z0 else nn.Parameter(torch.rand(size=(n_points,2))*0.5, requires_grad=True) 
            # Unused param, which should still be kept here to make this model compatible with the training code
            self.v0 = nn.Parameter(torch.zeros(size=(n_points,2)), requires_grad=False)

    def log_intensity_function(self, distances):
        '''
        The log version of the  model intensity function between node i and j at time t.

        :returns:   The log of the intensity between i and j as a measure of 
                    the two nodes' log-likelihood of interacting.
        '''
        return self.beta - distances

    def intensity_function(self, distances):
        '''
        The model intensity function between node i and j at time t.

        :param i:   Index of node i
        :param j:   Index of node j
        :param t:   The time to update the latent position vector z with

        :returns:   The intensity between i and j 
        '''
        return torch.exp(self.beta - distances)


    def forward(self, data:torch.Tensor, t0, tn) -> torch.Tensor:
        '''
        Standard torch method for training of the model.

        :param data:    Node pair interaction data with columns [node_i, node_j, time_point]
        :param t0:      Unused param keep to make compatible with other model trainings
        :param tn:      Unused param keep to make compatible with other model trainings

        :returns:       Log liklihood of the data based on the current model params
        '''
        i = data[:,0].long() #Long for indexing
        j = data[:,1].long()
        distances = vec_squared_euclidean_dist(self.z0)

        log_intensities = self.log_intensity_function(distances)
        event_intensity = torch.sum(log_intensities[i,j])

        intensities = self.intensity_function(distances)
        non_event_intensity = torch.sum(intensities[i,j])


        log_likelihood = event_intensity - non_event_intensity

        return -log_likelihood