import torch
import torch.nn as nn
from utils.nodes.distances import get_squared_euclidean_dist
from utils.integrals.analytical import analytical_integral as evaluate_integral


class ConstantVelocityModel(nn.Module):
    '''
    Model for predicting initial conditions of a temporal dynamic graph network.
    The model predicts starting postion z0, starting velocities v0, and starting background node intensity beta
    using a Euclidean distance measure in latent space for the intensity function.
    '''
    def __init__(self, n_points:int, beta:float):
            '''
            :param n_points:                Number of nodes in the temporal dynamics graph network
            :param intensity_func:          The intensity function of the model
            :param integral_approximator:   The function used to approximate the non-event intensity integral
            '''
            super().__init__()
    
            self.beta = nn.Parameter(torch.tensor([[beta]]), requires_grad=True)
            self.z0 = nn.Parameter(torch.rand(size=(n_points,2))*0.5, requires_grad=True) 
            self.v0 = nn.Parameter(torch.rand(size=(n_points,2))*0.5, requires_grad=True) 
    
            self.num_of_nodes = n_points
            self.n_node_pairs = n_points*(n_points-1) // 2
            self.node_pair_idxs = torch.triu_indices(row=self.num_of_nodes, col=self.num_of_nodes, offset=1)


    def step(self, t:torch.Tensor) -> torch.Tensor:
        '''
        Increments the model's time by t by
        updating the latent node position vector z
        based on a constant velocity dynamic.

        :param t:   The time to update the latent position vector z with

        :returns:   The updated latent position vector z
        '''
        zt = self.z0[:,:] + self.v0[:,:]*t
        return zt

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


    ## Change 1 function
    def evaluate_integral(self, i, j, t0, tn, z, v, beta):
        a = z[i,0] - z[j,0]
        b = z[i,1] - z[j,1]
        m = v[i,0] - v[j,0]
        n = v[i,1] - v[j,1]
        return -torch.sqrt(torch.pi)*torch.exp(((-b**2 + beta)*m**2 + 2*a*b*m*n - n**2*(a**2 - beta))/(m**2 + n**2))*(torch.erf(((m**2 + n**2)*t0 + a*m + b*n)/torch.sqrt(m**2 + n**2)) - torch.erf(((m**2 + n**2)*tn + a*m + b*n)/torch.sqrt(m**2 + n**2)))/(2*torch.sqrt(m**2 + n**2))


    def forward(self, data:torch.Tensor, t0:torch.Tensor, tn:torch.Tensor) -> torch.Tensor:
        '''
        Standard torch method for training of the model.

        :param data:    Node pair interaction data with columns [node_i, node_j, time_point]
        :param t0:      Start of the interaction period
        :param tn:      End of the interaction period

        :returns:       Log liklihood of the model based on the given data
        '''
        event_intensity = 0.
        for i, j, event_time in data:
            i, j = int(i), int(j) # cast to int for indexing
            event_intensity += self.log_intensity_function(i, j, event_time)


        non_event_intensity = 0.
        for i, j in zip(self.node_pair_idxs[0], self.node_pair_idxs[1]):

            ## Change 1
            # non_event_intensity += evaluate_integral(t0=t0, tn=tn, 
            #                                 z=self.z0, v=self.v0, 
            #                                 i=i, j=j, 
            #                                 beta=self.beta)
            non_event_intensity += self.evaluate_integral(i=i, j=j, t0=t0, tn=tn, z=self.z0, v=self.v0, beta=self.beta)
            ## Change 1

        log_likelihood = event_intensity - non_event_intensity
        ratio = event_intensity / non_event_intensity

        return log_likelihood#, ratio