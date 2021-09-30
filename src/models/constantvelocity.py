import torch
import torch.nn as nn


class ConstantVelocityModel(nn.Module):
    '''
    Model for predicting initial conditions of a temporal dynamic graph network.
    The model predicts starting postion z0, starting velocities v0, and starting background node intensity beta
    using a Euclidean distance measure in latent space for the intensity function.
    '''
    def __init__(self, n_points:int, non_intensity_weight:int, intensity_func, integral_approximator):
            '''
            :param n_points:                Number of nodes in the temporal dynamics graph network
            :param non_intensity_weight:    Weight factor for the non_event part of the log likelihood function
            :param intensity_func:          The intensity function of the model
            :param integral_approximator:   The function used to approximate the non-event intensity integral
            '''
            super().__init__()
    
            self.z0 = nn.Parameter(self.init_vector(torch.zeros(size=(n_points,2))), requires_grad=True)
            self.v0 = nn.Parameter(self.init_vector(torch.zeros(size=(n_points,2))), requires_grad=True)
    
            self.n_points = n_points
            self.n_node_pairs = n_points*(n_points-1) // 2
            self.node_pair_idxs = torch.tril_indices(row=self.n_points, col=self.n_points, offset=-1)

            self.intensity_function = intensity_func
            self.integral_approximator = integral_approximator
            self.non_event_weight = non_intensity_weight
    
            self.pdist = nn.PairwiseDistance(p=2)

    def init_vector(self, tensor:torch.Tensor) -> torch.Tensor:
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
            u, v = int(u), int(v) # cast to int for indexing
            event_intensity += self.intensity_function.log_result(self.step(event_time), u, v)

        for u, v in zip(self.node_pair_idxs[0], self.node_pair_idxs[1]):
            non_event_intensity += self.integral_approximator(t0, tn, lambda t: self.step(t), u, v, self.intensity_function.result)

        log_likelihood = event_intensity - self.non_event_weight*non_event_intensity

        return log_likelihood