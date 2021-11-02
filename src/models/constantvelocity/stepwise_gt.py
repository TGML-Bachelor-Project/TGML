import torch
import torch.nn as nn
from utils.nodes.distances import get_squared_euclidean_dist, vec_squared_euclidean_dist
from utils.integrals.analytical import vec_analytical_integral as evaluate_integral


class GTStepwiseConstantVelocityModel(nn.Module):
    '''
    Model for predicting initial conditions of a temporal dynamic graph network.
    The model predicts starting postion z0, starting velocities v0, and starting background node intensity beta
    using a Euclidean distance measure in latent space for the intensity function.
    '''
    def __init__(self, n_points:int, z, v, beta, steps, max_time, device):
            '''
            :param n_points:                Number of nodes in the temporal dynamics graph network
            :param intensity_func:          The intensity function of the model
            :param integral_approximator:   The function used to approximate the non-event intensity integral
            '''
            super().__init__()
    
            self.device = device
            self.beta = beta
            self.z0 = z
            self.v0 = v
    
            self.num_of_nodes = n_points
            self.n_node_pairs = n_points*(n_points-1) // 2
            self.node_pair_idxs = torch.triu_indices(row=self.num_of_nodes, col=self.num_of_nodes, offset=1)

            # Creating the time step deltas
            #Equally distributed
            time_intervals = torch.linspace(0, max_time, steps+1)
            shifted_time_intervals = time_intervals[:-1]
            time_intervals = time_intervals[1:]
            self.time_deltas = time_intervals-shifted_time_intervals
            # All deltas should be equal do to linspace, so we can take the first
            self.time_delta_size = self.time_deltas[0].item()


    def steps(self, times:torch.Tensor) -> torch.Tensor:
        '''
        Increments the model's time by t by
        updating the latent node position vector z
        based on a constant velocity dynamic.

        :param t:   The time to update the latent position vector z with

        :returns:   The updated latent position vector z
        '''
        Z_steps = self.z0.unsqueeze(2) + self.v0*self.time_deltas
        Z_steps = torch.cat((self.z0.unsqueeze(2), Z_steps), dim=2)
        time_step_values = times/self.time_delta_size
        time_step_floored = torch.floor(time_step_values)
        time_step_delta_difs = (times-time_step_floored*self.time_delta_size)[:-1] #Don't take last delta, as we do not step from last
        time_step_indices = time_step_floored.tolist()[:-1] #Don't take last step
        Z_step_starting_positions = Z_steps[:,:,time_step_indices]
        Zt = Z_step_starting_positions + self.v0[:,:,time_step_indices]*time_step_delta_difs
        
        #We don't use first and last Z0 because first is always z0 and not zt0 and last Z_steps is not a starting step
        return Zt, Z_steps[1:,:,:-1] 

    def step(self, t:torch.Tensor) -> torch.Tensor:
        '''
        Increments the model's time by t by
        updating the latent node position vector z
        based on a constant velocity dynamic.

        :param t:   The time to update the latent position vector z with

        :returns:   The updated latent position vector z
        '''
        Z_steps = self.z0.unsqueeze(2) + self.v0*self.time_deltas
        time_step_value = torch.tensor([t/self.time_delta_size])
        time_step_floored = torch.floor(time_step_value)
        time_step_delta_dif = t-(time_step_floored*self.time_delta_size)
        time_step_index = time_step_floored.tolist()
        Z_step_starting_positions = Z_steps[:,:,time_step_index]
        Zt = Z_step_starting_positions + self.v0[:,:,time_step_index]*time_step_delta_dif
        return Zt

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

    def vec_log_intensity_function(self, times:torch.Tensor):
        '''
        The log version of the  model intensity function between node i and j at time t.
        The intensity function measures the likelihood of node i and j
        interacting at time t using a common bias term beta


        :param t:   The time to update the latent position vector z with

        :returns:   The log of the intensity between i and j at time t as a measure of
                    the two nodes' log-likelihood of interacting.
        '''
        Zt, Z0 = self.steps(times)
        d = vec_squared_euclidean_dist(Zt)
        #Only take upper triangular part, since the distance matrix is symmetric and exclude node distance to same node
        return Z0, self.beta - d


    def forward(self, data:torch.Tensor, t0:torch.Tensor, tn:torch.Tensor) -> torch.Tensor:
        '''
        Standard torch method for training of the model.

        :param data:    Node pair interaction data with columns [node_i, node_j, time_point]
        :param t0:      Start of the interaction period
        :param tn:      End of the interaction period

        :returns:       Log liklihood of the model based on the given data
        '''
        Z0, log_intensities = self.vec_log_intensity_function(times=data[:,2])
        event_intensity = torch.sum(torch.sum(log_intensities, dim=2).triu(diagonal=1))
        all_integrals = evaluate_integral(t0, tn, 
                                    z0=Z0, v0=self.v0, 
                                    beta=self.beta, device=self.device)
        #Sum over time dimension, dim 2, and then sum upper triangular
        integral = torch.sum(torch.sum(all_integrals,dim=2).triu(diagonal=1))
        non_event_intensity = torch.sum(integral)

        # Log likelihood
        return event_intensity - non_event_intensity