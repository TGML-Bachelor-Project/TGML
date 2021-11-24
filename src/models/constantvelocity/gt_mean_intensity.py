from numpy.core.fromnumeric import diagonal
import torch
import numpy as np
import torch.nn as nn
from utils.nodes.distances import get_squared_euclidean_dist, vec_squared_euclidean_dist
from utils.integrals.analytical import vec_analytical_integral as evaluate_integral


class GTMeanIntensity(nn.Module):
    def __init__(self, n_points:int, z, v, beta, steps, max_time, device):
            '''
            :param n_points:                Number of nodes in the temporal dynamics graph network
            :param intensity_func:          The intensity function of the model
            :param integral_approximator:   The function used to approximate the non-event intensity integral
            '''
            super().__init__()
    
            self.device = device
            self.num_of_steps = steps
            self.beta = nn.Parameter(torch.tensor([[beta]]), requires_grad=False).to(self.device)
            z0_copy = torch.from_numpy(z.astype(np.float)) if isinstance(z, np.ndarray) else z
            v0_copy = v.clone().detach()
            self.z0 = nn.Parameter(z0_copy, requires_grad=False).to(self.device) 
            self.v0 = nn.Parameter(v0_copy, requires_grad=False).to(self.device)
    
            self.num_of_nodes = n_points
            self.node_pair_idxs = torch.triu_indices(row=self.num_of_nodes, col=self.num_of_nodes, offset=1)

            # Creating the time step deltas
            #Equally distributed
            time_intervals = torch.linspace(0, max_time, steps+1)
            self.start_times = time_intervals[:-1].to(self.device, dtype=torch.float32)
            self.end_times = time_intervals[1:].to(self.device, dtype=torch.float32)
            self.time_intervals = list(zip(self.start_times.tolist(), self.end_times.tolist()))
            self.time_deltas = (self.end_times-self.start_times)
            # All deltas should be equal do to linspace, so we can take the first
            self.step_size = self.time_deltas[0]



    def steps(self, times:torch.Tensor) -> torch.Tensor:
        '''
        Increments the model's time by t by
        updating the latent node position vector z
        based on a constant velocity dynamic.
        :param t:   The time to update the latent position vector z with
        :returns:   The updated latent position vector z
        '''
        step_mask = ((times.unsqueeze(1) > self.start_times) | (self.start_times == 0).unsqueeze(0))
        step_end_times = step_mask*torch.cumsum(step_mask*self.step_size, axis=1)
        time_mask = times.unsqueeze(1) <= step_end_times
        time_deltas = (self.step_size - (step_end_times - times.unsqueeze(1))*time_mask)*step_mask
           
        movement = torch.sum(self.v0.unsqueeze(2)*time_deltas, dim=3)
    
        #Latent Z positions for all times
        zt = self.z0.unsqueeze(2) + movement

        return zt

    def step(self, t:torch.Tensor) -> torch.Tensor:
        '''
        Increments the model's time by t by
        updating the latent node position vector z
        based on a constant velocity dynamic.

        :param t:   The time to update the latent position vector z with

        :returns:   The updated latent position vector z
        '''
        step_mask = ((t > self.start_times) | (self.start_times == 0).unsqueeze(0))
        step_end_times = step_mask*torch.cumsum(step_mask*self.step_size, axis=1)
        time_mask = t <= step_end_times
        time_deltas = (self.step_size - (step_end_times - t)*time_mask)*step_mask
           
        movement = torch.sum(self.v0.unsqueeze(2)*time_deltas, dim=3)
    
        #Latent Z positions for all times
        zt = self.z0.unsqueeze(2) + movement

        return zt

    def steps_z0(self):
        steps_z0 = self.z0.unsqueeze(2) + torch.cumsum(self.v0*self.time_deltas, dim=2)
        # Adding the initial Z0 position as first step
        steps_z0 = torch.cat((self.z0.unsqueeze(2), steps_z0), dim=2)
        # We don't take the very last z0, because that is the final z positions and not the start of any new step
        return steps_z0[:,:,:-1]

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
        d = vec_squared_euclidean_dist(z)
        return torch.mean(self.beta - d)

    def vec_log_intensity_function(self, times:torch.Tensor):
        '''
        The log version of the  model intensity function between node i and j at time t.
        The intensity function measures the likelihood of node i and j
        interacting at time t using a common bias term beta
        :param t:   The time to update the latent position vector z with
        :returns:   The log of the intensity between i and j at time t as a measure of
                    the two nodes' log-likelihood of interacting.
        '''
        Zt = self.steps(times)
        d = vec_squared_euclidean_dist(Zt)[self.node_pair_idxs[0,:], self.node_pair_idxs[1,:],:]

        #Only take upper triangular part, since the distance matrix is symmetric and exclude node distance to same node
        return torch.mean(self.beta - d, dim=0)


    def forward(self, data:torch.Tensor, t0:torch.Tensor, tn:torch.Tensor) -> torch.Tensor:
        '''
        Standard torch method for training of the model.

        :param data:    Node pair interaction data with columns [node_i, node_j, time_point]
        :param t0:      Start of the interaction period
        :param tn:      End of the interaction period

        :returns:       Log liklihood of the model based on the given data
        '''
        times = data[:,2].to(self.device, dtype=torch.float32)
        unique_times, unique_time_indices = torch.unique(times, return_inverse=True)
        log_intensities = self.vec_log_intensity_function(times=unique_times)

        # Have to batch event intensity sum, otherwise there are too many time points
        event_intensity = torch.tensor(0.)
        for time_batch in torch.split(unique_time_indices, 10000):
            event_intensity += torch.sum(log_intensities[time_batch])

        all_integrals = evaluate_integral(t0, tn, z0=self.steps_z0(), 
                                            v0=self.v0, beta=self.beta)
        #Sum over time dimension, dim 2, and then sum upper triangular
        integral = torch.sum(torch.sum(all_integrals,dim=2).triu(diagonal=1))
        non_event_intensity = torch.sum(integral)

        log_likelihood = event_intensity - non_event_intensity
        return -log_likelihood
