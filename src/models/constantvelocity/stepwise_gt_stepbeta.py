import torch
import numpy as np
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
            self.num_of_steps = steps
            self.beta = nn.Parameter(beta, requires_grad=False).to(device)
            z0_copy = z.clone().detach()
            v0_copy = v.clone().detach()
            self.z0 = nn.Parameter(z0_copy, requires_grad=False).to(device)
            self.v0 = nn.Parameter(v0_copy, requires_grad=False).to(device)
    
            self.num_of_nodes = n_points
            self.node_pair_idxs = torch.triu_indices(row=self.num_of_nodes, col=self.num_of_nodes, offset=1)

            # Creating the time step deltas
            #Equally distributed
            time_intervals = torch.linspace(0, max_time, steps+1)
            start_times = time_intervals[:-1]
            end_times = time_intervals[1:]
            self.time_deltas = (end_times-start_times).to(self.device)
            # All deltas should be equal do to linspace, so we can take the first
            self.step_size = self.time_deltas[0].to(self.device)



    def steps(self, times:torch.Tensor) -> torch.Tensor:
        '''
        Increments the model's time by t by
        updating the latent node position vector z
        based on a constant velocity dynamic.

        :param t:   The time to update the latent position vector z with

        :returns:   The updated latent position vector z
        '''
        steps_z0 = self.z0.unsqueeze(2) + torch.cumsum(self.v0*self.time_deltas, dim=2)
        # Adding the initial Z0 position as first step
        steps_z0 = torch.cat((self.z0.unsqueeze(2), steps_z0), dim=2)
        #Calculate how many steps each time point corresponds to
        time_step_ratio = times/self.step_size
        #Make round down time_step_ratio to find the index of the step which each time fits into
        time_to_step_index = torch.floor(time_step_ratio)
        #Calculate the remainding time that will be inside the matching step for each time
        remainding_time = (times-time_to_step_index*self.step_size)
        #Make sure times that lands on tn is put into the last time step by subtracting 1 from their step index
        time_step_indices = [ t if t < self.num_of_steps else t-1 for t in  time_to_step_index.tolist()]
        #The step positions we will start from for each time point and then use to find their actual position
        Z_step_starting_positions = steps_z0[:,:,time_step_indices]
        #Latent Z positions for all times
        Zt = Z_step_starting_positions + self.v0[:,:,time_step_indices]*remainding_time

        # We don't take the very last z0, because that is the final z positions and not the start of any new step
        return Zt, steps_z0[:,:,:-1], time_step_indices

    def step(self, t:torch.Tensor) -> torch.Tensor:
        '''
        Increments the model's time by t by
        updating the latent node position vector z
        based on a constant velocity dynamic.

        :param t:   The time to update the latent position vector z with

        :returns:   The updated latent position vector z
        '''
        Z_steps = self.z0.unsqueeze(2) + torch.cumsum(self.v0*self.time_deltas, dim=2)
        # Adding the initial Z0 position as first step
        Z_steps = torch.cat((self.z0.unsqueeze(2), Z_steps), dim=2)
        # Adds self.time_delta_size*10**(-10) to make time points directly on step time fall into the right step
        time_step_value = t/self.step_size
        time_step_floored = int(time_step_value)
        time_step_delta_dif = t-time_step_floored*self.step_size
        time_step_index = [time_step_floored if time_step_floored < self.num_of_steps else time_step_floored-1]
        Z_step_starting_positions = Z_steps[:,:,time_step_index]
        Zt = Z_step_starting_positions + self.v0[:,:,time_step_index]*time_step_delta_dif
        return Zt, time_step_index

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
        zt, time_step_index = self.step(t)
        d = get_squared_euclidean_dist(zt, i, j)
        return self.beta[time_step_index] - d

    def vec_log_intensity_function(self, times:torch.Tensor):
        '''
        The log version of the  model intensity function between node i and j at time t.
        The intensity function measures the likelihood of node i and j
        interacting at time t using a common bias term beta


        :param t:   The time to update the latent position vector z with

        :returns:   The log of the intensity between i and j at time t as a measure of
                    the two nodes' log-likelihood of interacting.
        '''
        Zt, steps_z0, time_step_indices = self.steps(times)
        d = vec_squared_euclidean_dist(Zt)
        #Only take upper triangular part, since the distance matrix is symmetric and exclude node distance to same node
        return steps_z0, (self.beta[time_step_indices] - d)


    def forward(self, data:torch.Tensor, t0:torch.Tensor, tn:torch.Tensor) -> torch.Tensor:
        '''
        Standard torch method for training of the model.

        :param data:    Node pair interaction data with columns [node_i, node_j, time_point]
        :param t0:      Start of the interaction period
        :param tn:      End of the interaction period

        :returns:       Log liklihood of the model based on the given data
        '''
        steps_z0, log_intensities = self.vec_log_intensity_function(times=data[:,2])
        t = list(range(data.shape[0]))
        i = torch.floor(data[:,0]).tolist() #torch.floor to make i and j int
        j = torch.floor(data[:,1]).tolist()
        event_intensity = torch.sum(log_intensities[i,j,t])
        #event_intensity = torch.sum(torch.sum(log_intensities, dim=2))
        all_integrals = evaluate_integral(t0, tn, 
                                    z0=steps_z0, v0=self.v0, 
                                    beta=self.beta, device=self.device)
        #Sum over time dimension, dim 2, and then sum upper triangular
        integral = torch.sum(torch.sum(all_integrals,dim=2).triu(diagonal=1))
        non_event_intensity = torch.sum(integral)

        # Log likelihood
        return event_intensity - non_event_intensity