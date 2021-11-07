import torch
import numpy as np
import torch.nn as nn
from utils.nodes.distances import vec_squared_euclidean_dist, old_vec_squared_euclidean_dist
from utils.integrals.analytical import vec_analytical_integral as evaluate_integral


class StepwiseVectorizedConstantVelocityModel(nn.Module):
    '''
    Model for predicting initial conditions of a temporal dynamic graph network.
    The model predicts starting postion z0, starting velocities v0, and starting background node intensity beta
    using a Euclidean distance measure in latent space for the intensity function.
    '''
    def __init__(self, n_points:int, beta:float, steps, max_time, device, z0, v0, true_init):
            '''
            :param n_points:                Number of nodes in the temporal dynamics graph network
            :param intensity_func:          The intensity function of the model
            :param integral_approximator:   The function used to approximate the non-event intensity integral
            '''
            super().__init__()
    
            self.device = device
            self.num_of_steps = steps
            self.beta = nn.Parameter(torch.tensor([[beta]]), requires_grad=True)
            
            if true_init:
                z0_copy = z0
                v0_copy = v0.detach().clone()
                self.z0 = nn.Parameter(torch.tensor(z0_copy), requires_grad=True) 
                self.v0 = nn.Parameter(v0_copy, requires_grad=True) 
            else:
                self.z0 = nn.Parameter(torch.rand(size=(n_points,2))*0.5, requires_grad=True) 
                self.v0 = nn.Parameter(torch.rand(size=(n_points,2, steps))*0.5, requires_grad=True) 
            
            self.num_of_nodes = n_points
            self.node_pair_idxs = torch.triu_indices(row=self.num_of_nodes, col=self.num_of_nodes, offset=1)

            # Creating the time step deltas
            #Equally distributed
            time_intervals = torch.linspace(0, max_time, steps+1)
            shifted_time_intervals = time_intervals[:-1]
            time_intervals = time_intervals[1:]
            self.time_deltas = time_intervals-shifted_time_intervals
            # All deltas should be equal do to linspace, so we can take the first
            self.time_delta_size = self.time_deltas[0]


    def steps(self, times:torch.Tensor) -> torch.Tensor:
        '''
        Increments the model's time by t by
        updating the latent node position vector z
        based on a constant velocity dynamic.

        :param t:   The time to update the latent position vector z with

        :returns:   The updated latent position vector z
        '''
        Z_steps = self.z0.unsqueeze(2) + torch.cumsum(self.v0*self.time_deltas, dtype=torch.float64, dim=2)
        # Adding the initial Z0 position as first step
        Z_steps = torch.cat((self.z0.unsqueeze(2), Z_steps), dim=2)
        # Adds self.time_delta_size*10**(-10) to make time points directly on step time fall into the right step
        time_step_values = times/self.time_delta_size
        time_step_floored = torch.floor(time_step_values)
        time_step_delta_difs = (times-time_step_floored*self.time_delta_size)
        time_step_indices = [ t if t < self.num_of_steps else t-1 for t in  time_step_floored.tolist()]
        unique_time_steps, unique_time_indices = np.unique(time_step_indices, return_index=True)
        ts = times[unique_time_indices]
        tf = torch.cat((ts[1:], times[[-1]]), dim=0)
        Z_step_starting_positions = Z_steps[:,:,time_step_indices]
        Zt = Z_step_starting_positions + self.v0[:,:,time_step_indices]*time_step_delta_difs

        #We don't use first and last Z0 because first is always z0 and not zt0 and last Z_steps is not a starting step
        return Zt, Z_steps[:,:,unique_time_steps], self.v0[:,:,unique_time_steps], ts, tf


    def log_intensity_function(self, times:torch.Tensor):
        '''
        The log version of the  model intensity function between node i and j at time t.
        The intensity function measures the likelihood of node i and j
        interacting at time t using a common bias term beta


        :param t:   The time to update the latent position vector z with

        :returns:   The log of the intensity between i and j at time t as a measure of
                    the two nodes' log-likelihood of interacting.
        '''
        Zt, Z0, V0, ts, tf = self.steps(times)
        d = vec_squared_euclidean_dist(Zt)
        #Only take upper triangular part, since the distance matrix is symmetric and exclude node distance to same node
        return Z0, V0, ts, tf, (self.beta - d)


    def forward(self, data:torch.Tensor, t0:torch.Tensor, tn:torch.Tensor) -> torch.Tensor:
        '''
        Standard torch method for training of the model.

        :param data:    Node pair interaction data with columns [node_i, node_j, time_point]
        :param t0:      Start of the interaction period
        :param tn:      End of the interaction period

        :returns:       Log liklihood of the model based on the given data
        '''
        Z0, V0, ts, tf, log_intensities = self.log_intensity_function(times=data[:,2])
        event_intensity = torch.sum(torch.sum(log_intensities, dim=2).triu(diagonal=1))
        all_integrals = evaluate_integral(ts, tf, 
                                    z0=Z0, v0=V0, 
                                    beta=self.beta, device=self.device)
        #Sum over time dimension, dim 2, and then sum upper triangular
        integral = torch.sum(torch.sum(all_integrals,dim=2).triu(diagonal=1))
        non_event_intensity = torch.sum(integral)

        # Log likelihood
        return event_intensity - non_event_intensity

        '''
        # First step
        times0 = data[:,2][(0 <= data[:,2]) & (data[:,2] <= 8)]
        log_intensities0 = torch.sum(torch.sum(self.old_log_intensity_function(self.v0[:,:,0],times0), dim=2).triu(diagonal=1))
        a = torch.sum(evaluate_integral(0, 8,z0=self.z0, v0=self.v0[:,:,0],
                                     beta=self.beta, device=self.device).triu(diagonal=1))
        
        #Second step
        times1 = data[:,2][(8 < data[:,2]) & (data[:,2] <= 16)]
        log_intensities1 = torch.sum(torch.sum(self.old_log_intensity_function(self.v0[:,:,1],times1), dim=2).triu(diagonal=1))
        z0b = self.z0 + self.v0[:,:,0]*8
        b = torch.sum(evaluate_integral(8, 16,z0=z0b, v0=self.v0[:,:,1],
                                     beta=self.beta, device=self.device).triu(diagonal=1))

        #Third step
        times2 = data[:,2][(16 < data[:,2]) & (data[:,2] <= 24)]
        log_intensities2 = torch.sum(torch.sum(self.old_log_intensity_function(self.v0[:,:,2],times2), dim=2).triu(diagonal=1))
        z0c = self.z0 + self.v0[:,:,0]*8 + self.v0[:,:,1]*8
        c = torch.sum(evaluate_integral(16, 24,z0=z0c, v0=self.v0[:,:,2],
                                     beta=self.beta, device=self.device).triu(diagonal=1))
        
        #Fourth step
        times3 = data[:,2][(24 < data[:,2]) & (data[:,2] <= 32)]
        log_intensities3 = torch.sum(torch.sum(self.old_log_intensity_function(self.v0[:,:,3],times3), dim=2).triu(diagonal=1))
        z0d = self.z0 + self.v0[:,:,0]*8 + self.v0[:,:,1]*8 + self.v0[:,:,2]*8
        d = torch.sum(evaluate_integral(24, 32,z0=z0d, v0=self.v0[:,:,3],
                                     beta=self.beta, device=self.device).triu(diagonal=1))

        #Fith step
        times4 = data[:,2][(32 < data[:,2]) & (data[:,2] <= 40)]
        log_intensities4 = torch.sum(torch.sum(self.old_log_intensity_function(self.v0[:,:,4],times4), dim=2).triu(diagonal=1))
        z0e = self.z0 + self.v0[:,:,0]*8 + self.v0[:,:,1]*8 + self.v0[:,:,2]*8 + self.v0[:,:,3]*8
        e = torch.sum(evaluate_integral(32, 40,z0=z0e, v0=self.v0[:,:,4],
                                     beta=self.beta, device=self.device).triu(diagonal=1))
        
        event_intensity = log_intensities0+log_intensities1+log_intensities2+log_intensities3+log_intensities4
        non_event_intensity = a + b + c + d + e

        # Log likelihood
        return event_intensity - non_event_intensity
        '''