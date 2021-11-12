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
    def __init__(self, n_points:int, beta:float, steps, max_time, device, z0, v0, true_init, batch_size):
            '''
            :param n_points:                Number of nodes in the temporal dynamics graph network
            :param intensity_func:          The intensity function of the model
            :param integral_approximator:   The function used to approximate the non-event intensity integral
            '''
            super().__init__()
    
            self.batch_size = batch_size
            self.device = device
            self.num_of_steps = steps
            self.beta = nn.Parameter(torch.tensor([[beta]]), requires_grad=True)

            if true_init:
                z0_copy = z0.astype(np.float) if isinstance(z0, np.ndarray) else z0
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
            self.start_times = time_intervals[:-1].to(self.device)
            self.end_times = time_intervals[1:].to(self.device)
            self.time_intervals = list(zip(self.start_times.tolist(), self.end_times.tolist()))
            self.time_deltas = (self.end_times-self.start_times)
            # All deltas should be equal do to linspace, so we can take the first
            self.step_size = self.time_deltas[0]

    def old_steps(self, v0, times:torch.Tensor) -> torch.Tensor:
        '''
        Increments the model's time by t by
        updating the latent node position vector z
        based on a constant velocity dynamic.

        :param t:   The time to update the latent position vector z with

        :returns:   The updated latent position vector z
        '''
        Zt = self.z0.unsqueeze(2) + v0.unsqueeze(2) * times
        return Zt

    def steps_z0(self):
        steps_z0 = self.z0.unsqueeze(2) + torch.cumsum(self.v0*self.time_deltas, dim=2)
        # Adding the initial Z0 position as first step
        steps_z0 = torch.cat((self.z0.unsqueeze(2), steps_z0), dim=2)
        return steps_z0[:,:,:-1]

    def steps(self, times:torch.Tensor) -> torch.Tensor:
        '''
        Increments the model's time by t by
        updating the latent node position vector z
        based on a constant velocity dynamic.

        :param t:   The time to update the latent position vector z with

        :returns:   The updated latent position vector z
        '''
        '''
        #Calculate how many steps each time point corresponds to
        time_step_ratio = times/self.step_size
        #Make round down time_step_ratio to find the index of the step which each time fits into
        time_to_step_index = torch.floor(time_step_ratio)
        #Make sure times that lands on tn is put into the last time step by subtracting 1 from their step index
        time_step_indices = [ t if t < self.num_of_steps else t-1 for t in  time_to_step_index.tolist()]
        #Calculate the remainding time that will be inside the matching step for each time
        remainding_time = (times-torch.tensor(time_step_indices).to(self.device)*self.step_size)
        '''

        ## Testing new computation of zt
        step_mask = ((times.unsqueeze(1) > self.start_times) | (self.start_times == 0).unsqueeze(0))
        step_end_times = step_mask*torch.cumsum(step_mask*self.step_size, axis=1)
        time_mask = times.unsqueeze(1) <= step_end_times
        time_deltas = (self.step_size - (step_end_times - times.unsqueeze(1))*time_mask)*step_mask
        movement = torch.sum(self.v0.unsqueeze(2)*time_deltas, dim=3)
        zt = self.z0.unsqueeze(2) + movement

        #Latent Z positions for all times

        #zt = steps_z0[:,:,time_step_indices] + self.v0[:,:,time_step_indices]*remainding_time

        # We don't take the very last z0, because that is the final z positions and not the start of any new step
        return zt

    def old_log_intensity_function(self, v0, times:torch.Tensor):
        '''
        The log version of the  model intensity function between node i and j at time t.
        The intensity function measures the likelihood of node i and j
        interacting at time t using a common bias term beta


        :param t:   The time to update the latent position vector z with

        :returns:   The log of the intensity between i and j at time t as a measure of
                    the two nodes' log-likelihood of interacting.
        '''
        z = self.old_steps(v0, times)
        d = vec_squared_euclidean_dist(z)
        #Only take upper triangular part, since the distance matrix is symmetric and exclude node distance to same node
        return self.beta - d

    def log_intensity_function(self, times:torch.Tensor):
        '''
        The log version of the  model intensity function between node i and j at time t.
        The intensity function measures the likelihood of node i and j
        interacting at time t using a common bias term beta


        :param t:   The time to update the latent position vector z with

        :returns:   The log of the intensity between i and j at time t as a measure of
                    the two nodes' log-likelihood of interacting.
        '''
        Zt = self.steps(times)
        d = vec_squared_euclidean_dist(Zt)
        #Only take upper triangular part, since the distance matrix is symmetric and exclude node distance to same node
        return self.beta - d


    def forward(self, data:torch.Tensor, t0:torch.Tensor, tn:torch.Tensor) -> torch.Tensor:
        '''
        Standard torch method for training of the model.

        :param data:    Node pair interaction data with columns [node_i, node_j, time_point]
        :param t0:      Start of the interaction period
        :param tn:      End of the interaction period

        :returns:       Log liklihood of the model based on the given data
        '''
        event_intensity = torch.tensor([0.]).to(self.device)
        batch_size = self.batch_size if self.batch_size > 0 else len(data)
        batches = torch.split(data, batch_size, dim=0)
        for batch in batches:
            log_intensities = self.log_intensity_function(times=batch[:,2])
            t_index = list(range(len(batch)))
            i = torch.floor(batch[:,0]).tolist() #torch.floor to make i and j int
            j = torch.floor(batch[:,1]).tolist()
            event_intensity += torch.sum(log_intensities[i,j,t_index])
        '''
        log_intensities = self.log_intensity_function(times=data[:,2])
        t = list(range(data.size()[0]))
        i = torch.floor(data[:,0]).tolist() #torch.floor to make i and j int
        j = torch.floor(data[:,1]).tolist()

        event_intensity = torch.sum(log_intensities[i,j,t])
        '''

        all_integrals = evaluate_integral(t0, tn, 
                                    z0=self.steps_z0(), v0=self.v0, 
                                    beta=self.beta, device=self.device)
        #Sum over time dimension, dim 2, and then sum upper triangular
        integral = torch.sum(torch.sum(all_integrals,dim=2).triu(diagonal=1))
        non_event_intensity = torch.sum(integral)

        # Log likelihood
        return event_intensity - non_event_intensity
