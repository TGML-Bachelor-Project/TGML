import torch
import torch.nn as nn
from utils.integralapproximation import riemann_sum

#torch.pi = torch.tensor(torch.acos(torch.zeros(1)).item()*2) # not sure of this line

class BasicEuclideanDistModel(nn.Module):
    def __init__(self, n_points, init_betas, riemann_samples, node_pair_samples, non_intensity_weight=1):
        super().__init__()

        if len(init_betas) != 2:
            raise Exception('The BasicEuclideanDistModel uses two beta values for the loglikelihood')
        self.beta1 = nn.Parameter(torch.tensor([[init_betas[0]]]), requires_grad=True)
        self.beta2 = nn.Parameter(torch.tensor([[init_betas[1]]]), requires_grad=True)
        self.z0 = nn.Parameter(self.init_parameter(torch.zeros(size=(n_points,2))), requires_grad=False)
        self.v0 = nn.Parameter(self.init_parameter(torch.zeros(size=(n_points,2))), requires_grad=False)
        self.z0 = nn.Parameter(self.init_parameter(self.z0), requires_grad=True)
        self.v0 = nn.Parameter(self.init_parameter(self.v0), requires_grad=True)

        self.n_points = n_points
        self.n_node_pairs = n_points*(n_points-1) // 2

        self.ind = torch.triu_indices(row=self.n_points, col=self.n_points, offset=1)
        self.pdist = nn.PairwiseDistance(p=2) # euclidean
        self.integral_samples=riemann_samples
        self.node_pair_samples = node_pair_samples
        self.non_event_weight = non_intensity_weight

    def init_parameter(self, tensor):
        #a,b = shape, r1,r2 = range
        #torch.FloatTensor(1, 1).uniform_(-0.025, 0.025)
        #return torch.nn.init.normal_(tensor, mean=0.0, std=0.025)
        return torch.nn.init.uniform_(tensor, a=-0.025, b=0.025)

    def step(self, t):
        self.z = self.z0[:,:] + self.v0[:,:]*t + 0.5*self.a0[:,:]*t**2
        return self.z

    def get_euclidean_dist(self, t, u, v):
        z = self.step(t)
        z_u = torch.reshape(z[u], shape=(1,2))
        z_v = torch.reshape(z[v], shape=(1,2))
        d = self.pdist(z_u, z_v)
        return d

    def intensity_fun(self, t, u, v):
        z = self.step(t)
        d = self.get_euclidean_dist(t, u, v)
        return torch.exp(self.beta1  + self.beta2 - d)


    def forward(self, data, t0, tn):
        event_intensity = 0.
        non_event_intensity = 0.

        for u, v, event_time in data:
            u, v = u.long(), v.long() # cast to int for indexing
            event_intensity += self.beta1 - self.get_euclidean_dist(event_time, u, v)

        triu_samples = torch.randperm(self.n_node_pairs)[:self.node_pair_samples]
        for idx in triu_samples:
            u, v = self.ind[0][idx], self.ind[1][idx]
            non_event_intensity += riemann_sum(u, v, t0, tn, n_samples=self.integral_samples)

        log_likelihood = event_intensity - self.non_event_weight*non_event_intensity

        return log_likelihood