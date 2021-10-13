import torch
from torch.functional import tensordot
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_squared_euclidean_dist(z:torch.Tensor, i:torch.Tensor, j:torch.Tensor) -> torch.Tensor:
    '''
    Calculates the squared Euclidean distance between node i and j at time t

    :param t:   The time at which to calculate distance between i and j
                The time point is a floating point number
    :param i:   Index of node i
    :param j:   Index of node j

    :returns:   The Euclidean distance between node i and j at time t
    '''
    z_i = torch.reshape(z[i], shape=(1,2))
    z_j = torch.reshape(z[j], shape=(1,2))

    # Squared Euclediean distance
    pdist = nn.PairwiseDistance(p=2)
    return pdist(z_i, z_j)**2

def vec_squared_euclidean_dist(Z):
    ZT = torch.transpose(Z,1,2)
    dot_Z = torch.tensordot(Z, Z,    dims=  ([0], [0]))
    dot_ZZT = torch.tensordot(Z, ZT, dims=  ([0], [0])) 
    dot_ZT = torch.tensordot(ZT, ZT, dims=  ([0], [0]))
    return dot_Z - 2 * dot_ZZT + dot_ZT