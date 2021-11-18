import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import numpy as np
from utils.visualize.animation import animate_nomodel
if __name__ == '__main__':

    zbase = np.asarray([[1., 1.], [1., 0.], [1., -1.], [0., -1.], [-1., -1.], [-1., 0.], [-1., 1.], [0., 1.]])
    vbase = torch.tensor([
                        [[-0.15, -0.15, 0.01], 
                        [-0.05, 0.05, -0.25]
                        ],
                        [[-0.15, -0.15, 0.05],
                        [-0.05, 0.05, 0.05] 
                        ],
                        [[-0.15, -0.15, 0.01],
                        [-0.05, 0.05, 0.25] 
                        ],
                        [[-0.15, -0.15, 0.4], 
                        [-0.05, 0.05, 0.4] 
                        ],
                        [[0.15, 0.15, -0.01],
                        [0.05, -0.05, 0.25] 
                        ],
                        [[0.15, 0.15, -0.05],
                        [0.05, -0.05, 0.05] 
                        ],
                        [[0.15, 0.15, -0.01],
                        [0.05, -0.05, -0.25] 
                        ],
                        [[0.15, 0.15, -0.4],
                        [0.05, -0.05, -0.4] 
                        ]])
    z0 = np.append(zbase, zbase*1.5, axis=0)
    v0 = np.append(vbase, vbase*1.5, axis=0)
    for i in range(2,20):
        z0 = np.append(z0, zbase*(i+0.5), axis=0)
        v0 = np.append(v0, vbase*(i+0.5), axis=0)
    z0 = torch.tensor(z0)
    v0 = torch.tensor(v0)
    max_time = 15
    true_beta = 7.5
    model_beta = 8.

    # z0_xmean = np.mean(z0.numpy()[:,0])
    # z0_ymean = np.mean(z0.numpy()[:,1])
    # z0[:,0], z0[:,1] = z0[:,0] - z0_xmean, z0[:,1] -z0_ymean
    # for i in range(len(v0[0,0])):
    #   sumx = 0
    #   sumy = 0
    #   for node in range(len(v0)):
    #     sumx += v0[node,0][i]
    #     sumy += v0[node,1][i]
    #   meanx = sumx / len(v0)
    #   meany = sumy / len(v0)
    #   for node in range(len(v0)):
    #     v0[node,0][i] = v0[node,0][i] - meanx
    #     v0[node,1][i] = v0[node,1][i] - meany

    time_intervals = torch.linspace(0, max_time, v0.shape[2] + 1)
    start_times = time_intervals[:-1]
    end_times = time_intervals[1:]
    time_intervals = list(zip(start_times.tolist(), end_times.tolist()))
    time_deltas = (end_times-start_times)
    # All deltas should be equal do to linspace, so we can take the first
    step_size = time_deltas[0]
    animate_nomodel(z0=z0, v0=v0, time_deltas=time_deltas, step_size=step_size, start_times=start_times, t_start=0, t_end=max_time, num_of_time_points=100, device=None)
