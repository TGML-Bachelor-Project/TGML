import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
from utils.visualize.animation import animate_nomodel

def __try_numpy_to_tensor(t):
    return t if isinstance(t, torch.Tensor) else torch.from_numpy(t)

def center_z0(z0:torch.Tensor):
    z0 = __try_numpy_to_tensor(z0)
    return z0 - torch.mean(z0, dim=0)

def remove_v_drift(v0:torch.Tensor):
    v0 = __try_numpy_to_tensor(v0)
    return v0 - torch.mean(v0, dim=0)

def remove_rotation(z0:torch.Tensor, v0:torch.Tensor):
    z0 = __try_numpy_to_tensor(z0)
    v0 = __try_numpy_to_tensor(v0)
    zv = torch.vstack([z0, v0.reshape(-1,2)])
    U, S, VT = torch.linalg.svd(zv, full_matrices=False)
    new_coords = U * S.unsqueeze(0)
    # z0 and v0 without rotation
    return new_coords[:z0.shape[0],:], new_coords[z0.shape[0]:,:].reshape(v0.shape)


if __name__ == '__main__':

    '''
    ## 10 steps sunny-moon-12 lr 0.001
    z0 = torch.tensor([[ 1.8849,  1.2768],
            [-0.0105,  0.0707],
            [-0.0254,  0.7653],
            [ 0.5165,  0.9590],
            [-0.1513, -0.2314],
            [ 0.5383, -0.0806],
            [ 0.3739,  0.6369],
            [-0.1463,  0.5094],
            [ 1.1075,  0.3236]])

    v0 = torch.tensor([[[ 0.0992,  0.1218,  0.4256, -0.4223,  0.3776,  0.0274,  0.7000, 0.5947,  0.0779,  0.3296],
            [-0.3057,  0.9854,  0.0038,  0.5760,  0.4719, -0.4002, -0.1191, 0.8827, -0.1514,  1.0771]],

            [[ 0.5138, -0.3079,  0.6557, -0.2403,  0.6563,  0.3019, -0.0363,
            0.5979, -0.0587,  0.3666],
            [ 0.4395,  0.3136, -0.0670,  0.2746,  0.6875, -0.1249,  0.5111,
            0.1666,  0.7089, -0.3074]],

            [[ 0.2866,  0.3817, -0.1853,  0.6293,  0.0204,  0.4280,  0.0654,
            0.3660,  0.0607,  0.5453],
            [-0.0068,  0.4027,  0.4131,  0.1589,  0.4479,  0.4020, -0.1249,
            0.6779, -0.1024,  0.5796]],

            [[ 0.0617,  0.2425,  0.2230,  0.1450,  0.5310,  0.0865,  0.4892,
            -0.0924,  0.3062,  0.3479],
            [-0.1299,  0.5932,  0.2357,  0.0510,  0.4819,  0.4129,  0.1228,
            0.2735,  0.5090,  0.0657]],

            [[ 0.2597,  0.0016,  0.5485, -0.0080,  0.3535,  0.6106,  0.1428,
            -0.1910,  0.6658, -0.0618],
            [ 0.5708,  0.0496,  0.3720,  0.2351,  0.3442,  0.3456,  0.0162,
            0.5251,  0.1954,  0.5574]],

            [[-0.1124,  0.5696, -0.0012,  0.0983,  0.7911, -0.1779,  0.5560,
            0.2751, -0.1575,  0.6741],
            [ 0.2436,  0.5838, -0.1101,  0.5616,  0.2609,  0.2771,  0.3530,
            0.2899,  0.2483,  0.1532]],

            [[ 0.1821,  0.1660,  0.2271,  0.2699,  0.1807,  0.4607,  0.1509,
            0.1551,  0.3392,  0.1295],
            [ 0.1694,  0.3025,  0.0799,  0.5428,  0.1584,  0.3170,  0.3048,
            0.3509,  0.2672,  0.2990]],

            [[ 0.3725,  0.1302,  0.0070,  0.3860,  0.4371,  0.1520,  0.3504,
            0.0692,  0.4582, -0.0814],
            [ 0.2859,  0.0656,  0.3073,  0.3204,  0.2227,  0.2866,  0.6256,
            -0.1095,  0.7032, -0.0617]],

            [[-0.1748,  0.4713,  0.1149,  0.3816, -0.1415,  0.8001, -0.1058,
            0.3830,  0.1189, -0.0469],
            [ 0.4250, -0.1339,  0.7135, -0.0787,  0.4723,  0.1095,  0.6859,
            -0.0862,  0.3294,  0.3916]]])
    '''
    z0 = torch.tensor([   
                        [0., 1.], 
                        [0., -1.]
                    ])
    v0 = torch.tensor([
                        [[0.], #Vx node 0
                        [-0.01] #Vy node 0
                        ],
                        [[0.], #Vx node 1
                        [0.01] #Vy node 1
                        ]])

    #Create you own folder called result_z0_v0 in the root folder and add z0 and v0 there
    # load_folder = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'result_z0_v0')
    # z0 = torch.load(os.path.join(load_folder,'z0.pt'))
    # v0 = torch.load(os.path.join(load_folder, 'v0.pt'))
    z0, v0 = remove_rotation(z0=center_z0(z0), v0=remove_v_drift(v0))

    time_intervals = torch.linspace(0, 40.67, v0.shape[2] + 1)
    start_times = time_intervals[:-1]
    end_times = time_intervals[1:]
    time_intervals = list(zip(start_times.tolist(), end_times.tolist()))
    time_deltas = (end_times-start_times)
    # All deltas should be equal do to linspace, so we can take the first
    step_size = time_deltas[0]
    animate_nomodel(z0=z0, v0=v0, time_deltas=time_deltas, step_size=step_size, num_of_steps=v0.shape[2], t_start=0, t_end=40.67, num_of_time_points=100, device=None)





