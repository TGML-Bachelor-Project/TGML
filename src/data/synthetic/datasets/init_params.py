import numpy as np
import torch



def get_initial_parameters(dataset_number, vectorized):

    steps = 1
    # if dataset_number == 1:
    #     z0 = np.asarray([[-0.6, 0.], [0.6, 0]])
    #     if vectorized != 2:
    #         v0 = np.asarray([[0.09, 0], [-0.09, -0.1]])
    #     elif vectorized == 2:
    #         v0 = torch.tensor([
    #                         [[0.09, 0, -0.09], #Vx node 0
    #                         [0, 0, 0] #Vy node 0
    #                         ],
    #                         [[-0.09, 0, 0.09], #Vx node 1
    #                         [0, 0, 0] #Vy node 1
    #                         ]])
    if dataset_number == 1:
        z0 = np.asarray([[-1, 0.], [1, 0]])
        if vectorized == 2:
            v0 = torch.tensor([
                            [[0.1, 0, -0.1], #Vx node 0
                            [0, 0, 0] #Vy node 0
                            ],
                            [[-0.1, 0, 0.1], #Vx node 1
                            [0, 0, 0] #Vy node 1
                            ]])
        max_time = 30
        true_beta = 7.5
        model_beta = 8.
        steps = 3

    if dataset_number == 2:
        z0 = np.asarray([[-1, 0.], [1, 0], [0, 1]])
        if vectorized == 2:
            v0 = torch.tensor([
                            [[0.1, 0, -0.1], #Vx node 0
                            [0, 0, 0] #Vy node 0
                            ],
                            [[-0.1, 0, 0.1], #Vx node 1
                            [0, 0, 0] #Vy node 1
                            ],
                            [[0, 0, 0], #Vx node 1
                            [-0.1, 0, 0.1] #Vy node 1
                            ]])
        max_time = 30
        true_beta = 7.5
        model_beta = 8.
        steps = 3
            

    elif dataset_number == 7:
        max_time = 60
        true_beta = 7.5
        model_beta = 10.
        z0 = np.asarray([[-3, 0], [3, 0], [0, 3], [0, -3], [3, 3], [3, -3], [-3, -3], [-3, 3]])
        v0 = np.asarray([[0.11, 0], [-0.1, 0], [0, -0.11], [0, 0.1], [-0.11, -0.09], [0, 0.05], [0, 0], [0.051, 0]])
    
    
    elif dataset_number == 8:
        max_time = 60
        true_beta = 5.
        model_beta = 7.5
        zbase = np.asarray([[-3, 0], [3, 0], [0, 3], [0, -3], [3, 3], [3, -3], [-3, -3], [-3, 3]])
        vbase = np.asarray([[0.11, 0], [-0.1, 0], [0, -0.11], [0, 0.1], [-0.11, -0.09], [0, 0.05], [0, 0], [0.051, 0]])
        z0 = np.append(zbase, zbase*2, axis=0)
        v0 = np.append(vbase, vbase*2, axis=0)
        for i in range(3,20):
            z0 = np.append(z0, zbase*i, axis=0)
            v0 = np.append(v0, vbase*i, axis=0)


    elif dataset_number == 10:
        max_time = 10
        true_beta = 7.5
        model_beta = 10.
        steps = 1
        z0 = np.asarray([[-0.6, 0.], [0.6, 0.1], [0., 0.6], [0., -0.6]])
        if vectorized != 2:
            v0 = np.asarray([[0.09, 0.01], [-0.01, -0.01], [0.01, -0.09], [-0.01, 0.09]])
        elif vectorized == 2:
            v0 = torch.tensor([
            [
                [0.09], #Vx node 0
                [0.01] #Vy node 0
            ],
            [
                [-0.01], #Vx node 1
                [-0.01] #Vy node 1
            ],
            [
                [0.01], #Vx node 2
                [-0.09] #Vy node 2
            ],
            [
                [-0.01], #Vx node 3
                [0.09]  #Vy node 3
            ]
        ])


    elif dataset_number == 20:
        max_time = 10
        true_beta = 7.5
        model_beta = 10.
        z0 = np.asarray([[-1., 0.], [0.6, 0.1], [0., 0.6], [0., -0.6]])
        v0 = np.asarray([[0.09, 0.01], [-0.01, -0.01], [0.01, -0.09], [-0.01, 0.09]])




    return z0, v0, true_beta, model_beta, max_time, steps