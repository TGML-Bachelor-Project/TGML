import numpy as np
import torch



def get_initial_parameters(dataset_number, vectorized):

    if dataset_number == 0:
        z0 = np.asarray([[1., 1.], [-1., -1.]])
        if vectorized == 2:
            v0 = torch.tensor([
                            [[-0.15, 0.15, -0.3, 0.3], #Vx node 0
                            [-0.15, 0.15, -0.3, 0.3] #Vy node 0
                            ],
                            [[0.15, -0.15, 0.3, -0.3], #Vx node 1
                            [0.15, -0.15, 0.3, -0.3] #Vy node 1
                            ]])
        max_time = 40
        true_beta = 7.5
        model_beta = 8. #must be floating point


    if dataset_number == 1:
        z0 = np.asarray([[-0.6, 0.], [0.6, 0.1], [0., 0.6], [0., -0.6]])
        if vectorized != 2:
            v0 = np.asarray([[0.09, 0.01], [-0.01, -0.01], [0.01, -0.09], [-0.01, 0.09]])
        elif vectorized == 2:
            v0 = torch.tensor([
                        [[0.09], #Vx node 0
                        [0.01] #Vy node 0
                        ],
                        [[-0.01], #Vx node 1
                        [-0.01] #Vy node 1
                        ],
                        [[0.01], #Vx node 2
                        [-0.09] #Vy node 2
                        ],
                        [[-0.01], #Vx node 3
                        [0.09]  #Vy node 3
                        ]])
        max_time = 10
        true_beta = 7.5
        model_beta = 8.


    elif dataset_number == 2:
        z0 = np.asarray([[1., 1.], [1., -1.], [-1., 1.], [-1., -1.], [0., 2.]])
        v0 = torch.tensor([
                        [[-0.15, 0.15, -0.3, 0.3, -0.01, -0.3, 0.3, -0.2, 0.1, -0.15], 
                        [-0.15, 0.15, -0.3, 0.3, -0.5, 0.01, 0.8, -0.4, -0.1, 0.05]
                        ],
                        [[-0.15, 0.15, -0.3, 0.3, -0.01, -0.3, 0.3, -0.2, 0.1, -0.15],
                        [0.15, -0.15, 0.3, -0.3, 0.5, -0.01, -0.8, 0.4, 0.1, -0.05] 
                        ],
                        [[0.15, -0.15, 0.3, -0.3, 0.01, 0.3, -0.3, 0.2, -0.1, 0.15],
                        [-0.15, 0.15, -0.3, 0.3, -0.5, 0.01, 0.8, -0.4, -0.1, -0.05] 
                        ],
                        [[0.15, -0.15, 0.3, -0.3, 0.01, 0.3, -0.3, 0.2, -0.1, 0.15], 
                        [0.15, -0.15, 0.3, -0.3, 0.5, -0.01, -0.8, 0.4, 0.1, 0.05] 
                        ],
                        [[0.05, -0.1, 0.1, -0.05, 0.2, -0.2, 0.01, 0.01, -0.1, 0.1],
                        [-0.3, -0.05, 0.4, -0.3, -0.05, -0.05, 0.01, 0.01, 0.01, 0.05] 
                        ]])
        max_time = 50
        true_beta = 7.5
        model_beta = 8. #must be floating point


    elif dataset_number == 3:
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
        v0 = torch.tensor(v0)
        max_time = 15
        true_beta = 5.0
        model_beta = 5.5




    return z0, v0, true_beta, model_beta, max_time