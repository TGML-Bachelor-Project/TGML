import numpy as np
import torch
import os


### Function for loading the three datasets
def load_data(dataset_path):
    dataset = np.genfromtxt(dataset_path, delimiter=',')
    num_nodes = len(np.unique([dataset[:,0],dataset[:,1]]))
    return dataset, num_nodes


### Loading the designated dataset
def load_real_dataset(dataset_number, debug):

    path = os.path.dirname(os.path.realpath(__file__))
    
    if dataset_number == 1:
        path = os.path.join(path,'datasets/resistance_game4.csv')
        dataset, num_nodes = load_data(dataset_path=path)
        print(f'Length of "Resistance Game 4" dataset: {len(dataset)}')
        model_beta = 4.

    elif dataset_number == 2:
        path = os.path.join(path, 'datasets/email_eu_core_temporal_500nodes.csv')
        dataset, num_nodes = load_data(dataset_path=path)
        print(f'Length of "EU Email" dataset: {len(dataset)}')
        model_beta = 10.
    
    elif dataset_number == 3:
        path = os.path.join(path, 'datasets/tij_pres_LyonSchool.csv')
        dataset, num_nodes = load_data(dataset_path=path)
        print(f'Length of "Lyon School" dataset: {len(dataset)}')
        model_beta = 2.5
    
    return torch.tensor(dataset, dtype=torch.float64), num_nodes, model_beta