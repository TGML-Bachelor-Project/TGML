import numpy as np
import torch
import os
from utils.report_plots.event_distribution import plot_event_dist_temp

### Functions for loading the three datasets

def load_real_dataset_1(dataset_path):
    dataset = np.genfromtxt(dataset_path, delimiter=',')
    print(f'Length of EU Email original dataset: {len(dataset)}')
    
    num_nodes = len(np.unique([dataset[:,0],dataset[:,1]]))
    # dataset = dataset[:,:2][dataset[:,:2].astype(int) < 51]

    plot_event_dist_temp(dataset=dataset)
    return dataset, num_nodes


def load_real_dataset_2(dataset_path):
    dataset = np.genfromtxt(dataset_path, delimiter=',')
    print(f'Length of Resistance Game 4 original dataset: {len(dataset)}')
    
    plot_event_dist_temp(dataset=dataset)

    num_nodes = len(np.unique([dataset[:,0],dataset[:,1]]))
    return dataset, num_nodes


def load_real_dataset_3(dataset_path):
    dataset = np.genfromtxt(dataset_path, delimiter=' ')


    plot_event_dist_temp(dataset=dataset)

    num_nodes = len(np.unique([dataset[:,0],dataset[:,1]]))
    return dataset, num_nodes



### Loading the designated dataset
def load_real_dataset(dataset_number, debug):

    path = os.getcwd() + '/src/data/real/'
    
    if dataset_number == 1:
        if debug == 1:
            path = path + 'email_eu_core_temporal.csv'
        else:
            path = 'email_eu_core_temporal.csv'
        dataset, num_nodes = load_real_dataset_1(dataset_path=path)
    
    elif dataset_number == 2:
        if debug == 1:
            path = path + 'resistance_game4.csv'
        else:
            path = 'resistance_game4.csv'
        dataset, num_nodes = load_real_dataset_2(dataset_path=path)
    
    elif dataset_number == 3:
        if debug == 1:
            path = path + 'email_eu_core_temporal.csv'
        else:
            path = 'email_eu_core_temporal.csv'
        dataset, num_nodes = load_real_dataset_3(dataset_path=path)
    
    return torch.tensor(dataset, dtype=torch.float64), num_nodes

