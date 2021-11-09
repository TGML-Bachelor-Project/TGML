import numpy as np
import torch
import pandas as pd
import os
from utils.report_plots.event_distribution import plot_event_dist_temp

### Functions for loading the three datasets

def load_real_dataset_1(path):
    path = path + '/src/data/real/'
    dataset_path = 'email_eu_core_temporal.csv'
    dataset = np.genfromtxt(dataset_path, delimiter=' ')
    print(f'Length of real life original dataset: {len(dataset)}')
    
    ## Remove datapoints the lie faar out timewise
    dataset = dataset[dataset[:,2] < 5e7]
    print(f'Length of reduced real life dataset: {len(dataset)}')

    ## Transform time-intervals
    t_min = min(dataset[:,2])
    t_max = max(dataset[:,2])
    dataset[:,2] =  525.52* ((dataset[:,2] - t_min) / t_max)

    ## Map node indices to the span of 0 to 986
    # for i in range(len(np.unique([dataset[:,0],dataset[:,1]]))):
    #     print(1)
    #plot_event_dist_temp(dataset=dataset)
    
    num_nodes = len(np.unique([dataset[:,0],dataset[:,1]]))

    return dataset, num_nodes

def load_real_dataset_2(path):
    dataset_path = path + '/src/data/real/XXXcsv'
    dataset = pd.read_csv(dataset_path)

    num_nodes = len(np.unique([dataset[:,0],dataset[:,1]]))

    return dataset, num_nodes

def load_real_dataset_3(path):
    dataset_path = path + '/src/data/real/XXX.csv'
    dataset = pd.read_csv(dataset_path)

    num_nodes = len(np.unique([dataset[:,0],dataset[:,1]]))

    return dataset, num_nodes


### Loading the designated dataset

def load_real_dataset(dataset_number):

    path = os.getcwd()
    if dataset_number == 1:
        dataset, num_nodes = load_real_dataset_1(path=path)
    elif dataset_number == 2:
        dataset, num_nodes = load_real_dataset_2(path=path)
    elif dataset_number == 3:
        dataset, num_nodes = load_real_dataset_3(path=path)
    
    return torch.tensor(dataset, dtype=torch.float64), num_nodes

