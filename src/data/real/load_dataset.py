import numpy as np
import torch
import os
from utils.report_plots.event_distribution import plot_event_dist_eu_data
from utils.report_plots.event_distribution import plot_event_dist_resistance_data

### Functions for loading the three datasets

def load_real_dataset_1(dataset_path):
    dataset = np.genfromtxt(dataset_path, delimiter=',')
    print(f'Length of "EU Email" dataset: {len(dataset)}')
    
    # dataset = dataset[:,:2][dataset[:,:2].astype(int) < 51]

    # plot_event_dist_eu_data(dataset=dataset)
    num_nodes = len(np.unique([dataset[:,0],dataset[:,1]]))
    return dataset, num_nodes
    



def load_real_dataset_2(dataset_path):
    dataset = np.genfromtxt(dataset_path, delimiter=',')
    print(f'Length of "Resistance Game 4" dataset: {len(dataset)}')
    

    # plot_event_dist_resistance_data(dataset=dataset)

    num_nodes = len(np.unique([dataset[:,0],dataset[:,1]]))
    return dataset, num_nodes



def load_real_dataset_3(dataset_path):
    dataset = np.genfromtxt(dataset_path, delimiter=',')
    print(f'Length of "Lyon School" dataset: {len(dataset)}')

    # plot_event_dist_temp(dataset=dataset)

    num_nodes = len(np.unique([dataset[:,0],dataset[:,1]]))
    return dataset, num_nodes




### Loading the designated dataset
def load_real_dataset(dataset_number, debug):

    path = os.path.dirname(os.path.realpath(__file__))
    
    if dataset_number == 1:
        path = os.path.join(path, 'edited_datasets/email_eu_core_temporal.csv')
        dataset, num_nodes = load_real_dataset_1(dataset_path=path)
        model_beta = 10.
    
    elif dataset_number == 2:
        path = os.path.join(path,'edited_datasets/resistance_game4.csv')
        dataset, num_nodes = load_real_dataset_2(dataset_path=path)
        model_beta = 4.
    
    elif dataset_number == 3:
        path = os.path.join(path, 'edited_datasets/tij_pres_LyonSchool.csv')
        dataset, num_nodes = load_real_dataset_3(dataset_path=path)
        model_beta = 5.
    
    return torch.tensor(dataset, dtype=torch.float64), num_nodes, model_beta

