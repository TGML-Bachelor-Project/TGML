import numpy as np
import torch
from sklearn.model_selection import train_test_split


def remove_interactions(dataset, percentage, device):

    
    dataset_reduced, removed_interactions = train_test_split(dataset, test_size=percentage, random_state=1)


    return dataset_reduced, removed_interactions


