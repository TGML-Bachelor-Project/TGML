import random
random.seed(1)
import numpy as np
import torch
from sklearn.model_selection import train_test_split


def remove_interactions(dataset, percentage, device):

    dataset_reduced = []
    removed_interactions = []
    for tup in dataset.tolist():
        if random.random() > percentage:
            dataset_reduced.append(tup)
        else:
            removed_interactions.append(tup)

    dataset_reduced = torch.from_numpy(np.asarray(dataset_reduced)).to(device)
    print(f'Reduced training set by random selection, training set now contains interactions: {len(dataset_reduced)}')
    return dataset_reduced, removed_interactions


def auc_removed_interactions(s):

    return 