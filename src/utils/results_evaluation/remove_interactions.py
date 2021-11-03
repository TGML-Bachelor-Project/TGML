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
    removed_interactions = torch.from_numpy(np.asarray(removed_interactions)).to(device)
    print(f'Reduced training set by random selection, interactions: {len(dataset_reduced)}')
    print(f'Removed number of interactions: {len(removed_interactions)}')
    return dataset_reduced, removed_interactions


def auc_removed_interactions(removed_interactions, num_nodes, result_model):
    
    nodepair_ind = np.triu_indices(num_nodes, k=1)
    all_node_pairs = list(zip(nodepair_ind[0], nodepair_ind[1]))

    ## First we sample alernative node pairs for each of the timestamps in the removed interactions
    alternate_node_pairs = []
    for tup in removed_interactions.tolist():
        tup_copy = tup
        excluded_node_pair = [tup[0], tup[1]]
        alternate_node_pair = random.choice([i for i in all_node_pairs if i not in excluded_node_pair])
        tup_copy[0], tup_copy[1] = alternate_node_pair[0], alternate_node_pair[1]
        alternate_node_pairs.append(tup_copy)

    auc_score = 1
    return auc_score