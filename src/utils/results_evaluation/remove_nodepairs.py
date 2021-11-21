import numpy as np
import random
random.seed(1)
import torch

def remove_node_pairs(dataset, num_nodes, percentage, device, node_pairs=None):
    
    ## Unless specified, node pairs to be removed will be randomly selected
    if node_pairs == None:
        nodepair_ind = np.triu_indices(num_nodes, k=1)
        all_node_pairs = list(zip(nodepair_ind[0], nodepair_ind[1]))
        num_pairs = len(all_node_pairs)
        num_pairs_remove = max(int(num_pairs*percentage), 1)
        removed_node_pairs = random.choices(all_node_pairs, k=num_pairs_remove)
    else:
        removed_node_pairs = node_pairs
        
    dataset_reduced = []
    for tup in dataset.tolist():
        keep = True
        for node_pair in removed_node_pairs:
            if int(tup[0]) == node_pair[0] and int(tup[1]) == node_pair[1] or int(tup[0]) == node_pair[1] and int(tup[1]) == node_pair[0]:
                keep = False
            else:
                pass
        if keep:
            dataset_reduced.append(tup)

    dataset_reduced = torch.from_numpy(np.asarray(dataset_reduced)).to(device)
    print(f'Removed node pairs: {removed_node_pairs}, training set now contains interactions: {len(dataset_reduced)}')
    return dataset_reduced, removed_node_pairs