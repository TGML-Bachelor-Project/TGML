import numpy as np
import random
random.seed(1)


def remove_node_pairs(dataset, num_nodes, percentage, node_pairs=None):
    
    ## Unless specified, node pairs to be removed will be randomly selected
    if node_pairs == None:
        nodepair_ind = np.triu_indices(num_nodes, k=1)
        all_node_pairs = list(zip(nodepair_ind[0], nodepair_ind[1]))
        num_pairs = len(all_node_pairs)
        num_pairs_remove = max(int(num_pairs*(1-percentage)), 1)
        node_pairs = random.choices(all_node_pairs, k=num_pairs_remove)

    dataset_reduced = []
    for tup in dataset:
        keep = True
        for node_pair in node_pairs:
            if int(tup[0]) == node_pair[0] and int(tup[1]) == node_pair[1]:
                keep = False
            else:
                pass
        if keep:
            dataset_reduced.append(tup)
    
    return dataset_reduced, node_pairs