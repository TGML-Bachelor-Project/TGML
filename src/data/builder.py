import numpy as np

def build_dataset(num_of_nodes:int, events:list) -> list:
    '''
    Builds a dataset of node pair interactions

    :param num_of_nodes:    Number of nodes that performs the interactions
    :param events:          A 2d list of events. First dimension is interactions 
                            second dimension is [index_node_i, index_node_j, interaction_time]
                            for instance events[0,:] could be [0,1,2.43] where
                            node 0 and node 1 interacted at time 2.43
    '''
    dataset = []
    for i in reversed(range(num_of_nodes)):
        for j in range(i):
            nodepair_events = events[i][j]
            for np_event in nodepair_events:
                dataset.append([i,j, np_event])

    # Make sure dataset is numpy array
    dataset = np.asarray(dataset)
    # Make sure dataset is sorted according to increasing event times in column index 2
    time_column_idx = 2
    dataset = dataset[dataset[:, time_column_idx].argsort()]
    print('Training and evaluation dataset with events for node pairs')
    print(dataset)

    return dataset