import torch
import numpy as np

class DatasetBuilder:
    def __init__(self, simulator, device, normalization_max_time=None) -> None:
        self.simulator = simulator
        self.device = device
        self.max_time = normalization_max_time

    def build_dataset(self, num_of_nodes:int, time_column_idx:int) -> list:
        '''
        Builds a dataset of node pair interactions

        :param num_of_nodes:    Number of nodes that performs the interactions
        :param time_column_idx: Index of the column in the events data which 
                                holds the time of the interaction
        '''
        events = self.simulator.sample_interaction_times_for_all_node_pairs()
        nodepair_ind = np.triu_indices(num_of_nodes, k=1)
        dataset = []
        for i,j in zip(nodepair_ind[0], nodepair_ind[1]):
            nodepair_events = events[i][j]
            for np_event in nodepair_events:
                dataset.append([i,j, np_event])

        # Make sure dataset is numpy array
        dataset = np.asarray(dataset)
        dataset = dataset[dataset[:, time_column_idx].argsort()]
        # Make sure dataset is sorted according to increasing event times in column index 2
        if len(dataset) == 0:
            raise Exception('No node interactions have happened. Try increasing the max_time')

        print(f'Dataset generated with number of interactions: {len(dataset)}')
        print(dataset)
        dataset = torch.from_numpy(dataset)
        if self.max_time:
            self.max_time = torch.tensor(self.max_time)
            dataset[:,2] = dataset[:,2]/self.max_time


        return dataset