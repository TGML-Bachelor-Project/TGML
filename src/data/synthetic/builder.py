import numpy as np
from data.synthetic.sampling import ConstantVelocitySimulator

class DatasetBuilder:
    def __init__(self, starting_positions, starting_velocities,
                    max_time, common_bias, seed) -> None:
        self.simulator = ConstantVelocitySimulator(starting_positions,
                            starting_velocities, max_time, common_bias, seed)

    def build_dataset(self, num_of_nodes:int, time_column_idx:int) -> list:
        '''
        Builds a dataset of node pair interactions

        :param num_of_nodes:    Number of nodes that performs the interactions
        :param time_column_idx: Index of the column in the events data which 
                                holds the time of the interaction
        '''
        events = self.simulator.sample_interaction_times_for_all_node_pairs()
        dataset = []
        for i in reversed(range(num_of_nodes)):
            for j in range(i):
                nodepair_events = events[i][j]
                for np_event in nodepair_events:
                    dataset.append([i,j, np_event])

        # Make sure dataset is numpy array
        dataset = np.asarray(dataset)
        # Make sure dataset is sorted according to increasing event times in column index 2
        if len(dataset) == 0:
            raise Exception('No node interactions have happened. Try increasing the max_time')

        dataset = dataset[dataset[:, time_column_idx].argsort()]
        print('Training and evaluation dataset with events for node pairs')
        print(dataset)
        print(f'Number of interactions: {len(dataset)}')

        return dataset