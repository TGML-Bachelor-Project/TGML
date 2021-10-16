import numpy as np
import matplotlib.pyplot as plt
from data.synthetic.builder import DatasetBuilder
import torch




if __name__ == '__main__':
    

    """First we build a dataset that matches the one we use in training"""

    ## Seeding of model run
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    np.seterr(all='raise')
    device = 'cpu'
    time_col_index = 2
    training_portion = 0.8

    data_set_test = 10
    ## Defining Z and V for synthetic data generation
    if data_set_test == 1:    
        z0 = np.asarray([[-3, 0], [3, 0]])
        v0 = np.asarray([[1, 0], [-1, 0]])
    elif data_set_test == 2:
        z0 = np.asarray([[-3, 0], [3, 0], [0, 3], [0, -3]])
        v0 = np.asarray([[1, 0], [-1, 0], [0, -1], [0, 1]])
    elif data_set_test == 3:
        z0 = np.asarray([[-1, 0], [2, 0], [0, 3], [0, -3]])
        v0 = np.asarray([[1, 0], [-1, 0], [0, -1], [0, 1]])
    elif data_set_test == 4:
        z0 = np.asarray([[-1, 0], [2, 0], [0, 3], [0, -3]])
        v0 = np.asarray([[0.2, 0], [-0.2, 0], [0, -0.2], [0, 0.2]])
    elif data_set_test == 5:
        z0 = np.asarray([[-3, 0], [3, 0], [0, 3], [0, -3], [3, 3], [3, -3]])
        v0 = np.asarray([[1, 0], [-1, 0], [0, -1], [0, 1], [-1, -1], [0, 0.5]])
    elif data_set_test == 6:
        z0 = np.asarray([[-3, 0], [3, 0], [0, 3], [0, -3], [3, 3], [3, -3], [-3, -3], [-3, 3]])
        v0 = np.asarray([[1, 0], [-1, 0], [0, -1], [0, 1], [-1, -1], [0, 0.5], [0, 0], [0.5, 0]])
    elif data_set_test == 7:
        z0 = np.asarray([[-3, 0], [3, 0], [0, 3], [0, -3], [3, 3], [3, -3], [-3, -3], [-3, 3]])
        v0 = np.asarray([[0.11, 0], [-0.1, 0], [0, -0.11], [0, 0.1], [-0.11, -0.09], [0, 0.05], [0, 0], [0.051, 0]])
    ## Simon's synthetic constant velocity data
    elif data_set_test == 10:
        z0 = np.asarray([[-0.6, 0.], [0.6, 0.1], [0., 0.6], [0., -0.6]])
        v0 = np.asarray([[0.09, 0.01], [-0.01, -0.01], [0.01, -0.09], [-0.01, 0.09]])
        a0 = np.array([[0., 0.], [0., 0.], [0., 0.], [0., 0.]])  # Negligble
    
    num_nodes = z0.shape[0]
    true_beta = 7.5
    max_time = 10

    data_builder = DatasetBuilder(starting_positions=z0, 
                                        starting_velocities=v0,
                                        max_time=max_time, 
                                        common_bias=true_beta, 
                                        seed=seed, 
                                        device=device)
    dataset = data_builder.build_dataset(num_nodes, time_column_idx=time_col_index)
    interaction_count = len(dataset)
    
    # Verify time ordering
    prev_t = 0.
    for row in dataset:
        cur_t = row[time_col_index]
        assert cur_t > prev_t
        prev_t = cur_t

    last_training_idx = int(len(dataset)*training_portion)
    dataset = np.asarray(dataset)
    print(dataset)

    """Now we plot the dataset in the ways we want"""
    
    ## Histogram of the generated dataset, red line indicating 80 percent training data split
    plt.hist(dataset[:, 2], bins=50, )
    plt.title("Data: Event histogram")
    plt.vlines(x=dataset[last_training_idx][2].item(), ymin=0, ymax=2000, color="r")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()










