import os
import sys
import wandb
from torch.optim.optimizer import Optimizer

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append('/home/augustsemrau/drive/bachelor/TGML/src')
print(sys.path)
print(os.path.dirname(__file__))
# Set device as cpu or gpu for pytorch
import torch
device = 'cpu'
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Running with pytorch device: {device}')
torch.pi = torch.tensor(torch.acos(torch.zeros(1)).item()*2)

# Imports
import numpy as np
from utils.nodes.positions import get_contant_velocity_positions 
from argparse import ArgumentParser
import utils.visualize as visualize
from traintestgyms.ignitegym import TrainTestGym
from utils.visualize.positions import node_positions

from models.constantvelocity.standard import ConstantVelocityModel

## Data import
from data.synthetic.builder import DatasetBuilder  # Type 0, ours
from data.simon_synthetic.nhpp_simon import NodeSpace  # Type 1, Simon's
from data.simon_synthetic.nhpp_simon import root_matrix, monotonicity_mat, nhpp_mat, get_entry


if __name__ == '__main__':

    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    np.seterr(all='raise')

    ### Parse Arguments for running in terminal
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--max_time', '-MT', default=100, type=int)
    arg_parser.add_argument('--true_beta', '-TB', default=7., type=float)
    arg_parser.add_argument('--model_beta', '-MB', default=5., type=float)
    arg_parser.add_argument('--learning_rate', '-LR', default=0.025, type=float)
    arg_parser.add_argument('--num_epochs', '-NE', default=1000, type=int)
    arg_parser.add_argument('--train_batch_size', '-TBS', default=1000, type=int)
    arg_parser.add_argument('--training_portion', '-TP', default=0.8, type=float)
    arg_parser.add_argument('--data_type', '-DT', default=0, type=int)
    arg_parser.add_argument('--data_set_test', '-DS', default=6, type=int)
    arg_parser.add_argument('--sequential_training', '-SEQ', default=0, type=int)
    args = arg_parser.parse_args()


    ### Set all input arguments
    max_time = args.max_time
    true_beta = args.true_beta
    model_beta = args.model_beta  # Model-initialized beta
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    train_batch_size = args.train_batch_size
    training_portion = args.training_portion
    data_type = args.data_type
    data_set_test = args.data_set_test
    sequential_training = args.sequential_training


    ### Build dataset

    ## Initial Z and V
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
        true_beta = 7.5
        model_beta = 7.1591

    num_nodes = z0.shape[0]


    ### Set input parameters as config for Weights and Biases
    wandb_config = {'seed': seed,
                    'max_time': max_time,
                    'true_beta': true_beta,
                    'model_beta': model_beta,
                    'learning_rate': learning_rate,
                    'num_epochs': num_epochs,
                    # 'non_intensity_weight': non_intensity_weight,
                    'train_batch_size': train_batch_size,
                    'num_nodes': num_nodes,
                    'training_portion': training_portion,
                    'sequential_training': sequential_training,
                    'data_type': data_type}

    ## Initialize WandB for logging config and metrics
    wandb.init(project='TGML1', entity='augustsemrau', config=wandb_config)

    time_col_index = 2

    ### Initialize data_builder for simulating node interactions from known Poisson Process
    ## Our data generation
    if data_type == 0:
        data_builder = DatasetBuilder(starting_positions=z0, 
                                        starting_velocities=v0,
                                        max_time=max_time, 
                                        common_bias=true_beta, 
                                        seed=seed, 
                                        device=device)
        dataset = data_builder.build_dataset(num_nodes, time_column_idx=time_col_index)
        interaction_count = len(dataset)

    ## Simon's data generation
    elif data_type == 1:
        ns_gt = NodeSpace()
        ns_gt.beta = true_beta
        ns_gt.init_conditions(z0, v0, a0)
        
        t = np.linspace(0, max_time)
        rmat = root_matrix(ns_gt) 
        mmat = monotonicity_mat(ns_gt, rmat)
        nhppmat = nhpp_mat(ns=ns_gt, time=t, root_matrix=rmat, monotonicity_matrix=mmat)

        # create data set and sort by time
        ind = np.triu_indices(num_nodes, k=1)
        dataset = []
        for u,v in zip(ind[0], ind[1]):
            event_times = get_entry(nhppmat, u=u, v=v)
            for e in event_times:
                dataset.append([u, v, e])

        dataset = np.array(dataset)
        dataset = dataset[dataset[:,time_col_index].argsort()]

        # verify time ordering
        prev_t = 0.
        for row in dataset:
            cur_t = row[time_col_index]
            assert cur_t > prev_t
            prev_t = cur_t
        
        interaction_count = len(dataset)
        dataset = torch.from_numpy(dataset).to(device)


    ### Setup model
    model = ConstantVelocityModel(n_points=num_nodes, beta=model_beta)
    print('Model initial node start positions\n', model.z0)
    model = model.to(device)  # Send model to torch

    ### Train and evaluate model
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    metrics = {'train_loss': [], 'test_loss': [], 'Bias Term - Beta': []}

    gym = TrainTestGym(dataset=dataset, 
                        model=model, 
                        device=device, 
                        batch_size=train_batch_size, 
                        training_portion=training_portion,
                        optimizer=optimizer, 
                        metrics=metrics, 
                        time_column_idx=time_col_index)

    
    ### Model training starts

    ## Non-sequential model training
    if sequential_training == 0:
        model.z0.requires_grad, model.v0.requires_grad, model.beta.requires_grad = True, True, True
        # gym.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        gym.train_test_model(epochs=num_epochs)

    ## Sequential model training
    elif sequential_training == 1:
        model.z0.requires_grad, model.v0.requires_grad, model.beta.requires_grad = False, False, False
        for i in range(3):
            if i == 0:
                model.z0.requires_grad = True
            elif i == 1:
                #model.z0.requires_grad = False
                model.v0.requires_grad = True
            elif i == 2:
                #model.z0.requires_grad = False
                #model.v0.requires_grad = False
                model.beta.requires_grad = True

            gym.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            gym.train_test_model(epochs=num_epochs)

    ### Results

    ## Print model params
    model_z0 = model.z0.cpu().detach().numpy() 
    model_v0 = model.v0.cpu().detach().numpy()
    print(f'Beta: {model.beta.item()}')
    print(f'Z: {model_z0}')
    print(f'V: {model_v0}')


    ### Log metrics to Weights and Biases
    wandb_metrics = {'number_of_interactions': interaction_count,
                    'metric_final_beta': metrics['Bias Term - Beta'][-1],
                    'metric_final_testloss': metrics['test_loss'][-1],
                    'metric_final_trainloss': metrics['train_loss'][-1],
                    'beta': metrics['Bias Term - Beta'],
                    'test_loss': metrics['test_loss'],
                    'train_loss': metrics['train_loss']}
    wandb.log(wandb_metrics)

    ### Visualizations
    '''
    visualize.metrics(metrics)

    ## Learned Z and true Z
    latent_space_positions = [model_z0, z0]
    visualize.node_positions(latent_space_positions, 'Actual vs Predicted')
    visualize.compare_positions(latent_space_positions, ['Predicted', 'Actual'])

    ## Animation of learned node movements
    node_positions = get_contant_velocity_positions(model_z0, model_v0, max_time, time_steps=100)
    visualize.node_movements(node_positions, 'Predicted Node Movements', trail=False)
    '''