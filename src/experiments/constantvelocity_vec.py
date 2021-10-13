import os
import sys

from torch.optim.optimizer import Optimizer

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append('/home/augustsemrau/drive/bachelor/TGML/src')
print(sys.path)
print(os.path.dirname(__file__))
# Set device as cpu or gpu for pytorch
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Running with pytorch device: {device}')
torch.pi = torch.tensor(torch.acos(torch.zeros(1)).item()*2)

# Imports
import numpy as np
from utils.nodes.positions import get_contant_velocity_positions 
from argparse import ArgumentParser
import utils.visualize as visualize
from traintestgyms.ignitegym import TrainTestGym
from utils.visualize.positions import node_positions
from data.synthetic.builder import DatasetBuilder
from models.constantvelocity.vectorized import ConstantVelocityModel


if __name__ == '__main__':

    ### Parse Arguments for running in terminal
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--max_time', '-MT', default=100, type=int)
    arg_parser.add_argument('--true_beta', '-TB', default=7., type=float)
    arg_parser.add_argument('--model_beta', '-MB', default=0.01, type=float)
    arg_parser.add_argument('--learning_rate', '-LR', default=0.025, type=float)
    arg_parser.add_argument('--num_epochs', '-NE', default=1000, type=int)
    arg_parser.add_argument('--train_batch_size', '-TBS', default=1000, type=int)
    arg_parser.add_argument('--training_portion', '-TP', default=0.8, type=float)
    arg_parser.add_argument('--data_set_test', '-DATA', default=1, type=int)
    arg_parser.add_argument('--sequential_training', '-SEQ', default=0, type=int)
    args = arg_parser.parse_args()


    ### Set all input arguments
    seed = 2
    max_time = args.max_time
    true_beta = args.true_beta
    model_beta = args.model_beta  # Model-initialized beta
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    train_batch_size = args.train_batch_size
    training_portion = args.training_portion
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
        v0 = np.asarray([[1, 0], [-1, 0], [0, -1], [0, 1], [-1, -1], [0, 0.5], [0, 0], [0.5, 0]])
    num_nodes = z0.shape[0]

    ## Initialize data_builder for simulating node interactions from known Poisson Process
    data_builder = DatasetBuilder(starting_positions=z0, 
                                    starting_velocities=v0,
                                    max_time=max_time, 
                                    common_bias=true_beta, 
                                    seed=seed, 
                                    device=device)
    dataset = data_builder.build_dataset(num_nodes, time_column_idx=2)

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
                        time_column_idx=2)

    
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