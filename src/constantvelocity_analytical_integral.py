
# Set device as cpu or gpu for pytorch
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Running with pytorch device: {device}')
torch.pi = torch.tensor(torch.acos(torch.zeros(1)).item()*2)

# Imports
import numpy as np
import wandb
from argparse import ArgumentParser

from data.synthetic.simulators.constantvelocity import ConstantVelocitySimulator
from models.intensityfunctions.commonbias import CommonBias
from models.constantvelocity.base import ConstantVelocityModel
from utils.integralapproximation import analytical_squared_euclidean, riemann_sum
from traintestgyms.standardgym import TrainTestGym
from utils.visualize.positions import node_positions
from utils import movement
import utils.visualize as visualize




if __name__ == '__main__':

    ### Parse Arguments for running in terminal
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--max_time', '-MT', default=100, type=int)
    arg_parser.add_argument('--true_beta', '-TB', default=0.5, type=float)
    arg_parser.add_argument('--model_beta', '-MB', default=0.25, type=float)
    arg_parser.add_argument('--learning_rate', '-LR', default=0.01, type=float)
    arg_parser.add_argument('--num_epochs', '-NE', default=10, type=int)
    arg_parser.add_argument('--non_intensity_weight', '-NIW', default=0.2, type=float)
    arg_parser.add_argument('--train_batch_size', '-TBS', default=250, type=int)
    arg_parser.add_argument('--training_portion', '-TP', default=0.8, type=float)
    args = arg_parser.parse_args()


    ### Set all input arguments
    seed = 2
    max_time = args.max_time
    true_beta = args.true_beta
    model_beta = args.model_beta  # Model-initialized beta
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    non_intensity_weight = args.non_intensity_weight
    train_batch_size = args.train_batch_size
    training_portion = args.training_portion

    ## Set the initial position and velocity
    z0 = np.asarray([[-5, 0], [4, 0], [0, 3], [0, -2]])
    v0 = np.asarray([[0.02, 0], [-0.02, 0], [0, -0.02], [0, 0.02]])
    num_nodes = z0.shape[0]  # Number of nodes


    ### Set input parameters as config for Weights and Biases
    wandb_config = {'max_time': max_time,
                    'true_beta': true_beta,
                    'model_beta': model_beta,
                    'learning_rate': learning_rate,
                    'num_epochs': num_epochs,
                    'non_intensity_weight': non_intensity_weight,
                    'train_batch_size': train_batch_size,
                    'training_portion': training_portion,
                    'num_nodes': num_nodes}

    ## Initialize WandB for logging config and metrics
    wandb.init(project='TGML', entity='augustsemrau', config=wandb_config)

    
    ### Simulate events from a non-homogeneous Poisson distribution
    ## Initialize simulator
    event_simulator = ConstantVelocitySimulator(starting_positions=z0, 
                                                velocities=v0, 
                                                T=max_time, 
                                                beta=true_beta, 
                                                seed=seed)
    ## Compute events
    events = event_simulator.sample_interaction_times_for_all_node_pairs()


    ### Setup model
    intensity_fun = CommonBias(model_beta)
    model = ConstantVelocityModel(n_points=num_nodes, 
                                    non_intensity_weight=non_intensity_weight, 
                                    intensity_func=intensity_fun, 
                                    integral_approximator=analytical_squared_euclidean)
    print('Model initial node start positions\n', model.z0)

    ## Send data and model to same Pytorch device
    model = model.to(device)

    ## Setting up training and evaluation using pytorch-ignite framework
    metrics = {
        'train_loss': [],
        'test_loss': [],
        'Bias Term - Beta': []
    }

    ## 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    ### Train and evaluate model
    gym = TrainTestGym(num_nodes, events, model, device, 
                        batch_size=train_batch_size, 
                        training_portion=training_portion,
                        optimizer=optimizer, 
                        metrics=metrics, 
                        time_column_idx=2)
    gym.train_test_model(epochs=num_epochs)

    # Print model params
    model_z0 = model.z0.cpu().detach().numpy() 
    model_v0 = model.v0.cpu().detach().numpy()
    print(f'Beta: {model.intensity_function.beta.item()}')
    print(f'Z: {model_z0}')
    print(f'V: {model_v0}')


    ### Log metrics to Weights and Biases
    wandb_metrics = {'metric_final_beta': metrics['Bias Term - Beta'][-1],
                    'metric_final_testloss': metrics['test_loss'][-1],
                    'metric_final_trainloss': metrics['train_loss'][-1],
                    'beta': metrics['Bias Term - Beta'],
                    'test_loss': metrics['test_loss'],
                    'train_loss': metrics['train_loss']}
    wandb.log(wandb_metrics)





    ### Visualizations

    ## Logloss metrics and Bias term Beta
    visualize.metrics(metrics)

    ## Learned Z and true Z
    latent_space_positions = [model_z0, z0]
    visualize.compare_positions(latent_space_positions, ['Predicted', 'Actual'])

    ## Animation of learned node movements
    node_positions = movement.contant_velocity(model_z0, model_v0, max_time, time_steps=100)
    visualize.node_movements(node_positions, 'Predicted Node Movements', trail=False)

