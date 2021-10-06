# Add necessary folders/files to path
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Set device as cpu or gpu for pytorch
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Running with pytorch device: {device}')
torch.pi = torch.tensor(torch.acos(torch.zeros(1)).item()*2)

# Imports
import numpy as np
from utils import movement
import utils.visualize as visualize
from traintestgyms.standardgym import TrainTestGym
from utils.integralapproximation import riemann_sum
from utils.visualize.positions import node_positions
from models.intensityfunctions.commonbias import CommonBias
from models.constantvelocity.base import ConstantVelocityModel
from data.synthetic.simulators.constantvelocity import ConstantVelocitySimulator


if __name__ == '__main__':
    # A simple example
    seed = 2

    # Set the initial position and velocity
    z0 = np.asarray([[-5, 0], [4, 0], [0, 3], [0, -2]])
    # v0 = np.asarray([[1, 0], [-1, 0], [0, -1], [0, 1]])
    v0 = np.asarray([[0.02, 0], [-0.02, 0], [0, -0.02], [0, 0.02]])

    # Get the number of nodes and dimension size
    num_of_nodes = z0.shape[0]
    dim = z0.shape[1]

    # Set the max time
    maxTime = 6

    # Bias values for nodes
    true_beta = 0.75
    # Simulate events from a non-homogeneous Poisson distribution
    event_simulator = ConstantVelocitySimulator(starting_positions=z0, velocities=v0, 
                                                        T=maxTime, beta=true_beta, seed=seed)
    events = event_simulator.sample_interaction_times_for_all_node_pairs()

    # Define model
    beta = 0.25
    intensity_fun = CommonBias(beta)
    integral_approximator = lambda t0, tn, z, v, i, j, intensity_fun: riemann_sum(t0, tn, n_samples=10, z=z, u=i, v=j, func=intensity_fun)
    model = ConstantVelocityModel(n_points=num_of_nodes, non_intensity_weight=0.2, 
                        intensity_func=intensity_fun, integral_approximator=integral_approximator)

    # Send data and model to same Pytorch device
    model = model.to(device)

    ### Setting up training and evaluation using pytorch-ignite framework
    metrics = {
        'train_loss': [],
        'test_loss': [],
        'Bias Term - Beta': []
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    #Train and evaluate model
    gym = TrainTestGym(num_of_nodes, events, model, device, batch_size=10, training_portion=0.8,
                        optimizer=optimizer, metrics=metrics, time_column_idx=2)
    gym.train_test_model(epochs=100)

    # Print model params
    model_z0 = model.z0.cpu().detach().numpy() 
    model_v0 = model.v0.cpu().detach().numpy()
    print(f'Beta: {model.intensity_function.beta.item()}')
    print(f'Z: {model_z0}')
    print(f'V: {model_v0}')



    # Visualize logloss
    visualize.metrics(metrics)

    # Visualize model Z prediction
    latent_space_positions = [model_z0, z0]
    visualize.compare_positions(latent_space_positions, ['Predicted', 'Actual'])

    # Animate node movements
    node_positions = movement.contant_velocity(model_z0, model_v0, maxTime, time_steps=100)
    visualize.node_movements(node_positions, 'Predicted Node Movements', trail=False)