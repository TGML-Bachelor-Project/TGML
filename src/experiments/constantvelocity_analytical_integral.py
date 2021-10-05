# Add necessary folders/files to path
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Set device as cpu or gpu for pytorch
import torch
from ignite.engine import Events
from torch.utils.data import DataLoader
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Running with pytorch device: {device}')
torch.pi = torch.tensor(torch.acos(torch.zeros(1)).item()*2)

# Imports
import numpy as np
from utils import movement
import utils.visualize as visualize
from data.builder import build_dataset
from utils.integralapproximation import analytical_squared_euclidean, riemann_sum
from utils.visualize.positions import node_positions
from models.intensityfunctions.commonbias import CommonBias
from models.constantvelocity.base import ConstantVelocityModel
from data.synthetic.simulators.constantvelocity import ConstantVelocitySimulator

from ignite.engine import Engine



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
    beta = 0.75

    # Simulate events from a non-homogeneous Poisson distribution
    event_simulator = ConstantVelocitySimulator(starting_positions=z0, velocities=v0, 
                                                        T=maxTime, beta=beta, seed=seed)
    events = event_simulator.sample_interaction_times_for_all_node_pairs()


    # Split in train and test set
    time_column_idx = 2
    dataset = build_dataset(num_of_nodes, events, time_column_idx)
    training_portion = 0.8
    last_training_idx = int(len(dataset)*training_portion)
    train_data = dataset[:last_training_idx]
    train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=False)
    test_data = dataset[last_training_idx:]
    val_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

    
    # Define model
    beta = 0.25
    intensity_fun = CommonBias(beta)
    model = ConstantVelocityModel(n_points=num_of_nodes, non_intensity_weight=0.2, 
                        intensity_func=intensity_fun, integral_approximator=analytical_squared_euclidean)

    # Send data and model to same Pytorch device
    model = model.to(device)


    ### Setting up training and evaluation using pytorch-ignite framework
    metrics = {
        'train_loss': [],
        'test_loss': [],
        'Bias Term - Beta': []
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    ### Training setup
    def train_step(engine, batch):
        X = batch.to(device)

        model.train()
        optimizer.zero_grad()
        train_loglikelihood = model(X, t0=engine.t_start, tn=batch[-1][time_column_idx])
        loss = - train_loglikelihood
        loss.backward()
        optimizer.step()
        metrics['train_loss'].append(loss.item())
        metrics['Bias Term - Beta'].append(model.intensity_function.beta.item())
        engine.t_start = batch[-1][time_column_idx].to(device)

        return loss

    trainer = Engine(train_step)
    trainer.t_start = 0.


    ### Evaluation setup
    def validation_step(engine, batch):
        model.eval()

        with torch.no_grad():
            X = batch.to(device)
            test_loglikelihood = model(X, t0=engine.t_start, tn=batch[-1][time_column_idx].item())
            test_loss = - test_loglikelihood
            # optimizer.step()
            metrics['test_loss'].append(test_loss.item())
            engine.t_start = batch[-1][time_column_idx]
            return test_loss

    evaluator = Engine(validation_step)
    evaluator.t_start = 0.

    ### Handlers
    @trainer.on(Events.EPOCH_COMPLETED(every=5))
    def evaluate_model():
        evaluator.run(val_loader)


    epochs = 100
    print(f'Starting model training with {epochs} epochs')
    trainer.run(train_loader, max_epochs=epochs)
    print('Completed model training')

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