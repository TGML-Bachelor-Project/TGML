# Add necessary folders/files to path
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Set device as cpu or gpu for pytorch
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.pi = torch.tensor(torch.acos(torch.zeros(1)).item()*2)

# Imports
from utils.nodes import visualize as node_visual
from utils.metrics import visualize as metric_visual
import numpy as np
from models.basiceuclideandist import BasicEuclideanDistModel
from data.synthetic.simulators.constantvelocity import ConstantVelocitySimulator

if __name__ == '__main__':
    # A simple example
    seed = 2 

    # Set the initial position and velocity
    z0 = np.asarray([[-3, 0], [3, 0], [0, 3], [0, -3]])
    v0 = np.asarray([[1, 0], [-1, 0], [0, -1], [0, 1]])

    # Get the number of nodes and dimension size
    numOfNodes = z0.shape[0]
    dim = z0.shape[1]

    # Set the max time
    maxTime = 6

    # Bias values for nodes
    beta = 0.5

    # Simulate events from a non-homogeneous Poisson distribution
    event_simulator = ConstantVelocitySimulator(starting_positions=z0, velocities=v0, T=maxTime, beta=beta, seed=seed)
    events = event_simulator.sample_interaction_times_for_all_node_pairs()

    # Build dataset of node pair interactions
    dataset = []
    for i in reversed(range(numOfNodes)):
        for j in range(i):
            nodepair_events = events[i][j]
            for np_event in nodepair_events:
                dataset.append([i,j, np_event])

    # Make sure dataset is numpy array
    dataset = np.asarray(dataset)
    # Make sure dataset is sorted according to increasing event times in column index 2
    dataset = dataset[dataset[:, 2].argsort()]
    print('Training and evaluation dataset with events for node pairs')
    print(dataset)
    # Split in train and test set
    training_portion = 0.8
    last_training_idx = int(len(dataset)*training_portion)
    train_data = dataset[:last_training_idx]
    train_loader = DataLoader(train_data, batch_size=2, shuffle=False)
    test_data = dataset[last_training_idx:]
    val_loader = DataLoader(test_data, batch_size=2, shuffle=False)

    
    # Define model
    beta = 0.5
    model = BasicEuclideanDistModel(n_points=numOfNodes, init_beta=beta, riemann_samples=10)

    # Send data and model to same Pytorch device
    model = model.to(device)

    # Model training and evaluation using pytorch-ignite framework
    time_column_idx = 2

    ######### Setting up training
    metrics = {
        'train_loss': [],
        'test_loss': []
    }
    optimizer = Adam(model.parameters(), lr=0.025)

    from ignite.engine import Engine

    def train_step(engine, batch):
        X = batch.to(device)

        model.train()
        optimizer.zero_grad()
        loglikelihood = model(X, t0=engine.t_start, tn=batch[-1][time_column_idx])
        loss = -loglikelihood
        loss.backward()
        optimizer.step()
        metrics['train_loss'].append(loss.item())
        engine.t_start = batch[-1][time_column_idx].to(device)

        return loss

    trainer = Engine(train_step)
    trainer.t_start = torch.tensor([0.0]).to(device)

    ######## Setting up evaluation
    def validation_step(engine, batch):
        model.eval()

        with torch.no_grad():
            X = batch
            test_loss = model(X, t0=engine.t_start, tn=batch[-1][time_column_idx])
            metrics['test_loss'].append(test_loss.item())
            engine.t_start = batch[-1][time_column_idx]
            return test_loss

    evaluator = Engine(validation_step)
    evaluator.t_start = torch.tensor([0.0]).to(device)

    ########## Handlers
    print('Starting model training')
    epochs = 100
    trainer.run(train_loader, max_epochs=epochs)
    print('Completed model training')
    print('Starting model evaluation')
    evaluator.run(val_loader, max_epochs=epochs)
    print('Completed model evaluation')

    # Print model params
    print(f'Beta: {model.beta.item()}')
    print(f'Z: {model.z0.detach().numpy()}')
    print(f'V: {model.v0.detach().numpy()}')

    #Visualize logloss
    metric_visual.logloss(metrics)

    # Visualize model Z prediction
    node_visual.compare_positions(model.z0.detach().numpy(), z0,
        'Model Prediction of Node Starting Positions in 2D Latent Space')