# Add necessary folders/files to path
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Set device as cpu or gpu for pytorch
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from integralapproximation import riemann_sum
device = 'cpu'# 'cuda' if torch.cuda.is_available() else 'cpu'

# Imports
import numpy as np
from models.constantvelocity import ConstantVelocityModel
from data.synthetic.simulators.constantvelocity import ConstantVelocitySimulator

if __name__ == '__main__':
    # A simple example for 2 nodes
    seed = 2

    # Set the initial position and velocity
    x0 = np.asarray([[-3, 0], [3, 0]])
    v0 = np.asarray([[1, 0], [-1, 0]])

    # Get the number of nodes and dimension size
    numOfNodes = x0.shape[0]
    dim = x0.shape[1]

    # Set the max time
    maxTime = 6

    # Bias values for nodes
    gamma = 0.5 * np.ones(shape=(numOfNodes, ))

    # Simulate events from a non-homogeneous Poisson distribution
    event_simulator = ConstantVelocitySimulator(starting_positions=x0, velocities=v0, T=maxTime, beta=gamma, seed=seed)
    events = event_simulator.sample_interaction_times_for_all_node_pairs()

    # Build dataset of node pair interactions
    dataset = []
    for i in range(numOfNodes):
        for j in range(i+1, numOfNodes):
            nodepair_events = events[i][j]
            print("Events for node pair ({}-{}): {}".format(i, j, nodepair_events))
            for np_event in nodepair_events:
                dataset.append([i,j, np_event])

    # Make sure dataset is numpy array
    dataset = np.asarray(dataset)
    # Make sure dataset is sorted according to increasing event times in column index 2
    dataset = dataset[dataset[:, 2].argsort()]
    print(dataset)
    # Split in train and test set
    training_portion = 0.8
    last_training_idx = int(len(dataset)*training_portion)
    train_data = dataset[:last_training_idx]
    train_loader = DataLoader(train_data, batch_size=2, shuffle=False)
    test_data = dataset[last_training_idx:]
    val_loader = DataLoader(test_data, batch_size=2, shuffle=False)

    
    # Define model
    betas = [0.1, 0.1]
    model = ConstantVelocityModel(n_points=4, init_beta=betas, riemann_samples=2, node_pair_samples=3)

    # Send data and model to same Pytorch device
    model = model.to(device)

    # Model training and evaluation using pytorch-ignite framework
    t_start = torch.tensor([0.0]).to(device)
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
        loglikelihood = model(X, t0=t_start, tn=batch[-1][time_column_idx])
        loss = -loglikelihood
        metrics['train_loss'].append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    trainer = Engine(train_step)

    ######## Setting up evaluation
    def validation_step(engine, batch):
        model.eval()

        with torch.no_grad():
            X = batch
            test_loss = model(X, t0=t_start, tn=batch[-1][time_column_idx])
            metrics['test_loss'].append(test_loss.item())
            return test_loss

    evaluator = Engine(validation_step)


    ########## Handlers
    from ignite.engine import Events

    # Show a message when the training begins
    @trainer.on(Events.STARTED)
    def start_message():
        print("Start training!")

    # Handler can be what you want, here a lambda ! 
    trainer.add_event_handler(
        Events.COMPLETED, 
        lambda _: print("Training completed!")
    )

    # Run evaluator on val_loader every trainer's epoch completed
    @trainer.on(Events.EPOCH_COMPLETED)
    def run_validation():
        evaluator.run(val_loader)

    @evaluator.on(Events.STARTED)
    def eval_start_message():
        print('Starting evaluation!')

    # Handler can be what you want, here a lambda ! 
    evaluator.add_event_handler(
        Events.COMPLETED, 
        lambda _: print("Evaluation completed!")
    )

    trainer.run(train_loader, max_epochs=100)

    # Print model params
    print(f'Beta: {model.beta}')
    print(f'Z: {model.z0}')
    print(f'V: {model.v0}')

    # Plot loss in training and test
    import matplotlib.pyplot as plt
    plt.title('Model Log Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Log Loss')
    for k in metrics.keys():
        plt.plot(metrics[k], label=k)
    plt.legend()
    plt.show()