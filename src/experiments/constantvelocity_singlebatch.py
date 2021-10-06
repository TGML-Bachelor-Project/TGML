# Add necessary folders/files to path
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Set device as cpu or gpu for pytorch
import torch
# from torch.optim import Adam
from torch.utils.data import DataLoader
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Running with pytorch device: {device}')
torch.pi = torch.tensor(torch.acos(torch.zeros(1)).item()*2)

# Imports
import time
import numpy as np
from utils import movement
import utils.visualize as visualize
from data.builder import build_dataset
from utils.integralapproximation import riemann_sum
from utils.visualize.positions import node_positions
from models.intensityfunctions.commonbias import CommonBias
from models.constantvelocity.base import ConstantVelocityModel
from data.synthetic.simulators.constantvelocity import ConstantVelocitySimulator

def nll(ll):
    return -ll

def single_batch_train(net, training_data, test_data, num_epochs, learning_rate=0.001):
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    training_losses = []
    test_losses = []
    beta = []
    tn_train = training_data[-1][-1] # last time point in training data
    tn_test = test_data[-1][-1] # last time point in test data
    n_train = len(train_data)
    n_test = len(test_data)


    for epoch in range(num_epochs):
        start_time = time.time()
        running_loss = 0.

        net.train()
        optimizer.zero_grad()
        output = net(training_data, t0=0, tn=tn_train)
        loss = nll(output)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        net.eval()
        with torch.no_grad():
            test_output = net(test_data, t0=tn_train, tn=tn_test)
            test_loss = nll(test_output)
                

        avg_train_loss = running_loss / n_train
        avg_test_loss = test_loss.cpu() / n_test
        # avg_test_loss
        current_time = time.time()
        
        if epoch == 0 or (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}")
            print(f"elapsed time: {current_time - start_time}" )
            print(f"train loss: {avg_train_loss}")
            print(f"test loss: {avg_test_loss}")
            print("State dict:")
            print(net.state_dict())
            #print(f"train event to non-event ratio: {train_ratio.item()}")
            #print(f"test event to non-event-ratio: {test_ratio.item()}")
        
        training_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        beta.append(net.intensity_function.beta.item())
    
    return net, training_losses, test_losses, beta



if __name__ == '__main__':
    # A simple example
    seed = 2

    # Set the initial position and velocity
    z0 = np.asarray([[-5, 0], [4, 0], [0, 3], [0, -2]])
    # v0 = np.asarray([[1, 0], [-1, 0], [0, -1], [0, 1]])
    v0 = np.asarray([[0.2, 0], [-0.2, 0], [0, -0.2], [0, 0.2]])

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
    train_loader = DataLoader(train_data, batch_size=10, shuffle=False)
    test_data = dataset[last_training_idx:]
    val_loader = DataLoader(test_data, batch_size=10, shuffle=False)

    
    # Define model
    non_weight = 0.2
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
        'beta': []
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=0.025)


    print('Starting model training')
    epochs = 100
    model, metrics['train_loss'], metrics['test_loss'], metrics['beta'] = single_batch_train(net=model, training_data=train_data, 
                        test_data=test_data, num_epochs=epochs)
    print('Completed model training')

    ## Extract model params
    model_z0 = model.z0.cpu().detach().numpy() 
    model_v0 = model.v0.cpu().detach().numpy()
    print(f'Beta: {model.intensity_function.beta.item()}')
    print(f'Z: {model_z0}')
    print(f'V: {model_v0}')

    ### Visualization
    ## Visualize logloss
    visualize.metrics(metrics)

    ## Visualize model Z prediction
    latent_space_positions = [model_z0, z0]
    visualize.compare_positions(latent_space_positions, ['Predicted', 'Actual'])

    ## Animate node movements
    node_positions = movement.contant_velocity(model_z0, model_v0, maxTime, time_steps=100)
    visualize.node_movements(node_positions, 'Predicted Node Movements', trail=False)