### Packages
import os
import sys
import wandb
import numpy as np
import torch
from argparse import ArgumentParser

from wandb.sdk import wandb_run

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append('/home/augustsemrau/drive/bachelor/TGML/src')



### Code imports
## Data
from data.synthetic.datasets.init_params import get_initial_parameters
from data.synthetic.builder import DatasetBuilder
from data.synthetic.stepwisebuilder import StepwiseDatasetBuilder
from data.synthetic.sampling.constantvelocity import ConstantVelocitySimulator
from data.synthetic.sampling.tensor_stepwiseconstantvelocity import StepwiseConstantVelocitySimulator
from data.real.load_dataset import load_real_dataset
from utils.results_evaluation.remove_nodepairs import remove_node_pairs
from utils.results_evaluation.remove_interactions import acc_removed_interactions, remove_interactions

## Models
from models.nodynamics import NoDynamicsModel
from models.constantvelocity.standard import ConstantVelocityModel
from models.constantvelocity.vectorized import VectorizedConstantVelocityModel
from models.constantvelocity.stepwise import StepwiseVectorizedConstantVelocityModel
from models.constantvelocity.stepwise_stepbeta import StepwiseVectorizedConstantVelocityModel as MultiBetaStepwise
from models.constantvelocity.standard_gt import GTConstantVelocityModel  
from models.constantvelocity.stepwise_gt import GTStepwiseConstantVelocityModel
from models.constantvelocity.stepwise_gt_stepbeta import GTStepwiseConstantVelocityModel as GTMultiBetaStepwise

## Training Gym's
from traintestgyms.ignitegym import TrainTestGym

## Plots
from utils.report_plots.training_tracking import plotres, plotgrad
from utils.report_plots.compare_intensity_rates import compare_intensity_rates_plot
from utils.report_plots.event_distribution import plot_event_dist

## Utils
from utils.visualize.animation import animate
from utils.nodes.remove_drift import center_z0, remove_v_drift, remove_rotation





if __name__ == '__main__':

    ### Parse Arguments for running in terminal
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--seed', '-seed', default=1, type=int)
    arg_parser.add_argument('--device', '-device', default='cpu', type=str)
    arg_parser.add_argument('--learning_rate', '-LR', default=0.025, type=float)
    arg_parser.add_argument('--num_epochs', '-NE', default=5, type=int)
    arg_parser.add_argument('--train_batch_size', '-TBS', default=-1, type=int)
    arg_parser.add_argument('--real_data', '-RD', default=0, type=int)
    arg_parser.add_argument('--dataset_number', '-DS', default=1, type=int)
    arg_parser.add_argument('--training_type', '-TT', default=2, type=int)
    arg_parser.add_argument('--vectorized', '-VEC', default=2, type=int)
    arg_parser.add_argument('--remove_node_pairs_b', '-T1', default=0, type=int)
    arg_parser.add_argument('--remove_interactions_b', '-T2', default=0, type=int)
    arg_parser.add_argument('--steps', '-steps', default=None, type=int)
    arg_parser.add_argument('--step_beta', '-SB', action='store_true')
    arg_parser.add_argument('--animation', '-ani', action='store_true')
    arg_parser.add_argument('--animation_time_points', '-ATP', default=500, type=int)
    arg_parser.add_argument('--velocity_gamma_regularization', '-VGR', default=None, type=float)
    arg_parser.add_argument('--wandb_entity', '-WE', default='augustsemrau', type=str)
    arg_parser.add_argument('--wandb_project', '-WP', default='TGML11', type=str)
    arg_parser.add_argument('--wandb_run_name', '-WRN', default=None, type=str)
    arg_parser.add_argument('--wandb_group', '-WG', default=None, type=str)
    args = arg_parser.parse_args()

    ## Set all input arguments
    seed = args.seed
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    train_batch_size = args.train_batch_size
    dataset_number = args.dataset_number
    training_type = args.training_type
    vectorized = args.vectorized
    remove_node_pairs_b = args.remove_node_pairs_b
    remove_interactions_b = args.remove_interactions_b
    device = args.device
    real_data = args.real_data
    num_steps = args.steps
    step_beta = args.step_beta
    animation = args.animation
    animation_time_points = args.animation_time_points
    velocity_gamma_regularization = args.velocity_gamma_regularization
    wandb_entity= args.wandb_entity
    wandb_project = args.wandb_project
    wandb_run_name = args.wandb_run_name
    wandb_group = args.wandb_group

    ## Seeding of model run
    np.random.seed(seed)
    torch.manual_seed(seed)
    np.seterr(all='raise')

    ## Device
    print(f'Running with pytorch device: {device}')
    torch.pi = torch.tensor(torch.acos(torch.zeros(1)).item()*2).to(device)
    torch.eps = torch.tensor(np.finfo(float).eps).to(device) #Adding eps to avoid devision by 0 



    ### Data: Either synthetically generated data, or loaded real world data
    if real_data == 0:
        ### Defining parameters for synthetic data generation
        z0, v0, true_beta, model_beta, max_time = get_initial_parameters(dataset_number=dataset_number, vectorized=vectorized)
        # Adjusting z0 and v0
        z0 = torch.from_numpy(z0)
        z0, v0 = center_z0(z0), remove_v_drift(v0)
        z0, v0 = remove_rotation(z0, v0)
        z0 = z0.numpy()
        if step_beta:
            #Use a beta parameter for each step in the model
            model_beta = np.asarray([model_beta]*num_steps)
        num_nodes = z0.shape[0]
        if num_steps == None:
            num_steps = v0.shape[2]
        print(f"Number of nodes: {num_nodes} \nz0: \n{z0} \nv0: \n{v0} \nTrue Beta: {true_beta} \nModel initiated Beta: {model_beta} \nMax time: {max_time}\nNumber of steps to fit: {num_steps}")

        ## Initialize data builder for simulating node interactions from known Poisson Process
        if vectorized != 2:
            simulator = ConstantVelocitySimulator(starting_positions=z0, velocities=v0, T=max_time, beta=true_beta, seed=seed)
            data_builder = DatasetBuilder(simulator, device=device)
            dataset_full = data_builder.build_dataset(num_nodes, time_column_idx=2)
        elif vectorized == 2:
            simulator = StepwiseConstantVelocitySimulator(starting_positions=z0, velocities=v0, max_time=max_time, beta=true_beta, seed=seed)
            data_builder = StepwiseDatasetBuilder(simulator=simulator, device=device, normalization_max_time=None)
            dataset_full = data_builder.build_dataset(num_nodes, time_column_idx=2)
    else:
        print(f"Loading real dataset number {dataset_number}")

        dataset_full, num_nodes, model_beta = load_real_dataset(dataset_number=dataset_number, debug=0)
        z0, v0, true_beta, = None, None, None 
        if num_steps == None:
            num_steps = 2
        max_time = max(dataset_full[:,2])

    dataset_size = len(dataset_full)
    num_dyads = (num_nodes * (num_nodes - 1)) / 2

    ### Testing sets: Either remove entire noode pairs, 10% of events, or both
    if remove_node_pairs_b == 1 and remove_interactions_b == 0:
        dataset, removed_node_pairs = remove_node_pairs(dataset=dataset_full, num_nodes=num_nodes, percentage=0.10, device=device)
        removed_interactions = None
    elif remove_node_pairs_b == 0 and remove_interactions_b == 1:
        dataset, removed_interactions = remove_interactions(dataset=dataset_full, percentage=0.1, device=device)
        removed_node_pairs = None
    elif remove_node_pairs_b == 1 and remove_interactions_b == 1:
        dataset_removed_nodes, removed_node_pairs = remove_node_pairs(dataset=dataset_full, num_nodes=num_nodes, percentage=0.05, device=device)
        dataset, removed_interactions = remove_interactions(dataset=dataset_removed_nodes, percentage=0.1, device=device)
    else:
        dataset, removed_node_pairs, removed_interactions = dataset_full, None, None

    ## Compute size of dataset and find training batch size
    training_set_size = len(dataset)

    ## Batch
    train_batch_size = train_batch_size if train_batch_size > 0 else training_set_size

    print(f"\nLength of entire dataset: {dataset_size}\nLength of training set: {training_set_size}\nTrain batch size: {train_batch_size}\n")



    ### WandB initialization
    ## Set input parameters as config for Weights and Biases
    wandb_config = {'seed': seed,
                    'device': device,
                    'learning_rate': learning_rate,
                    'vectorized': vectorized,  # 0 = non-vectorized, 1 = vectorized, 2 = stepwise
                    'training_type': training_type,  # 0 = non-sequential training, 1 = sequential training
                    'num_epochs': num_epochs,
                    'max_time': max_time,
                    'num_nodes': num_nodes,
                    'dataset_size': dataset_size,
                    'remove_nodepairs': remove_node_pairs_b,
                    'remove_interactions': remove_interactions_b,
                    'true_beta': true_beta,
                    'model_beta': model_beta,
                    'true_z0': z0,
                    'true_v0': v0,
                    'num_steps': num_steps,
                    'train_batch_size': train_batch_size,
                    'velocity_gamma_regularization': velocity_gamma_regularization
                    }

    ## Initialize WandB for logging config and metrics
    wandb.init(project=wandb_project, name=wandb_run_name, 
                entity=wandb_entity, group=wandb_group, config=wandb_config)

    ## Plot and log event distribution
    plot_event_dist(dataset=dataset_full, wandb_handler=wandb)

    wandb.log({'training_set_size': training_set_size, 'removed_node_pairs': removed_node_pairs, 'train_batch_size': train_batch_size, 'beta': model_beta})





    ### Setup Model: Either non-vectorized, vectorized or stepwise
    if vectorized == -1:
        model = NoDynamicsModel(n_points=num_nodes, beta=model_beta).to(device)
    if vectorized == 0:
        model = ConstantVelocityModel(n_points=num_nodes, beta=model_beta).to(device)
    elif vectorized == 1:
        model = VectorizedConstantVelocityModel(n_points=num_nodes, beta=model_beta, device=device, z0=z0, v0=v0, true_init=True).to(device)
    elif vectorized == 2:
        last_time_point = dataset[:,2][-1].item()
        if isinstance(model_beta, np.ndarray):
            model = MultiBetaStepwise(n_points=num_nodes, beta=model_beta, steps=num_steps, max_time=last_time_point, 
                                        device=device, z0=z0, v0=v0, true_init=False).to(device)
        else:
            model = StepwiseVectorizedConstantVelocityModel(n_points=num_nodes, beta=model_beta, steps=num_steps, 
                            max_time=last_time_point, device=device, z0=z0, v0=v0, v0_init=training_type, 
                            gamma=velocity_gamma_regularization).to(device, dtype=torch.float32)
              
    ## Optimizer is initialized here, Adam is used
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    ### Model training: Either non-sequential or sequential
    metrics = {'avg_train_loss': [], 'beta_est': []}
    
    ## Non-sequential model training
    if training_type == 0:
        model.z0.requires_grad, model.v0.requires_grad, model.beta.requires_grad = True, True, True
        gym = TrainTestGym(dataset=dataset, 
                            model=model, 
                            device=device, 
                            batch_size=train_batch_size, 
                            optimizer=optimizer, 
                            metrics=metrics, 
                            time_column_idx=2,
                            wandb_handler = wandb,
                            num_dyads=num_dyads)
        gym.train_test_model(epochs=num_epochs)
        
    ## Sequential model training
    elif training_type == 1:
        model.z0.requires_grad, model.v0.requires_grad, model.beta.requires_grad = False, False, False
        gym = TrainTestGym(dataset=dataset, 
                            model=model, 
                            device=device, 
                            batch_size=train_batch_size, 
                            optimizer=optimizer, 
                            metrics=metrics, 
                            time_column_idx=2,
                            wandb_handler = wandb)
        for i in range(3):
            if i == 0:
                model.z0.requires_grad = True  # Learn Z next
            elif i == 1:
                model.v0.requires_grad = True  # Learn V last
            elif i == 2:
                model.beta.requires_grad = True  # Learn beta first
                
            gym.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            gym.train_test_model(epochs=int(num_epochs/3))
    
    ## Baseline of no velocities
    elif training_type == 2:
        model.z0.requires_grad, model.v0.requires_grad, model.beta.requires_grad = True, False, True
        gym = TrainTestGym(dataset=dataset, 
                            model=model, 
                            device=device, 
                            batch_size=train_batch_size, 
                            optimizer=optimizer, 
                            metrics=metrics, 
                            time_column_idx=2,
                            wandb_handler = wandb,
                            num_dyads=num_dyads)
        gym.train_test_model(epochs=num_epochs)



    ### Results generation
    ## Build non-vectorized final and ground truth models
    if device == 'cuda':
        result_z0 = model.z0.detach().clone()
        result_v0 = model.v0.detach().clone()
        result_beta = model.beta.detach().clone()
        train_t = np.linspace(0, dataset_full.cpu()[-1][2])
    else:
        result_z0 = model.z0.detach().clone()
        result_v0 = model.v0.detach().clone()
        result_beta = model.beta.detach().clone()
        train_t = np.linspace(0, dataset_full[-1][2])
    
    # Save learned model parameters to weights and biases
    torch.save(result_z0, os.path.join(wandb.run.dir, "final_z0.pt"))
    torch.save(result_v0, os.path.join(wandb.run.dir, "final_v0.pt"))
    wandb.save(os.path.join(wandb.run.dir, "final_z0.pt"))
    wandb.save(os.path.join(wandb.run.dir, "final_v0.pt"))


    ## Data generation is diffrerent for synthetic and RL datasets
    if real_data == 0:
        if vectorized != 2: 
            result_model = GTConstantVelocityModel(n_points=num_nodes, z=result_z0 , v=result_v0 , beta=result_beta).to(device, dtype=torch.float32)
            gt_model = GTConstantVelocityModel(n_points=num_nodes, z=z0, v=v0, beta=true_beta).to(device, torch.float32)
        elif vectorized == 2:
            if isinstance(model_beta, np.ndarray):
                result_model = GTMultiBetaStepwise(n_points=num_nodes, z=result_z0, v=result_v0, beta=result_beta,
                                                                steps=num_steps, max_time=max_time, device=device).to(device, dtype=torch.float32)
                gt_model = GTMultiBetaStepwise(n_points=num_nodes, z=torch.from_numpy(z0), v=v0.clone().detach(), beta=torch.tensor([true_beta]*v0.shape[2]), 
                                                                steps=v0.shape[2], max_time=max_time, device=device).to(device, dtype=torch.float32)
            else:
                result_model = GTStepwiseConstantVelocityModel(n_points=num_nodes, z=result_z0, v=result_v0, beta=result_beta,
                                                                steps=num_steps, max_time=max_time, device=device).to(device, dtype=torch.float32)
                gt_model = GTStepwiseConstantVelocityModel(n_points=num_nodes, z=torch.from_numpy(z0), v=v0.clone().detach(), beta=true_beta, 
                                                                steps=v0.shape[2], max_time=max_time, device=device).to(device, dtype=torch.float32)
        
        ## Compare intensity rates of removed node pairs
        if remove_node_pairs_b == 1: 
            num = 0
            for removed_node_pair in removed_node_pairs:
                num += 1
                plot_num = '_removed_dyad' + str(num)
                compare_intensity_rates_plot(train_t=train_t, result_model=result_model, gt_model=gt_model, nodes=[list(removed_node_pair)], wandb_handler=wandb, num=plot_num)
    
    else:
        if vectorized != 2: 
            result_model = GTConstantVelocityModel(n_points=num_nodes, z=result_z0 , v=result_v0 , beta=result_beta)
        elif vectorized == 2:
            if isinstance(model_beta, np.ndarray):
                result_model = GTMultiBetaStepwise(n_points=num_nodes, z=result_z0, v=result_v0, beta=result_beta,
                                                                steps=num_steps, max_time=max_time, device=device)
            else:
                result_model = GTStepwiseConstantVelocityModel(n_points=num_nodes, z=result_z0, v=result_v0, beta=result_beta,
                                                                steps=num_steps, max_time=max_time, device=device)

    
    if animation:
        print(f'Creating animation of latent node positions on {animation_time_points} time points')
        animate(model, t_start=0, t_end=max_time, num_of_time_points=animation_time_points, device=device, wandb_handler=wandb)
    

    ## Compute ROC AUC for removed interactions
    if remove_interactions_b == 1:
        if real_data == 0:
            acc_removed_interactions(removed_interactions=removed_interactions, num_nodes=num_nodes, result_model=result_model, wandb_handler=wandb, gt_model=gt_model)
        else:
            acc_removed_interactions(removed_interactions=removed_interactions, num_nodes=num_nodes, result_model=result_model, wandb_handler=wandb, gt_model=None)

    if real_data == 0:
        ## Compute ground truth training loss for gt model and log  
        wandb.log({'gt_train_NLL': ((gt_model.forward(data=dataset_full.to(device), t0=dataset_full[0,2].item(), tn=dataset_full[-1,2].item()) / num_dyads))})
        ## Make intensity rate comparison plots for the synthetic datasets
        if dataset_number == 1:
            compare_intensity_rates_plot(train_t=train_t, result_model=result_model, gt_model=gt_model, nodes=[[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]], wandb_handler=wandb, num=1)
        elif dataset_number == 2:
            compare_intensity_rates_plot(train_t=train_t, result_model=result_model, gt_model=gt_model, nodes=[[0,1], [0,2], [0,3], [0,4], [3,4]], wandb_handler=wandb, num=1)
            compare_intensity_rates_plot(train_t=train_t, result_model=result_model, gt_model=gt_model, nodes=[[1,2], [1,3], [1,4], [2,3], [2,4]], wandb_handler=wandb, num=2)
        elif dataset_number == 3:
            compare_intensity_rates_plot(train_t=train_t, result_model=result_model, gt_model=gt_model, nodes=[[0,1], [0,21], [0,102], [0,143]], wandb_handler=wandb, num=1)
            compare_intensity_rates_plot(train_t=train_t, result_model=result_model, gt_model=gt_model, nodes=[[20,11], [95, 106], [45, 150], [77, 88]], wandb_handler=wandb, num=2)
            compare_intensity_rates_plot(train_t=train_t, result_model=result_model, gt_model=gt_model, nodes=[[13,120], [66, 133], [99, 144], [101, 102]], wandb_handler=wandb, num=3)