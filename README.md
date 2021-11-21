# TGML - Scalable Machine Learning for Temporal Dynamic Graphs
This repository contains the code written for reproducing the BSc thesis *Scalable Machine Learning for Temporal Dynamic Graphs* written by August Semrau Andersen and William Diedrichsen Marstrand at The Technical University of Denmark.

The project is supervised by Morten Mørup, Nikolaos Nakis, and Abdulkadir Çelikkanat from DTU Compute at The Technical University of Denmark.

## Description
The project contains the implementation of the models used in the project. This includes the Constant Velocity Model (CVM) on which this project extends and the Stepwise Constant Velocity Model (SCVM). A couple of ground truth models and a no dynamics model is also implemented for evaluation purposes.

To see some animations created using the code, checkout: https://tgml-bachelor-project.github.io/

## Folder Structure
The `src` folder contains all source code for the project. Inside this folder is:
* `data` - Contains all data for the project and the code for loading the data (some data might need to be unzipped before running the code)
* `models` - Contains all model code for the project
* `traintestgyms` - Contains the model training logic
* `utils` - Contains helper functions used throughout the code base

Finally the `src` contains the `main.py` script which is the primary entry point for running the project code through the commandline.

## Running the Code
**The project uses [Weights and Biases](https://wandb.ai) as logging framework** <br>
To run the code the user therefore has to create an account and use the platform. <br>
Weights and Biases supports free accounts that are easily created.


All required python packages and their correct versions can be installed using the `requirements.txt` script in the root of this project.

The code is run through the commandline using the `src/main.py` script. The following CLI arguments are supported:
```
    Options:                          Description:
    
    --seed:                           The random see for running the code
    
    --device:                         Device for torch i.e. cpu or cuda
    
    --learning_rate:                  Model learning rate
    
    --num_epochs:                     Number of training epochs
    
    --train_batch_size:               Model training batch size
    
    --real_data:                      Flag for data type. Choose 1 for real data and 0 for synthesized data
    
    --dataset_number:                 Id of the dataset to use. The datasets and their ids can be found in the scripts in the 'data' folder
    
    --vectorized:                     Defines the type of model used. -1 is 'No dynamics', 0 is 
                                      'Non-vectorized CVM', 1 is 'Vectorized CVM', and 2 is actual SCVM model proposed by this project
                  
    --remove_node_pairs_b:            Flag, set to 1 to perform remove node pair(dyad) test
    
    --remove_interactions_b:          Flag, set to 1 to perform remove event interaction test
    
    --steps:                          Number of velocity steps in the model. Only makes sense to use with the SCVM model.
    
    --keep_rotation:                  Flag for keeping rotation i.e. not perform the rotation position correction. 
                                      Do not give a number simply use --keep_rotation to activate this param
                     
    --animation:                      Flag to create an animation of the model fitting after training. 
                                      Also, just use --animation to activate this param
                 
    --animation_time_points:          Number of time points to use in the animation. Default i 1500
    
    --velocity_gamma_regularization:  Regularization parameter for the stepwise velocity change regularization. Default is 0.
    
    --wandb_entity:                   User name for the Weights and Biases account to use for logging (this is required to run the code)
    
    --wandb_project:                  Name of the Weights and Biases project to save the logging to (this is required to run the code)
    
    --wandb_run_name:                 Name of the specific run which will be save in Weights and Biases 
                                      (this is optional. If not set wandb creates a random name)
                      
    --wandb_group:                    Name of a group under which Weights and Biases will save the loggings in the project. 
                                      (This is optional. But, very nice for keeping track of runs)
```
