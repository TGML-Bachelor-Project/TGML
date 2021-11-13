import torch
import numpy as np
from utils.nodes.remove_drift import remove_v_drift, center_z0, remove_rotation
from ignite.engine import Engine
from ignite.engine import Events
from torch.utils.data import DataLoader
from ignite.contrib.handlers.tqdm_logger import ProgressBar

class TrainTestGym:
    def __init__(self, dataset, model, device, batch_size,
                    optimizer, metrics,
                    time_column_idx, wandb_handler) -> None:

        ## Split dataset and intiate dataloder
        len_training_set = int(len(dataset))
        train_data = dataset[:len_training_set]

        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle= False)


        self.model = model
        self.model_state = self.model.state_dict()
        self.device = device
        self.optimizer = optimizer
        self.trainer = Engine(self.__train_step)
        self.trainer.t_start = 0.0

        self.time_column_idx = time_column_idx

        ## Every Epoch print z, v and beta value to terminal for inspection
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), lambda: print(f'z0: {model.z0}  \
                                                                                        \n v0: {model.v0} \
                                                                                       \n beta: {model.beta}'))

        ### Metrics of training
        self.wandb_handler = wandb_handler
        self.metrics = metrics
        self.temp_metrics = {'train_loss': [], 'beta_est': []}  # Used for computing average of losses

        ## Keep count of epoch
        self.epoch_count = []
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), lambda: self.epoch_count.append(0))

        ## Every Epoch compute mean of training and test losses for loggin
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), lambda: self.metrics['avg_train_loss'].append(np.sum(self.temp_metrics['train_loss']) / len_training_set))
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), lambda: self.metrics['beta_est'].append(model.beta.detach().clone()))

        ## Clear temp_metrics for next epoch
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), lambda: self.temp_metrics['train_loss'].clear())
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), lambda: self.temp_metrics['beta_est'].clear())

        ## Log metrics using WandB
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), lambda: wandb_handler.log({'Epoch': len(self.epoch_count),
                                                                                                    'beta': model.beta.detach().clone(),
                                                                                                    'avg_train_loss': self.metrics['avg_train_loss'][len(self.epoch_count)-1]}))
                                                                                                

        pbar = ProgressBar()
        pbar.attach(self.trainer)

    ### Training step
    def __train_step(self, engine, batch):
        batch = batch.to(self.device, dtype=torch.float32)

        if engine.t_start != 0:
            engine.t_start = batch[0,self.time_column_idx].item()

        #Adjust model parameters for nicer visualizations
        self.model_state['z0'], self.model_state['v0'] = remove_rotation(
                                                            center_z0(self.model.z0), remove_v_drift(self.model.v0))
        self.model.load_state_dict(self.model_state)

        self.model.train()
        self.optimizer.zero_grad()
        train_loglikelihood = self.model(batch, t0=engine.t_start,
                                            tn=batch[-1,self.time_column_idx].item())
        loss = - train_loglikelihood
        loss.backward()
        self.optimizer.step()
        self.temp_metrics['train_loss'].append(loss.item())
        self.temp_metrics['beta_est'].append(self.model.beta.detach().clone())
        if engine.t_start == 0:
            engine.t_start = 1 #change t_start to flag it for updates

        return loss.item()


    ### Train and evaluate the model for n epochs
    def train_test_model(self, epochs:int):
        print(f'Starting model training with {epochs} epochs')
        self.trainer.run(self.train_loader, max_epochs=epochs)
        # Adjust model params after last training
        self.model_state['z0'], self.model_state['v0'] = remove_rotation(
                                                            center_z0(self.model.z0), remove_v_drift(self.model.v0))
        self.model.load_state_dict(self.model_state)
        print('Completed model training')
