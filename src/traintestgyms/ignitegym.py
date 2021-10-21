import numpy as np
import torch
from ignite.engine import Engine
from ignite.engine import Events
from torch.utils.data import DataLoader
from ignite.contrib.handlers.tqdm_logger import ProgressBar

class TrainTestGym:
    def __init__(self, dataset, model, device, batch_size,
                    training_portion, optimizer, metrics, 
                    time_column_idx, wandb_handler) -> None:
        
        len_training_set = int(len(dataset)*training_portion)
        len_test_set = int(len(dataset) - len_training_set)
        train_data = dataset[:len_training_set]
        test_data = dataset[len_training_set:]

        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle= False)
        self.val_loader = DataLoader(test_data, batch_size=batch_size, shuffle= False)
        self.model = model
        self.device = device
        self.trainer = Engine(self.__train_step)
        self.trainer.t_start = 0.0
        self.evaluator = Engine(self.__validation_step)
        self.evaluator.t_start = 0.0
        self.optimizer = optimizer
        self.metrics = metrics
        self.temp_metrics = {'train_loss': [], 'test_loss': [], 'beta_est': []}  # Used for computing average of losses
        self.time_column_idx = time_column_idx
        self.wandb_handler = wandb_handler

        
        ## Every Epoch run evaluator
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), lambda: self.evaluator.run(self.val_loader))
        ## Every Epoch print z, v and beta value to terminal for inspection
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), lambda: print(f'z0: {model.z0}  \
                                                                                        \n v0: {model.v0} \
                                                                                       \n beta: {model.beta}'))

        
        ## Keep count of epoch
        self.epoch_count = []
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), lambda: self.epoch_count.append(0))
        ## Every Epoch compute mean of training and test losses for loggin
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), lambda: self.metrics['avg_train_loss'].append(np.sum(self.temp_metrics['train_loss']) / len_training_set))
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), lambda: self.metrics['avg_test_loss'].append(np.sum(self.temp_metrics['test_loss']) / len_test_set))
        # self.trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), lambda: metrics['beta_est'].append(torch.mean(self.temp_metrics['beta_est'])))
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), lambda: self.metrics['beta_est'].append(model.beta.item()))

        ## Clear temp_metrics for next epoch
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), lambda: self.temp_metrics['train_loss'].clear())
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), lambda: self.temp_metrics['test_loss'].clear())
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), lambda: self.temp_metrics['beta_est'].clear())


        
        
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), lambda: wandb_handler.log({'Epoch': len(self.epoch_count), 
                                                                                                    'beta': model.beta.item(), 
                                                                                                    'avg_train_loss': self.metrics['avg_train_loss'][len(self.epoch_count)-1],
                                                                                                    'avg_test_loss': self.metrics['avg_test_loss'][len(self.epoch_count)-1]}))


        

        pbar = ProgressBar()
        pbar.attach(self.trainer)

    ### Training step
    def __train_step(self, engine, batch):
        batch = batch.to(self.device)
        
        engine.t_start = 0 if engine.t_start == 0 else batch[0,self.time_column_idx].item()
        self.model.train()
        self.optimizer.zero_grad()
        train_loglikelihood = self.model(batch, t0=engine.t_start, 
                                            tn=batch[-1,self.time_column_idx].item())
        loss = - train_loglikelihood
        loss.backward()
        self.optimizer.step()
        self.temp_metrics['train_loss'].append(loss.item())
        self.temp_metrics['beta_est'].append(self.model.beta.item())
        if engine.t_start == 0:
            engine.t_start = 1 #change t_start to flag it for updates

        return loss.item()

    ### Evaluation setup
    def __validation_step(self, engine, batch):
        self.model.eval()

        with torch.no_grad():
            X = batch.to(self.device)
            engine.t_start = 0 if engine.t_start == 0. else batch[0,self.time_column_idx]
            test_loglikelihood = self.model(X, t0=engine.t_start, 
                                                tn=batch[-1,self.time_column_idx].item())
            test_loss = - test_loglikelihood
            self.temp_metrics['test_loss'].append(test_loss.item())
            if engine.t_start == 0:
                engine.t_start = 1  
            return test_loss.item()


    ### Train and evaluate the model for n epochs
    def train_test_model(self, epochs:int):
        print(f'Starting model training with {epochs} epochs')
        self.trainer.run(self.train_loader, max_epochs=epochs)
        print('Completed model training')
