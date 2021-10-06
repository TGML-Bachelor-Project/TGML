import torch
from ignite.engine import Engine
from ignite.engine import Events
from data.builder import build_dataset
from torch.utils.data import DataLoader

class TrainTestGym:
    def __init__(self, num_of_nodes, events, model, device, 
                    training_portion, optimizer, metrics, time_column_idx) -> None:
        dataset = build_dataset(num_of_nodes, events, time_column_idx)
        last_training_idx = int(len(dataset)*training_portion)
        train_data = dataset[:last_training_idx]
        test_data = dataset[last_training_idx:]
        self.train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=False)
        self.val_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)
        self.model = model
        self.device = device
        self.trainer = Engine(self.__train_step)
        self.trainer.t_start = 0.
        self.evaluator = Engine(self.__validation_step)
        self.evaluator.t_start = 0.
        self.optimizer = optimizer
        self.metrics = metrics
        self.time_column_idx = time_column_idx
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED(every=5), lambda: self.evaluator.run(self.val_loader))

    def __train_step(self, engine, batch):
        X = batch.to(self.device)

        self.model.train()
        self.optimizer.zero_grad()
        train_loglikelihood = self.model(X, t0=engine.t_start, tn=batch[-1][self.time_column_idx])
        loss = - train_loglikelihood
        loss.backward()
        self.optimizer.step()
        self.metrics['train_loss'].append(loss.item())
        self.metrics['Bias Term - Beta'].append(self.model.intensity_function.beta.item())
        engine.t_start = batch[-1][self.time_column_idx].to(self.device)

        return loss

    ### Evaluation setup
    def __validation_step(self, engine, batch):
        self.model.eval()

        with torch.no_grad():
            X = batch.to(self.device)
            test_loglikelihood = self.model(X, t0=engine.t_start, tn=batch[-1][self.time_column_idx].item())
            test_loss = - test_loglikelihood
            # optimizer.step()
            self.metrics['test_loss'].append(test_loss.item())
            engine.t_start = batch[-1][self.time_column_idx]
            return test_loss


    def train_test_model(self, epochs:int):
        print(f'Starting model training with {epochs} epochs')
        self.trainer.run(self.train_loader, max_epochs=epochs)
        print('Completed model training')
