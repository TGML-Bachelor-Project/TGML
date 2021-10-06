import torch
from ignite.engine import Engine
from ignite.engine import Events

class TrainTestGym:
    def __init__(self, model, device, train_loader, val_loader,
                        optimizer, metrics, time_column_idx) -> None:
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.trainer = Engine(self.train_step)
        self.trainer.t_start = 0.
        self.evaluator = Engine(self.validation_step)
        self.evaluator.t_start = 0.
        self.optimizer = optimizer
        self.metrics = metrics
        self.time_column_idx = time_column_idx
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED(every=5), self.evaluate_model)

    def train_step(self, model, engine, batch):
        X = batch.to(self.device)

        model.train()
        self.optimizer.zero_grad()
        train_loglikelihood = model(X, t0=engine.t_start, tn=batch[-1][self.time_column_idx])
        loss = - train_loglikelihood
        loss.backward()
        self.optimizer.step()
        self.metrics['train_loss'].append(loss.item())
        self.metrics['Bias Term - Beta'].append(model.intensity_function.beta.item())
        engine.t_start = batch[-1][self.time_column_idx].to(self.device)

        return loss



    ### Evaluation setup
    def validation_step(self, model, engine, batch):
        model.eval()

        with torch.no_grad():
            X = batch.to(self.device)
            test_loglikelihood = model(X, t0=engine.t_start, tn=batch[-1][self.time_column_idx].item())
            test_loss = - test_loglikelihood
            # optimizer.step()
            self.metrics['test_loss'].append(test_loss.item())
            engine.t_start = batch[-1][self.time_column_idx]
            return test_loss


    ### Handlers
    def evaluate_model(self):
        self.evaluator.run(self.val_loader)


    def evaluate_model(self, epochs:int):
        print(f'Starting model training with {epochs} epochs')
        self.trainer.run(self.train_loader, max_epochs=epochs)
        print('Completed model training')

        # Print model params
        model_z0 = self.model.z0.cpu().detach().numpy() 
        model_v0 = self.model.v0.cpu().detach().numpy()
        print(f'Beta: {self.model.intensity_function.beta.item()}')
        print(f'Z: {model_z0}')
        print(f'V: {model_v0}')