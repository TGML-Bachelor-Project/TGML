import time
from tqdm import tqdm

class TrainTestGym:
    def __init__(self, dataset, model, device, optimizer, metrics,
                    time_column_idx, wandb_handler, batch_size=None) -> None:
        
        self.train_data = dataset
        self.train_data_len = len(dataset)
        self.model = model
        self.device = device
        self.optimizer = optimizer

        self.time_column_idx = time_column_idx

        ### Metrics of training
        self.wandb_handler = wandb_handler
        self.metrics = metrics
        self.temp_metrics = {'train_loss': [], 'beta_est': []}  # Used for computing average of losses


    def _single_batch_train(self, epochs):
        epoch_count = 0
        training_losses = []
        tn_train = self.train_data[-1][2] # last time point in training data

        for epoch in tqdm(range(epochs)):
            self.model.train()
            self.optimizer.zero_grad()
            log_likelihood = self.model(self.train_data, t0=0, tn=tn_train)
            loss = -log_likelihood
            loss.backward()
            self.optimizer.step()

            epoch_count += 1
            training_losses.append(loss.item())
            self.metrics['avg_train_loss'].append(sum(training_losses) / self.train_data_len)
            beta = self.model.beta.item()
            self.metrics['beta_est'].append(beta)
            self.wandb_handler.log({'Epoch': epoch_count,
                                    'beta': beta,
                                    'avg_train_loss': self.metrics['avg_train_loss'][epoch_count-1]})

            print(f'z0: {self.model.z0} \n v0: {self.model.v0} \n beta: {self.model.beta}')
            time.sleep(0.01) #Sleep so print does not interfer with tqdm progress bar


    ### Train and evaluate the model for n epochs
    def train_test_model(self, epochs:int):
        print(f'Starting model training with {epochs} epochs')
        self._single_batch_train(epochs)
        print('Completed model training')