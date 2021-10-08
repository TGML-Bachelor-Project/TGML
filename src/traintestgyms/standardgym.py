import torch
from ignite.engine import Engine
from ignite.engine import Events
from torch.utils.data import DataLoader
from ignite.contrib.handlers.tqdm_logger import ProgressBar

class TrainTestGym:
    def __init__(self, dataset, model, device, batch_size,
                    training_portion, optimizer, metrics, time_column_idx) -> None:
        last_training_idx = int(len(dataset)*training_portion)
        self.train_data = dataset[:last_training_idx]
        self.test_data = dataset[last_training_idx:]
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.metrics = metrics
        self.time_column_idx = time_column_idx

    def train_test_model(self, epochs:int):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=5, T_mult=1)
        training_losses = []
        test_losses = []
        tn_train = self.train_data[-1][-1] # last time point in training data
        tn_test = self.test_data[-1][-1] # last time point in test data
        iters = len(train_loader)
        n_test = len(test_data)

        best_loss = torch.tensor([1e100])
        save_path = r"state_dicts/training_experiment/hospital_best_state.pth"

        for epoch in range(num_epochs):
            start_time = time.time()
            running_loss = 0.
            start_t = torch.tensor([0.0])
            for idx, batch in enumerate(train_loader):
                if (idx+1)==1 or (idx+1) % 100 == 0:
                    print(f"Batch {idx+1} of {len(train_loader)}")

                net.train()
                optimizer.zero_grad()

                output = net(batch, t0=start_t, tn=batch[-1][2])
                loss = nll(output)
                loss.backward()

                clip_value = 30.0 
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip_value)

                optimizer.step()
                scheduler.step(epoch + idx / iters)

                running_loss += loss.item()
                start_t = batch[-1][2]

            print(f"Epoch {epoch+1} Last LR: {scheduler.get_last_lr()}")

            net.eval()
            with torch.no_grad():
                test_output = net(test_data, t0=tn_train, tn=tn_test)
                test_loss = nll(test_output).item()

            avg_train_loss = running_loss / n_train
            avg_test_loss = test_loss / n_test
            current_time = time.time()

            if avg_test_loss < best_loss:
                print("Lowest test loss.")
                #print(f"Saved model state_dict to {save_path}.")
                #best_loss = avg_test_loss
                #torch.save(net.state_dict(), save_path)

            if epoch == 0 or (epoch+1) % 1 == 0:
                print(f"Epoch {epoch+1}")
                print(f"elapsed time: {current_time - start_time}" )
                print(f"train loss: {avg_train_loss}")
                print(f"test loss: {avg_test_loss}")

            training_losses.append(avg_train_loss)
            test_losses.append(avg_test_loss)

        return net, training_losses, test_losses, optimizer


        print(f'Starting model training with {epochs} epochs')
        self.trainer.run(self.train_loader, max_epochs=epochs)
        print('Completed model training')
