import time
import numpy as np
import torch
from ignite.engine import Engine
from ignite.engine import Events
from torch.utils.data import DataLoader
from ignite.contrib.handlers.tqdm_logger import ProgressBar

class SimonTrainTestGym:
    def __init__(self, dataset, 
                        model, 
                        device, 
                        batch_size,
                        training_portion, 
                        optimizer, 
                        metrics, 
                        time_column_idx,
                        num_epochs) -> None:
        
        last_training_idx = int(len(dataset)*training_portion)
        self.train_data = dataset[:last_training_idx]
        self.test_data = dataset[last_training_idx:]
        
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.metrics = metrics
        self.time_column_idx = time_column_idx
        self.num_epochs = num_epochs



    def batch_train_track_mse(self, res_gt, 
                            track_nodes, 
                            n_train, 
                            train_batches, 
                            test_data):
        training_losses = []
        test_losses = []
        mse_train_losses = []
        mse_test_losses = []

        track_dict = {
            "mse_train_losses":[], 
            "mse_test_losses":[],
            "bgrad":[],
            "vgrad":[],
            "zgrad":[]}

        tn_train = train_batches[-1][-1][2] # last time point in training data
        tn_test = test_data[-1][2] # last time point in test data
        n_test = len(test_data)


        for epoch in range(self.num_epochs):
            start_time = time.time()
            running_loss = 0.
            # sum_ratio = 0.
            start_t = torch.tensor([0.0])
            
            epoch_bgrad=[]
            epoch_zgrad=[]
            epoch_vgrad=[]

            for idx, batch in enumerate(train_batches):
                self.model.train()
                self.optimizer.zero_grad()
                output = self.model(batch, t0=start_t, tn=batch[-1][2])
                loss = - output
                loss.backward()

                #net.beta.grad = net.beta.grad/10.0
                #net.v0.grad = net.v0.grad / 10.0
                #clip_value=30.0
                #torch.nn.utils.clip_grad_norm_(net.parameters(), clip_value)

                batch_bgrad = torch.mean(torch.abs(self.model.beta.grad))
                batch_zgrad = torch.mean(torch.abs(self.model.z0.grad))
                batch_vgrad = torch.mean(torch.abs(self.model.v0.grad))
                epoch_bgrad.append(batch_bgrad)
                epoch_zgrad.append(batch_zgrad)
                epoch_vgrad.append(batch_vgrad)


                self.optimizer.step()

                running_loss += loss.item()
                # sum_ratio += ratio
                start_t = batch[-1][2]

            self.model.eval()
            with torch.no_grad():
                res_train = []
                res_test = []
                for ti in np.linspace(0, tn_train):
                    res_train.append(torch.exp(self.model.log_intensity_function(track_nodes[0], track_nodes[1], ti)))
                
                for ti in np.linspace(tn_train, tn_test):
                    res_test.append(torch.exp(self.model.log_intensity_function(track_nodes[0], track_nodes[1], ti)))
                
                res_train = torch.tensor(res_train)
                res_test = torch.tensor(res_test)

                mse_train = torch.mean((res_gt[0]-res_train)**2)
                mse_test = torch.mean((res_gt[1]-res_test)**2)
                
                test_output = self.model(test_data, t0=tn_train, tn=tn_test)
                test_loss = - test_output.item()
                    
            avg_train_loss = running_loss / n_train
            # avg_train_ratio = sum_ratio / len(train_batches)
            avg_test_loss = test_loss / n_test
            current_time = time.time()

            # if epoch == 0 or (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}")
            print(f"elapsed time: {current_time - start_time}" )
            print(f"train loss: {avg_train_loss}")
            print(f"test loss: {avg_test_loss}")
            print(f"mse train loss / n_train {mse_train/n_train}")
            print(f"mse test loss / n_test {mse_test/n_test}")
                #print(f"train event to non-event ratio: {avg_train_ratio.item()}")
                #print(f"test event to non-event-ratio: {test_ratio.item()}")
            print("State dict:")
            print(self.model.state_dict())
            
            training_losses.append(avg_train_loss)
            test_losses.append(avg_test_loss)
            track_dict["mse_train_losses"].append(mse_train)
            track_dict["mse_test_losses"].append(mse_test)
            track_dict["bgrad"].append(torch.mean(torch.tensor(epoch_bgrad)))
            track_dict["zgrad"].append(torch.mean(torch.tensor(epoch_zgrad)))
            track_dict["vgrad"].append(torch.mean(torch.tensor(epoch_vgrad)))
        
        return self.model, training_losses, test_losses, track_dict



    # def train_test_model(self, epochs:int):
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=5, T_mult=1)
    #     training_losses = []
    #     test_losses = []
    #     tn_train = self.train_data[-1][-1] # last time point in training data
    #     tn_test = self.test_data[-1][-1] # last time point in test data
    #     iters = len(train_loader)
    #     n_test = len(test_data)

    #     best_loss = torch.tensor([1e100])
    #     save_path = r"state_dicts/training_experiment/hospital_best_state.pth"

    #     for epoch in range(self.num_epochs):
    #         start_time = time.time()
    #         running_loss = 0.
    #         start_t = torch.tensor([0.0])
    #         for idx, batch in enumerate(train_loader):
    #             if (idx+1)==1 or (idx+1) % 100 == 0:
    #                 print(f"Batch {idx+1} of {len(train_loader)}")

    #             self.model.train()
    #             self.optimizer.zero_grad()

    #             output = self.model(batch, t0=start_t, tn=batch[-1][2])
    #             loss = - output
    #             loss.backward()

    #             clip_value = 30.0 
    #             torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)

    #             self.optimizer.step()
    #             scheduler.step(epoch + idx / iters)

    #             running_loss += loss.item()
    #             start_t = batch[-1][2]

    #         print(f"Epoch {epoch+1} Last LR: {scheduler.get_last_lr()}")

    #         self.model.eval()
    #         with torch.no_grad():
    #             test_output = net(test_data, t0=tn_train, tn=tn_test)
    #             test_loss = nll(test_output).item()

    #         avg_train_loss = running_loss / n_train
    #         avg_test_loss = test_loss / n_test
    #         current_time = time.time()

    #         if avg_test_loss < best_loss:
    #             print("Lowest test loss.")
    #             #print(f"Saved model state_dict to {save_path}.")
    #             #best_loss = avg_test_loss
    #             #torch.save(net.state_dict(), save_path)

    #         if epoch == 0 or (epoch+1) % 1 == 0:
    #             print(f"Epoch {epoch+1}")
    #             print(f"elapsed time: {current_time - start_time}" )
    #             print(f"train loss: {avg_train_loss}")
    #             print(f"test loss: {avg_test_loss}")

    #         training_losses.append(avg_train_loss)
    #         test_losses.append(avg_test_loss)

    #     return net, training_losses, test_losses, optimizer


    #     print(f'Starting model training with {epochs} epochs')
    #     self.trainer.run(self.train_loader, max_epochs=epochs)
    #     print('Completed model training')
