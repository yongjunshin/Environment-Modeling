from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt

from fieldTestDataset import FieldTestDataset
from soft_dtw_cuda import SoftDTW


class BehaviorCloningEpisodeTrainer:
    def __init__(self, device, sut):
        self.device = device
        self.sut = sut

    def train(self, model, epochs, train_dataloaders, test_dataloaders, loss_metric="mse"):
        if loss_metric == "mse":
            loss_fn = torch.nn.MSELoss()
        elif loss_metric == "mdtw":
            class MDTWLoss(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.sdtw = SoftDTW(use_cuda=True, gamma=0.1)

                def forward(self, y_pred, y):
                    diffs = self.sdtw(y_pred, y)

                    # if diffs[0] < 0:
                    #     plt.figure(figsize=(10, 5))
                    #     plt.plot(y_pred[0, :, [0]].cpu().detach().numpy(), label="y_pred")
                    #     plt.plot(y[0, :, [0]].cpu().detach().numpy(), label="y")
                    #     plt.legend()
                    #     plt.show()
                    return diffs.mean()

            loss_fn = MDTWLoss()
        else:
            print("[Error] Wrong history distance metric and parameters.")
            quit()
        optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
        training_loss = np.zeros(epochs)


        num_batch = sum([len(dl) for dl in train_dataloaders])
        for i in tqdm(range(epochs), desc="Epochs:"):
            for dataloader in train_dataloaders:
                for batch_idx, (x, y) in enumerate(dataloader):
                    model.train()
                    y_pred = torch.zeros(y.shape, device=self.device)
                    sim_x = x
                    for sim_idx in range(y.shape[1]):
                        y_pred_one_step = model(sim_x)
                        y_pred[:, sim_idx, 0] = y_pred_one_step[:, 0]

                        env_prediction = y_pred_one_step.cpu().detach().numpy()
                        sys_operations = self.sut.act_sequential(env_prediction)
                        next_x = np.concatenate((env_prediction, sys_operations), axis=1)
                        next_x = torch.tensor(next_x).to(device=self.device).type(torch.float32)
                        next_x = torch.reshape(next_x, (next_x.shape[0], 1, next_x.shape[1]))
                        sim_x = sim_x[:, 1:]
                        sim_x = torch.cat((sim_x, next_x), 1)

                    loss = loss_fn(y_pred, y)
                    training_loss[i] = training_loss[i] + loss.item()

                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()

                    plt.figure(figsize=(10, 5))
                    plt.plot(y_pred[0, :, [0]].cpu().detach().numpy(), label="y_pred")
                    plt.plot(y[0, :, [0]].cpu().detach().numpy(), label="y")
                    plt.legend()
                    plt.show()
            training_loss[i] = training_loss[i] / num_batch

            print(training_loss[i])


        return training_loss
