from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import torch
import copy

from exp.evaluation import *


class BehaviorCloningEpisodeTrainer:
    def __init__(self, device, sut, lr):
        self.device = device
        self.sut = sut
        self.loss_fn = torch.nn.MSELoss()
        self.lr = lr

    def train(self, model: torch.nn.Module, epochs: int, x: torch.tensor, y: torch.tensor, xt: torch.tensor, yt: torch.tensor) -> list:
        optimiser = torch.optim.Adam(model.parameters(), lr=self.lr)

        evaluation_results = []
        dl = DataLoader(dataset=TensorDataset(x, y), batch_size=512, shuffle=True)
        testing_dl = DataLoader(dataset=TensorDataset(xt, yt), batch_size=512, shuffle=True)

        # initial model
        evaluation_results.append(simulation_and_comparison(model, self.sut, testing_dl, self.device))

        for _ in tqdm(range(epochs), desc="Training"):
            ed_sum = torch.zeros((), device=self.device)
            dtw_sum = torch.zeros((), device=self.device)
            for _, (x_batch, y_batch) in enumerate(dl):
                model.train()
                y_pred = torch.zeros(y_batch.shape, device=self.device)
                sim_x = x_batch
                for sim_idx in range(y_batch.shape[1]):
                    # action choice
                    action = model(sim_x.detach())

                    # state transition
                    sys_operations = self.sut.act_sequential(action.detach().cpu().numpy())
                    sys_operations = torch.tensor(sys_operations).to(device=self.device).type(torch.float32)
                    next_x = torch.cat((action, sys_operations), dim=1)
                    next_x = torch.reshape(next_x, (next_x.shape[0], 1, next_x.shape[1]))
                    time_col = torch.reshape(y_batch[:, sim_idx, 2], (next_x.shape[0], 1, 1))
                    next_x = torch.cat((next_x, time_col), dim=2)
                    sim_x = sim_x[:, 1:]
                    sim_x = torch.cat((sim_x, next_x), dim=1)
                    y_pred[:, sim_idx] = sim_x[:, -1]

                loss = batch_dynamic_time_warping(y_pred[:,:,[0]], y_batch[:,:,[0]])
                optimiser.zero_grad()
                loss.mean().backward()
                optimiser.step()

            evaluation_results.append(simulation_and_comparison(model, self.sut, testing_dl, self.device))

        return evaluation_results