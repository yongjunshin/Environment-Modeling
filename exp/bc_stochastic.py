from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch
import copy

from exp.evaluation import *


class BehaviorCloning1TickStochasticTrainer:
    def __init__(self, device, sut, lr):
        self.device = device
        self.sut = sut
        self.loss_fn = torch.nn.MSELoss()
        self.lr = lr

    def train(self, model: torch.nn.Module, epochs: int, x: torch.tensor, y: torch.tensor, xt: torch.tensor, yt: torch.tensor) -> list:
        optimiser = torch.optim.Adam(model.parameters(), lr=self.lr)

        evaluation_results = []
        dl = DataLoader(dataset=TensorDataset(x, y), batch_size=516, shuffle=True)
        testing_dl = DataLoader(dataset=TensorDataset(xt, yt), batch_size=516, shuffle=True)

        # initial model
        evaluation_results.append(simulation_and_comparison(model, self.sut, testing_dl, self.device))

        for _ in tqdm(range(epochs), desc="Training"):
            ed_sum = torch.zeros((), device=self.device)
            dtw_sum = torch.zeros((), device=self.device)
            for _, (x_batch, y_batch) in enumerate(dl):
                model.train()
                bc_dist = model.get_distribution(x_batch)
                bc_target_y = y_batch[:, 0, [0]]
                loss = -bc_dist.log_prob(bc_target_y)

                optimiser.zero_grad()
                loss.mean().backward()
                optimiser.step()

            evaluation_results.append(simulation_and_comparison(model, self.sut, testing_dl, self.device))

        return evaluation_results


