from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch
import copy

from exp.evaluation import *


class BehaviorCloning1TickTrainer:
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

        # initial model evaluation
        evaluation_results.append(simulation_and_comparison(model, self.sut, testing_dl, self.device))

        for _ in tqdm(range(epochs), desc="Training"):
            for _, (x_batch, y_batch) in enumerate(dl):
                model.train()
                y_pred = model(x_batch)
                y_target = y_batch[:, 0, [0]]
                loss = self.loss_fn(y_target, y_pred)

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            evaluation_results.append(simulation_and_comparison(model, self.sut, testing_dl, self.device))

        return evaluation_results


