from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch
import copy

from exp.evaluation import *
from exp.util import episode_to_datapoints


class BehaviorCloning1TickTrainer:
    def __init__(self, device, sut, lr):
        self.device = device
        self.sut = sut
        self.loss_fn = torch.nn.MSELoss()
        self.lr = lr

    def train(self, model: torch.nn.Module, epochs: int, x: torch.tensor, y: torch.tensor, xt: list, yt: list, episode_length: int) -> list:
        optimiser = torch.optim.Adam(model.parameters(), lr=self.lr)

        evaluation_results = []

        x_training_datapoints, y_training_datapoints = episode_to_datapoints(x, y)
        dl = DataLoader(dataset=TensorDataset(x_training_datapoints, y_training_datapoints), batch_size=512, shuffle=True)
        #testing_dl = DataLoader(dataset=TensorDataset(xt, yt), batch_size=512, shuffle=True)

        # initial model evaluation
        evaluation_results.append(simulation_and_comparison_with_multiple_testing_dataset(model, self.sut, xt, yt, self.device))

        for _ in tqdm(range(epochs), desc="Training"):
            for _, (x_batch, y_batch) in enumerate(dl):
                model.train()
                y_pred = model(x_batch)
                y_target = y_batch[:, 0, [0]]
                loss = self.loss_fn(y_target, y_pred)

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            evaluation_results.append(simulation_and_comparison_with_multiple_testing_dataset(model, self.sut, xt, yt, self.device))

        return evaluation_results


