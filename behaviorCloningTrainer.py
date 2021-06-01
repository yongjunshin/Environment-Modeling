from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np

from fieldTestDataset import FieldTestDataset


class BehaviorCloningTrainer:
    def __init__(self, device, sut):
        self.device = device
        self.sut = sut

    def train(self, model, epochs, train_dataloaders, test_dataloaders, dagger=False, max_dagger=50, dagger_threshold=0.01, dagger_batch_size=10000):
        loss_fn = torch.nn.MSELoss()
        optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
        training_loss = np.zeros(epochs)

        dagger_dataloaders = []
        dagger_count = 0
        for i in tqdm(range(epochs), desc="Epochs:"):
            num_batch = sum([len(dl) for dl in train_dataloaders])
            dagger_flag = False

            for dataloader in train_dataloaders:
                new_xs = []
                new_ys = []

                for batch_idx, (x, y) in enumerate(dataloader):
                    model.train()
                    y_pred = model(x)
                    loss = loss_fn(y, y_pred)
                    training_loss[i] = training_loss[i] + loss.item()

                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()

                    if dagger and dagger_count < max_dagger and loss < dagger_threshold:
                        dagger_flag = True
                        new_x, new_y = self.dagger(train_dataloader=dataloader, x=x, y_pred=y_pred)
                        new_xs.append(new_x)
                        new_ys.append(new_y)

                if len(new_xs) > 0:
                    new_x_tensor = torch.cat(new_xs, dim=0)
                    new_y_tensor = torch.cat(new_ys, dim=0)
                    dagger_dataloaders.append(DataLoader(dataset=FieldTestDataset(
                        new_x_tensor, new_y_tensor), batch_size=dagger_batch_size, shuffle=False))

            num_batch = num_batch + sum([len(dl) for dl in dagger_dataloaders])
            for dataloader in dagger_dataloaders:
                for batch_idx, (x, y) in enumerate(dataloader):
                    model.train()
                    y_pred = model(x)
                    loss = loss_fn(y, y_pred)
                    training_loss[i] = training_loss[i] + loss.item()

                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()

            training_loss[i] = training_loss[i]/num_batch
            if dagger_flag:
                dagger_count = dagger_count + 1

        return training_loss, dagger_count


    def dagger(self, train_dataloader, x, y_pred, recent_weight=0.9):
        env_prediction = y_pred.cpu().detach().numpy()

        sys_operations = np.zeros(len(env_prediction))
        for i in range(len(sys_operations)):
            sys_operations[i] = self.sut.act(env_prediction[i])
        sys_operations = torch.Tensor([sys_operations]).to(device=self.device)
        sys_operations = torch.reshape(sys_operations, (sys_operations.shape[1], 1))
        env_prediction = torch.Tensor(env_prediction).to(device=self.device)
        error_data = torch.cat((env_prediction[:], sys_operations[:]), 1)
        error_data = torch.reshape(error_data, (error_data.shape[0], 1, error_data.shape[1]))

        new_x = x[:-1, 1:, :]
        new_x = torch.cat([new_x, error_data[:-1]], dim=1)

        def input_distance(reference, target, weights, weights_sum):
            # weighted Manhattan distance
            weighted_diff = torch.abs(reference - target) * weights
            return torch.sum(weighted_diff) / weights_sum

        weights = [[pow(recent_weight, len(x[0]) - i)] for i in range(len(x[0]))]
        weights = torch.Tensor(weights).to(device=self.device)
        weights_sum = torch.sum(weights)

        new_y = torch.zeros(new_x.shape[0], device=self.device)
        for new_x_item_idx_i in tqdm(range(new_x.shape[0]), desc="DAgger search", leave=False):
            best_fit_y = None
            best_fit_diff = -1
            for batch_idx, (x, y) in enumerate(train_dataloader):
                for x_item_idx_j in range(x.shape[0]):
                    cur_diff = input_distance(new_x[new_x_item_idx_i], x[x_item_idx_j], weights, weights_sum)
                    if best_fit_diff == -1 or best_fit_diff > cur_diff:
                        best_fit_diff = cur_diff
                        best_fit_y = y[x_item_idx_j]
            new_y[new_x_item_idx_i] = best_fit_y
        new_y = torch.reshape(new_y, (new_y.shape[0], 1))

        return new_x, new_y
