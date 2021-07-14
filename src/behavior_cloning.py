from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
from src.field_test_dataset import FieldTestDataset
from src.soft_dtw_cuda import SoftDTW
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

class BehaviorCloning1TickTrainer:
    def __init__(self, device, sut):
        self.device = device
        self.sut = sut
        if self.device == 'cuda':
            self.sdtw = SoftDTW(use_cuda=True, gamma=0.1)
        else:
            self.sdtw = SoftDTW(use_cuda=False, gamma=0.1)

    def train(self, model: torch.nn.Module, epochs: int, train_dataloaders: list, validation_dataloaders: list,
              episode_length: int, dagger: bool = False, max_dagger: int = 10, dagger_threshold: float = 0.01,
              dagger_batch_size: int = 100, distance_metric: str = "dtw") -> (list, int):
        """
        Behavior cloning 1-tick NoDAgger/DAgger

        :param model: environment model (torch.nn.Module)
        :param epochs: training epochs (int)
        :param train_dataloaders: list of training dataloaders (list of dataloaders)
        :param test_dataloaders: list of testing dataloaders (list of dataloaders)
        :param dagger: dagger on/off (bool)
        :param max_dagger: maximum number of dagger (int)
        :param dagger_threshold: dagger operation threshold accuracy (float)
        :param dagger_batch_size: new dagger dataset batch size (int)
        :param distance_metric: ['ed', 'wed', 'md', 'wmd', 'dtw'] (default: dtw) (str)
        """
        loss_fn = torch.nn.MSELoss()
        optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
        training_loss = np.zeros(epochs)

        dagger_dataloaders = []
        dagger_count = 0
        for i in tqdm(range(epochs), desc="Epochs:"):
            num_batch = sum([len(dl) for dl in train_dataloaders]) + sum([len(dl) for dl in dagger_dataloaders])
            dagger_flag = False

            for dataloader in train_dataloaders + dagger_dataloaders:
                new_xs = []
                new_ys = []

                for batch_idx, (x, y) in enumerate(dataloader):
                    model.train()
                    y_pred = model(x)
                    loss = loss_fn(y, y_pred)
                    training_loss[i] = training_loss[i] + loss.item()

                    bc_dist = model.get_distribution(x)
                    target_y = y
                    bc_loss = -bc_dist.log_prob(target_y)

                    optimiser.zero_grad()
                    #loss.backward()
                    bc_loss.mean().backward()
                    optimiser.step()

                    if dagger and dagger_count < max_dagger and loss < dagger_threshold:
                        dagger_flag = True
                        new_x, new_y = self.dagger(train_dataloaders=train_dataloaders, x=x, distance_metric=distance_metric, y_pred=y_pred)
                        new_xs.append(new_x)
                        new_ys.append(new_y)

                        # plt.figure(figsize=(10, 5))
                        # plt.plot(x[1, :, [0]].cpu(), label="Original X")
                        # plt.plot(new_x[0, :, [0]].cpu(), label="DAgger new X")
                        # plt.legend()
                        # plt.show()

                if len(new_xs) > 0:
                    new_x_tensor = torch.cat(new_xs, dim=0)
                    new_y_tensor = torch.cat(new_ys, dim=0)
                    dagger_dataloaders.append(DataLoader(dataset=FieldTestDataset(
                        new_x_tensor, new_y_tensor), batch_size=dagger_batch_size, shuffle=False))


            training_loss[i] = training_loss[i]/num_batch
            if dagger_flag:
                dagger_count = dagger_count + 1

            # Simulation visualization start
            model.eval()
            for batch_idx, (x, y) in enumerate(train_dataloaders[0]):
                if batch_idx == 0:
                    y_pred = torch.zeros((x.shape[0], episode_length, x.shape[1]), device=self.device)
                    sim_x = x
                    for sim_idx in range(episode_length):
                        #y_pred_one_step = model(sim_x)
                        y_pred_one_step = model.act(sim_x)
                        y_pred[:, sim_idx, 0] = y_pred_one_step[:, 0]

                        env_prediction = y_pred_one_step.cpu().detach().numpy()
                        sys_operations = self.sut.act_sequential(env_prediction)
                        next_x = np.concatenate((env_prediction, sys_operations), axis=1)
                        next_x = torch.tensor(next_x).to(device=self.device).type(torch.float32)
                        next_x = torch.reshape(next_x, (next_x.shape[0], 1, next_x.shape[1]))
                        sim_x = sim_x[:, 1:]
                        sim_x = torch.cat((sim_x, next_x), 1)

                    plt.figure(figsize=(10, 5))
                    plt.plot(y_pred[0, :, [0]].cpu().detach().numpy(), label="y_pred")
                    plt.plot(y[:100, [0]].cpu().detach().numpy(), label="y")
                    plt.legend()
                    plt.show()
                    #plt.savefig('output/imgs/Dagger/fig'+str(i)+'.png', dpi=300)
            # Simulation visualization end

        return training_loss, dagger_count

    def dagger(self, train_dataloaders, x, y_pred, distance_metric="wmd"):
        env_prediction = y_pred.cpu().detach().numpy()

        # sys_operations = np.zeros(len(env_prediction))
        # for i in range(len(sys_operations)):
        #     sys_operations[i] = self.sut.act(env_prediction[i])
        sys_operations = self.sut.act_sequential(env_prediction)
        sys_operations = torch.tensor([sys_operations]).to(device=self.device).type(torch.float32)
        sys_operations = torch.reshape(sys_operations, (sys_operations.shape[1], 1))
        env_prediction = torch.tensor(env_prediction).to(device=self.device)
        error_data = torch.cat((env_prediction[:], sys_operations[:]), 1)
        error_data = torch.reshape(error_data, (error_data.shape[0], 1, error_data.shape[1]))

        new_x = x[:-1, 1:, :]
        new_x = torch.cat([new_x, error_data[:-1]], dim=1)

        recent_weight = 0.9
        weights = [[pow(recent_weight, len(x[0]) - i)] for i in range(len(x[0]))]
        weights = torch.tensor(weights).to(device=self.device)
        weights_sum = torch.sum(weights)

        new_y = torch.zeros(new_x.shape[0], device=self.device)
        for new_x_item_idx_i in tqdm(range(new_x.shape[0]), desc="DAgger search", leave=False):
            best_fit_y = None
            best_fit_diff = -1
            for dataloader in train_dataloaders:
                for batch_idx, (x, y) in enumerate(dataloader):
                    cur_diffs = self.input_distances(new_x[new_x_item_idx_i], x, distance_metric, weights, weights_sum)
                    np_cur_diffs = cur_diffs.cpu().numpy()
                    min_idx = np.argmin(np_cur_diffs)
                    cur_best_fit_diff = cur_diffs[min_idx]
                    if best_fit_diff == -1 or best_fit_diff > cur_best_fit_diff:
                        best_fit_diff = cur_best_fit_diff
                        best_fit_y = y[min_idx]


            new_y[new_x_item_idx_i] = best_fit_y
        new_y = torch.reshape(new_y, (new_y.shape[0], 1))

        return new_x, new_y

    def input_distances(self, new_x_data: torch.Tensor, reference_dataset: torch.Tensor, method: str, weights: torch.Tensor, weights_sum: torch.Tensor):
        """
        Calculate distances from new_x_data to elements of reference_dataset

        :param new_x_data: target time series (torch.Tensor: shape(#sequence, #factor))
        :param reference_dataset: set of reference time series (torch.Tensor: shape(#batch, #sequence, #factor))
        :param method: ['ed', 'wed', 'md', 'wmd', 'dtw'] (str)
        :param weights: tensor of weigths (torch.Tensor: shape(#sequence))
        :param weights_sum: sum of weights (torch.Tensor: shape(1))
        :return: tensor of distances (torch.Tensor: shape(#batch))
        """
        if method == "ed":
            # euclidean distance
            reshaped_new_dataset = new_x_data.repeat((len(reference_dataset), 1, 1))
            diffs = torch.pow(reference_dataset - reshaped_new_dataset, 2)
            diffs = torch.sum(diffs, dim=(1, 2))
            diffs = torch.sqrt(diffs)
            return diffs
        elif method == "wed" and weights is not None and weights_sum is not None:
            # weighted euclidean distance
            reshaped_new_dataset = new_x_data.repeat((len(reference_dataset), 1, 1))
            diffs = torch.pow(reference_dataset - reshaped_new_dataset, 2) * weights
            diffs = torch.sum(diffs, dim=(1, 2)) / weights_sum
            diffs = torch.sqrt(diffs)
            return diffs
        elif method == "md":
            # manhattan distance
            reshaped_new_dataset = new_x_data.repeat((len(reference_dataset), 1, 1))
            diffs = torch.abs(reference_dataset - reshaped_new_dataset)
            diffs = torch.sum(diffs, dim=(1, 2))
            return diffs
        elif method == "wmd" and weights is not None and weights_sum is not None:
            # weighted manhattan distance
            reshaped_new_dataset = new_x_data.repeat((len(reference_dataset), 1, 1))
            diffs = torch.abs(reference_dataset - reshaped_new_dataset) * weights
            diffs = torch.sum(diffs, dim=(1, 2)) / weights_sum
            return diffs
        elif method == "dtw":
            # (soft) dynamic time warping
            reshaped_new_dataset = new_x_data.repeat((len(reference_dataset), 1, 1))
            diffs = self.sdtw(reshaped_new_dataset, reference_dataset)
            return diffs
        else:
            print("[Error] Wrong history distance metric and parameters.")
            quit()


class BehaviorCloningEpisodeTrainer:
    def __init__(self, device, sut):
        self.device = device
        self.sut = sut

    def train(self, model: torch.nn.Module, epochs: int, train_dataloaders: list, validation_dataloaders: list, loss_metric: str = "mdtw"):
        """
        Behavior cloning episode

        :param model: environment model (torch.nn.Module)
        :param epochs: training epochs (int)
        :param train_dataloaders: list of training dataloaders (list of dataloaders)
        :param test_dataloaders: list of testing dataloaders (list of dataloaders)
        :param loss_metric: ['mse', 'mdtw'] (default: mdtw) (str)
        """
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
        elif loss_metric == "pcc":
            class PCCLoss(torch.nn.Module):
                def __init__(self):
                    super().__init__()

                def forward(self, y_pred, y):
                    vy_pred = y_pred - torch.mean(y_pred)
                    vy = y - torch.mean(y)

                    cor = torch.sum(vy_pred * vy) / (torch.sqrt(torch.sum(vy_pred ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
                    return -torch.abs(cor)
            loss_fn = PCCLoss()
        else:
            print("[Error] Wrong history distance metric and parameters.")
            quit()
        optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
        training_loss = np.zeros(epochs)


        num_batch = sum([len(dl) for dl in train_dataloaders])
        for i in tqdm(range(epochs), desc="Epochs:"):
            loader_idx = 0
            for dataloader in train_dataloaders:
                for batch_idx, (x, y) in enumerate(dataloader):
                    model.train()
                    y_pred = torch.zeros(y.shape, device=self.device)
                    sim_x = x
                    for sim_idx in range(y.shape[1]):
                        y_pred_one_step = model(sim_x)
                        y_pred_one_step = y_pred_one_step + torch.normal(mean=torch.zeros(y_pred_one_step.shape), std=torch.ones(y_pred_one_step.shape)*0.01).to(device=self.device)
                        y_pred[:, sim_idx, 0] = y_pred_one_step[:, 0]

                        env_prediction = y_pred_one_step.cpu().detach().numpy()
                        sys_operations = self.sut.act_sequential(env_prediction)
                        next_x = np.concatenate((env_prediction, sys_operations), axis=1)
                        next_x = torch.tensor(next_x).to(device=self.device).type(torch.float32)
                        next_x = torch.reshape(next_x, (next_x.shape[0], 1, next_x.shape[1]))
                        sim_x = sim_x[:, 1:]
                        sim_x = torch.cat((sim_x, next_x), 1)

                    # Simulation visualization start
                    if batch_idx == 0 and loader_idx == 0:
                        plt.figure(figsize=(10, 5))
                        plt.plot(y_pred[0, :, [0]].cpu().detach().numpy(), label="y_pred")
                        plt.plot(y[0, :, [0]].cpu().detach().numpy(), label="y")
                        plt.legend()
                        plt.show()
                        #plt.savefig('output/imgs/episode_pcc/fig' + str(i) + '.png', dpi=300)
                    # Simulation visualization end

                    loss = loss_fn(y_pred, y)
                    training_loss[i] = training_loss[i] + loss.item()

                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()
                loader_idx = loader_idx + 1

            training_loss[i] = training_loss[i] / num_batch

            #print(training_loss[i])

        return training_loss
