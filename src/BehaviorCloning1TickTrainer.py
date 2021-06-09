from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np

from src.fieldTestDataset import FieldTestDataset
from src.soft_dtw_cuda import SoftDTW


class BehaviorCloning1TickTrainer:
    def __init__(self, device, sut):
        self.device = device
        self.sut = sut
        self.sdtw = SoftDTW(use_cuda=True, gamma=0.1)

    def train(self, model, epochs, train_dataloaders, test_dataloaders, dagger=False, max_dagger=10, dagger_threshold=0.01, dagger_batch_size=100, distance_metric="wmd"):
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

                    optimiser.zero_grad()
                    loss.backward()
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

    def input_distances(self, new_x_data, reference_dataset, method, weights, weights_sum):
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
            #diff, path = fastdtw(reference.cpu().detach().numpy(), target.cpu().detach().numpy(), dist=euclidean)
            reshaped_new_dataset = new_x_data.repeat((len(reference_dataset), 1, 1))
            diffs = self.sdtw(reshaped_new_dataset, reference_dataset)
            return diffs
        else:
            print("[Error] Wrong history distance metric and parameters.")
            quit()


