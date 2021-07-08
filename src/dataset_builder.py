import pandas as pd
import numpy as np
import random
import torch

from torch.utils.data import DataLoader
from src.field_test_dataset import FieldTestDataset
from sklearn.preprocessing import MinMaxScaler


def normalize_dataframes_to_nparrays(raw_dfs: list) -> (list, MinMaxScaler):
    """
    Normalize list of dataframes and return normalzed list of nparrays

    :param raw_dfs: list of pd.DataFrames (list)
    :return list of np.ndarrays (list)
    :return used MinMaxScaler (MinMaxScaler)
    """
    # Merge files
    data_shape_list_for_each_file = [raw_df.shape for raw_df in raw_dfs]
    raw_concat_df = pd.concat(raw_dfs)

    # Normalize data

    mm = MinMaxScaler(feature_range=(-1, 1))
    normalized_concat_nparray = mm.fit_transform(raw_concat_df)

    # Split files
    noramlized_nparrays = []
    prev_index = 0
    for shape in data_shape_list_for_each_file:
        noramlized_nparrays.append(normalized_concat_nparray[prev_index: prev_index + shape[0], :])
        prev_index = prev_index + shape[0]

    return noramlized_nparrays, mm


def build_nparray_dataset_1tick(nparray: np.ndarray, history_length: int) -> (np.ndarray, np.ndarray):
    """
    Build X and Y dataset from given time series in nparray.
    History sequence length in X dataset is history_length.
    Future sequence length in Y dataset is 1.

    :param nparray: time series data (np.ndarray)
    :param history_length: length of a history (int)
    :return X data (np.ndarray)
    :return Y data (np.ndarray)
    """
    X_data = []
    Y_data = []
    for i in range(0, nparray.shape[0] - history_length):
        x = nparray[i:i+history_length, :]
        y = nparray[i+history_length, [0]]
        X_data.append(x)
        Y_data.append(y)
    return np.array(X_data), np.array(Y_data)


def build_nparray_dataset_episode(nparray: np.ndarray, history_length: int, episode_length: int) -> (np.ndarray, np.ndarray):
    """
    Build X and Y dataset from given time series in nparray.
    History sequence length in X dataset is history_length.
    Future sequence length in Y dataset is episode_length.

    :param nparray: time series data (np.ndarray)
    :param history_length: length of a history (int)
    :param episode_length: length of a future (int)
    :return X data (np.ndarray)
    :return Y data (np.ndarray)
    """
    X_data = []
    Y_data = []
    for i in range(0, nparray.shape[0] - history_length - episode_length):
        x = nparray[i:i+history_length, :]
        y = nparray[i+history_length:i+history_length+episode_length, [0]]
        X_data.append(x)
        Y_data.append(y)
    return np.array(X_data), np.array(Y_data)


def build_nparray_dataset_gail(nparray: np.ndarray, history_length: int, episode_length: int) -> (np.ndarray, np.ndarray):
    """
        Build X and Y dataset from given time series in nparray.
        History sequence length in X dataset is history_length.
        Future sequence length in Y dataset is episode_length.

        :param nparray: time series data (np.ndarray)
        :param history_length: length of a history (int)
        :param episode_length: length of a future (int)
        :return X data (np.ndarray)
        :return Y data (np.ndarray)
        """
    X_data = []
    Y_data = []
    for i in range(0, nparray.shape[0] - history_length - episode_length):
        x = nparray[i:i + history_length, :]
        y = nparray[i + history_length:i + history_length + episode_length, :]
        X_data.append(x)
        Y_data.append(y)
    return np.array(X_data), np.array(Y_data)


def build_train_test_validation_dataset(noramlized_nparrays: list, mode: int, history_length: int, episode_length: int, batch_size: int, device: torch.device) -> (list, list, list):
    """
    Build dataloaders for training, testing, validation from given time series datasets.

    :param noramlized_nparrays: time series data (list of np.ndarray)
    :param mode: model generation algorithm mode (int)
    :param history_length: length of a history (int)
    :param episode_length: length of a future (int)
    :param batch_size: batch size (int)
    :param device: available device (cpu or cuda) (torch.device)
    :return list of training dataloaders (list)
    :return list of testing dataloaders (list)
    :return list of validation dataloaders (list)
    """
    train_dataloaders = []
    val_dataloaders = []
    test_dataloaders = []
    for normalized_data in noramlized_nparrays:
        if mode == 0 or mode == 1:
            x_data, y_data = build_nparray_dataset_1tick(normalized_data, history_length)
        elif mode == 2:
            x_data, y_data = build_nparray_dataset_episode(normalized_data, history_length, episode_length)
        elif mode == 3:
            x_data, y_data = build_nparray_dataset_gail(normalized_data, history_length, episode_length)

        len = x_data.shape[0]
        split_loc = [7, 2, 1]
        random.shuffle(split_loc)
        split_idx = [0, int(split_loc[0] * len / 10), int((split_loc[0] + split_loc[1]) * len / 10), len]

        train_loc = split_loc.index(7)
        train_dataloaders.append(DataLoader(dataset=FieldTestDataset(
            torch.from_numpy(x_data[split_idx[train_loc]:split_idx[train_loc + 1]]).type(torch.Tensor).to(
                device=device),
            torch.from_numpy(y_data[split_idx[train_loc]:split_idx[train_loc + 1]]).type(torch.Tensor).to(
                device=device)),
            batch_size=batch_size, shuffle=False))

        val_loc = split_loc.index(2)
        val_dataloaders.append(DataLoader(dataset=FieldTestDataset(
            torch.from_numpy(x_data[split_idx[val_loc]:split_idx[val_loc + 1]]).type(torch.Tensor).to(device=device),
            torch.from_numpy(y_data[split_idx[val_loc]:split_idx[val_loc + 1]]).type(torch.Tensor).to(device=device)),
            batch_size=batch_size, shuffle=False))

        test_loc = split_loc.index(1)
        test_dataloaders.append(DataLoader(dataset=FieldTestDataset(
            torch.from_numpy(x_data[split_idx[test_loc]:split_idx[test_loc + 1]]).type(torch.Tensor).to(device=device),
            torch.from_numpy(y_data[split_idx[test_loc]:split_idx[test_loc + 1]]).type(torch.Tensor).to(device=device)),
            batch_size=batch_size, shuffle=False))

    return train_dataloaders, val_dataloaders, test_dataloaders


