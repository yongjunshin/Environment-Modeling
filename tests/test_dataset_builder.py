import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from sklearn.preprocessing import MinMaxScaler
import numpy as np
from src import dataset_builder

def test_normalize_dataframes_to_nparrays_input_output_shape_check():
    raw_dfs = []
    for i in range(3):
        raw_dfs.append(pd.DataFrame([[-1, -1], [1, 1]]))
    normalized_nparrays, scaler = dataset_builder.normalize_dataframes_to_nparrays(raw_dfs)

    assert len(raw_dfs) == len(normalized_nparrays)
    assert raw_dfs[0].shape == normalized_nparrays[0].shape
    assert raw_dfs[1].shape == normalized_nparrays[1].shape
    assert raw_dfs[2].shape == normalized_nparrays[2].shape

def test_build_nparray_dataset_1tick_input_output_shape_check():
    nparray = np.zeros((1000, 2))
    history_length = 10
    x, y = dataset_builder.build_nparray_dataset_1tick(nparray, history_length)

    assert nparray.shape[0] - history_length == x.shape[0]
    assert nparray.shape[0] - history_length == y.shape[0]
    assert x.shape[1] == history_length
    assert x.shape[2] == nparray.shape[1]
    assert y.shape[1] == 1

def test_build_nparray_dataset_episode_input_output_shape_check():
    nparray = np.zeros((1000, 2))
    history_length = 10
    episode_length = 15
    x, y = dataset_builder.build_nparray_dataset_episode(nparray, history_length, episode_length)

    assert nparray.shape[0] - history_length - episode_length == x.shape[0]
    assert nparray.shape[0] - history_length - episode_length == y.shape[0]
    assert x.shape[1] == history_length
    assert y.shape[1] == episode_length
    assert x.shape[2] == nparray.shape[1]
    assert y.shape[2] == 1

def test_build_nparray_dataset_gail_input_output_shape_check():
    nparray = np.zeros((1000, 2))
    history_length = 10
    episode_length = 15
    x, y = dataset_builder.build_nparray_dataset_gail(nparray, history_length, episode_length)

    assert nparray.shape[0] - history_length - episode_length == x.shape[0]
    assert nparray.shape[0] - history_length - episode_length == y.shape[0]
    assert x.shape[1] == history_length
    assert y.shape[1] == episode_length
    assert x.shape[2] == nparray.shape[1]
    assert y.shape[2] == nparray.shape[1]

def test_build_train_test_validation_dataset_input_output_shape_check():
    normalized_nparrays = []
    for i in range(3):
        normalized_nparrays.append(np.zeros((1000, 2)))
    history_length = 100
    batch_size = 300
    device = 'cpu'

    train_dls, test_dls, val_dls = dataset_builder.build_train_test_validation_dataset(normalized_nparrays, 0, history_length, None, batch_size, device)

    assert len(normalized_nparrays) == len(train_dls)
    assert len(normalized_nparrays) == len(test_dls)
    assert len(normalized_nparrays) == len(val_dls)
    assert len(train_dls[0]) >= len(test_dls[0]) and len(test_dls[0]) >= len(val_dls[0])

    train_dls, test_dls, val_dls = dataset_builder.build_train_test_validation_dataset(normalized_nparrays, 1,history_length, None, batch_size, device)

    assert len(normalized_nparrays) == len(train_dls)
    assert len(normalized_nparrays) == len(test_dls)
    assert len(normalized_nparrays) == len(val_dls)
    assert len(train_dls[0]) >= len(test_dls[0]) and len(test_dls[0]) >= len(val_dls[0])

    episode_length = 150
    train_dls, test_dls, val_dls = dataset_builder.build_train_test_validation_dataset(normalized_nparrays, 2, history_length, episode_length, batch_size, device)

    assert len(normalized_nparrays) == len(train_dls)
    assert len(normalized_nparrays) == len(test_dls)
    assert len(normalized_nparrays) == len(val_dls)
    assert len(train_dls[0]) >= len(test_dls[0]) and len(test_dls[0]) >= len(val_dls[0])
