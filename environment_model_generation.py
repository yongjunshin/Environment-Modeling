import numpy as np
import pandas as pd
import torch  # pytorch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import argparse

import glob
import os
import datetime
import random
from LineTracerEnvironmentModelGRU import LineTracerEnvironmentModelGRU
from BehaviorCloningTrainer import BehaviorCloningTrainer


# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--source_file", type=str, nargs='+',
                    help="<Mandatory (or -d)> field test data source files (list seperated by spaces)", default=["data/ver1_fixed_interval/ver1_ft_60_30.csv", "data/ver1_fixed_interval/ver1_ft_60_20.csv"])
parser.add_argument("-d", "--source_directory", type=str,
                    help="<Mandatory (or -f)> filed test data directory (It will read only *.csv files.)", default=None)
possible_mode = ["bc_1tick_noDagger"]
parser.add_argument("-m", "--mode", type=str,
                    help="model generation algorithm among "+str(possible_mode)+" (default: bc_1tick_noDagger)", default='bc_1tick_noDagger')
parser.add_argument("-l", "--history_length", type=int,
                    help="history length (default: 100)", default=100)

args = parser.parse_args()


# Running mode selection and wrong input handling
def mode_selection(args):
    if args.source_file is None and args.source_directory is None:
        print("[Error] Source files (-f) or a source directory (-d) must be given. Refer help (-h).")
        quit()
    if args.source_file is not None and args.source_directory is not None:
        print("[Error] Only source files (-f) or a source directory (-d) must be given at once. Refer help (-h).")
        quit()
    if args.mode not in possible_mode:
        print("[Error] Wrong mode was given. Select among " + str(possible_mode) + ". Refer help (-h).")
        quit()

    if args.mode == "bc_1tick_noDagger":
        mode = 0

    return mode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mode = mode_selection(args)
source_files = args.source_file
source_directory = args.source_directory
history_length = args.history_length



# Input specification
print("Environment model generation (start at", datetime.datetime.now(), ")")
print("=====Input specification=====")
print("Available device:", device)
print("Source data:", source_files)
if mode == 0:
    print("Environment model generation algorithm: 1-tick Behavior Cloning without DAgger")
    print("History length:", history_length)
print("=====(end)=====")
print()

print("=====Environment model generation process=====")
print("Step 1: Source data reading")
# Source data reading
if source_directory is not None:
    source_files = glob.glob(os.path.join(source_directory, "*.csv"))

raw_dfs = []
for file in source_files:
    raw_dfs.append(pd.read_csv(file, index_col='time'))
print("--data size:", [raw_df.shape for raw_df in raw_dfs])

print("Step 2: Data normalization")
# Merge files
data_shape_list_for_each_file = [raw_df.shape for raw_df in raw_dfs]
raw_concat_df = pd.concat(raw_dfs)

# Normalize data
from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler(feature_range=(-1,1))
normalized_concat_nparray = mm.fit_transform(raw_concat_df)

# Split files
noramlized_nparrays = []
prev_index = 0
for shape in data_shape_list_for_each_file:
    noramlized_nparrays.append(normalized_concat_nparray[prev_index : prev_index + shape[0], :])
    prev_index = prev_index + shape[0]

print("Step 3: Build train/test/validation dataset")
# Build train/test/validation dataset
def build_nparray_dataset(nparray, seq_length):
    X_data = []
    Y_data = []

    for i in range(0, nparray.shape[0]-seq_length):
        x = nparray[i: i+seq_length, :]
        y = nparray[i + seq_length, [0]]

        X_data.append(x)
        Y_data.append(y)
    return np.array(X_data), np.array(Y_data)

class FieldTestDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        super(FieldTestDataset, self).__init__()
        self.x = x_tensor
        self.y = y_tensor

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]


train_dataloaders = []
test_dataloaders = []
validation_dataloaders = []
for normalized_data in noramlized_nparrays:
    x_data, y_data = build_nparray_dataset(normalized_data, history_length)
    len = x_data.shape[0]
    split_loc = [7, 2, 1]
    random.shuffle(split_loc)
    split_idx = [0, int(split_loc[0] * len/10), int((split_loc[0] + split_loc[1]) * len/10), len]

    train_loc = split_loc.index(7)
    train_dataloaders.append(DataLoader(dataset=FieldTestDataset(
        torch.from_numpy(x_data[split_idx[train_loc]:split_idx[train_loc+1]]).type(torch.Tensor).to(device=device),
        torch.from_numpy(y_data[split_idx[train_loc]:split_idx[train_loc+1]]).type(torch.Tensor).to(device=device)),
        batch_size=128, shuffle=False))

    test_loc = split_loc.index(2)
    test_dataloaders.append(DataLoader(dataset=FieldTestDataset(
        torch.from_numpy(x_data[split_idx[test_loc]:split_idx[test_loc + 1]]).type(torch.Tensor).to(device=device),
        torch.from_numpy(y_data[split_idx[test_loc]:split_idx[test_loc + 1]]).type(torch.Tensor).to(device=device)),
        batch_size=128, shuffle=False))

    val_loc = split_loc.index(1)
    validation_dataloaders.append(DataLoader(dataset=FieldTestDataset(
        torch.from_numpy(x_data[split_idx[val_loc]:split_idx[val_loc + 1]]).type(torch.Tensor).to(device=device),
        torch.from_numpy(y_data[split_idx[val_loc]:split_idx[val_loc + 1]]).type(torch.Tensor).to(device=device)),
        batch_size=128, shuffle=False))

print("--train dataset shape:", [str(loader.dataset.x.shape) +'->' + str(loader.dataset.y.shape) for loader in train_dataloaders])
print("--test dataset shape:", [str(loader.dataset.x.shape) +'->' + str(loader.dataset.y.shape) for loader in test_dataloaders])
print("--validation dataset shape:", [str(loader.dataset.x.shape) +'->' + str(loader.dataset.y.shape) for loader in validation_dataloaders])

print("Step 4: Build environment model")
# instantiate a moddel
input_dim = 2
hidden_dim = 16
num_layers = 2
output_dim = 1
model = LineTracerEnvironmentModelGRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers, device=device)
model.to(device=device)
print("--environment model summary:", model)

print("Step 5: Train environment model")
# train the environment model
trainer = BehaviorCloningTrainer()
training_loss = trainer.train(model=model, epochs=30, train_dataloaders=train_dataloaders, test_dataloaders=test_dataloaders)
print("--training loss:", training_loss)


print("=====(end)=====")