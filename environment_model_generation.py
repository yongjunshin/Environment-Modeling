import numpy as np
import pandas as pd
import torch  # pytorch

from torch.utils.data import DataLoader
import argparse

import glob
import os
import datetime
import random

from behaviorCloningTrainer import BehaviorCloningTrainer
from LineTracerEnvironmentModelGRU import LineTracerEnvironmentModelGRU
from LineTracerVer1 import LineTracerVer1


# Argument parsing
from fieldTestDataset import FieldTestDataset

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--source_file", type=str, nargs='+',
                    help="<Mandatory (or -d)> field test data source files (list seperated by spaces)", default=["data/ver1_fixed_interval/ver1_ft_60_30.csv", "data/ver1_fixed_interval/ver1_ft_60_20.csv"])
parser.add_argument("-d", "--source_directory", type=str,
                    help="<Mandatory (or -f)> filed test data directory (It will read only *.csv files.)", default=None)
possible_mode = ["bc_1tick_noDagger", "bc_1tick_Dagger"]
parser.add_argument("-m", "--mode", type=str,
                    help="model generation algorithm among "+str(possible_mode)+" (default: bc_1tick_noDagger)", default='bc_1tick_Dagger')
parser.add_argument("-l", "--history_length", type=int,
                    help="history length (default: 100)", default=100)
parser.add_argument("-b", "--batch_size", type=int,
                    help="mini batch size (default: 128)", default=100)
parser.add_argument("-md", "--max_dagger", type=int,
                    help="maximum number of dagger", default=10)
parser.add_argument("-dt", "--dagger_threshold", type=float,
                    help="dagger operation flag threshold", default=0.02)
parser.add_argument("-dm", "--distance_metric", type=str,
                    help="history distance metric ['ed', 'wed', 'md', 'wmd', 'dtw'] (default: wmd)", default='wmd')

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
    elif args.mode == "bc_1tick_Dagger":
        mode = 1

    return mode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mode = mode_selection(args)
source_files = args.source_file
source_directory = args.source_directory
history_length = args.history_length
batch_size = args.batch_size
if mode == 1:
    dagger_on = True
else:
    dagger_on = False
max_dagger = args.max_dagger
dagger_threshold = args.dagger_threshold
distance_metric = args.distance_metric


# Input specification
print("Environment model generation (start at", datetime.datetime.now(), ")")
print("=====Input specification=====")
print("Available device:", device)
print("Source data:", source_files)
print("History length:", history_length)
if mode == 0:
    print("Environment model generation algorithm: 1-tick Behavior Cloning without DAgger")
elif mode == 1:
    print("Environment model generation algorithm: 1-tick Behavior Cloning with DAgger")
    print("Maximum number of DAgger execution:", max_dagger)
    print("DAgger execution thresthold (prediction threshold):", dagger_threshold)
    print("History distance metric:", distance_metric)
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
        batch_size=batch_size, shuffle=False))

    test_loc = split_loc.index(2)
    test_dataloaders.append(DataLoader(dataset=FieldTestDataset(
        torch.from_numpy(x_data[split_idx[test_loc]:split_idx[test_loc + 1]]).type(torch.Tensor).to(device=device),
        torch.from_numpy(y_data[split_idx[test_loc]:split_idx[test_loc + 1]]).type(torch.Tensor).to(device=device)),
        batch_size=batch_size, shuffle=False))

    val_loc = split_loc.index(1)
    validation_dataloaders.append(DataLoader(dataset=FieldTestDataset(
        torch.from_numpy(x_data[split_idx[val_loc]:split_idx[val_loc + 1]]).type(torch.Tensor).to(device=device),
        torch.from_numpy(y_data[split_idx[val_loc]:split_idx[val_loc + 1]]).type(torch.Tensor).to(device=device)),
        batch_size=batch_size, shuffle=False))

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

print("Step 5: Build system under test")
# instantiate a moddel
line_tracer = LineTracerVer1(mm)

print("Step 5: Train environment model")
# train the environment model
if mode == 0 or mode == 1:
    trainer = BehaviorCloningTrainer(device=device, sut=line_tracer)
    training_loss, dagger_count = trainer.train(model=model, epochs=30, train_dataloaders=train_dataloaders,
                                                test_dataloaders=test_dataloaders, dagger=dagger_on,
                                                max_dagger=max_dagger, dagger_threshold=dagger_threshold,
                                                dagger_batch_size=batch_size, distance_metric=distance_metric)
print("--training loss:", training_loss)
print("--dagger count:", dagger_count)



print("=====(end)=====")