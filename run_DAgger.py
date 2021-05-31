import numpy as np
import pandas as pd
import torch  # pytorch
import torch.nn as nn
import argparse
from tqdm import tqdm

# Checking available device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# initialize argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--num_lines", type=int, help="Number of lines to read from the data file", default=None)
args = parser.parse_args()

# Read files
raw_df = pd.read_csv('data/ver1_fixed_interval/ver1_ft_60_10.csv', index_col='time', nrows=args.num_lines)
print("raw_df.shape:", raw_df.shape)

# Normalize data
from sklearn.preprocessing import StandardScaler, MinMaxScaler

mm = MinMaxScaler(feature_range=(-1, 1))
normalized_nparray = mm.fit_transform(raw_df)
print("normalized_nparray.shape:", normalized_nparray.shape)


# Build dataset
def Build_dataset(nparray, seq_length):
    X_data = []
    Y_data = []
    for i in range(0, len(nparray) - seq_length):
        X_data.append(nparray[i: i + seq_length, :])
        Y_data.append(nparray[i + seq_length, [0]])  # column 0 is 'color' column
    return np.array(X_data), np.array(Y_data)


sequence_length = 100
X_train, Y_train = Build_dataset(normalized_nparray, sequence_length)
X_train = torch.from_numpy(X_train).type(torch.Tensor).to(device=device)
Y_train = torch.from_numpy(Y_train).type(torch.Tensor).to(device=device)
print("X_train.shape:", X_train.shape)
print("Y_train.shape:", Y_train.shape)


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device=device)
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :])
        return out


# Instantiate a model
input_dim = 2
hidden_dim = 32
num_layers = 2
output_dim = 1

model = GRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
model.to(device=device)
print(model)


# Virtual Line Tracer version 1
def Line_tracer_decision_making(color):
    black = 5
    white = 75
    threshold = (black + white) / 2
    turning_ratio = 10

    denormalized_color = mm.inverse_transform([[color, 0]])[0][0]  # De-normalize input data

    # System decision-making logic
    if denormalized_color > threshold:
        None
    elif denormalized_color < threshold:
        turning_ratio = -turning_ratio
    else:
        turning_ratio = 0

    normalized_turning_ratio = mm.fit_transform([[0, turning_ratio]])[0][1]  # Normalize output data
    return normalized_turning_ratio


def history_difference(reference, target, weights, weights_sum):
    # weighted Manhattan distance
    weighted_diff = torch.abs(reference - target) * weights
    return torch.sum(weighted_diff) / weights_sum


def DAgger(training_X, pred_Y, original_X, original_Y):
    recent_weight = 0.9  # weight for recent element in history comparison
    weights = [[pow(recent_weight, sequence_length - i)] for i in range(sequence_length)]  # exponential weights
    # weights = [[weight * (i/sequence_length)] for i in range(sequence_length)]  # linear weights
    # weights = [[weight] for i in range(sequence_length)]  # uniform weights
    weights = torch.Tensor(weights).to(device=device)
    weights_sum = torch.sum(weights)

    # Inject faulty prediction data to the given training data to get new X dataset
    new_X = training_X[:-1, 1:, :]
    new_X = torch.cat([new_X, pred_Y[:-1]], dim=1)

    # Search the most similar history and get new Y
    new_Y = torch.zeros(new_X.shape[0], device=device)
    for new_X_item_idx in tqdm(range(new_X.shape[0])):  # for each new X item
        best_fit_idx = None
        best_fit_diff = 2  # the theoretically possible maximum diff
        for original_X_item_idx in range(original_X.shape[0]):  # from all original X dataset
            cur_diff = history_difference(new_X[new_X_item_idx], original_X[original_X_item_idx], weights, weights_sum)
            if best_fit_diff > cur_diff:
                best_fit_diff = cur_diff
                best_fit_idx = original_X_item_idx
        new_Y[new_X_item_idx] = original_Y[best_fit_idx]
    new_Y = torch.reshape(new_Y, (new_Y.shape[0], 1))

    return new_X, new_Y


def heuristic_DAgger(training_X, training_Y, pred_Y):
    # Inject faulty prediction data to the given training data to get new X dataset
    new_X = training_X[:-1, 1:, :]
    new_X = torch.cat([new_X, pred_Y[:-1]], dim=1)

    # Search the most similar history and get new Y
    new_Y = training_Y[1:, :]
    return new_X, new_Y


# Forward propagation to get future environmental state data
Y_train_pred = model(X_train)
print("Y_train_pred.shape:", Y_train_pred.shape)

# System simulation to get future system operation data
Y_train_pred_data = Y_train_pred.detach()
sys_op_pred_data = np.zeros(Y_train_pred_data.shape[0])
for t in range(len(sys_op_pred_data)):
    sys_op_pred_data[t] = Line_tracer_decision_making(Y_train_pred_data[t])
sys_op_pred_data = torch.Tensor([sys_op_pred_data]).to(device=device)
sys_op_pred_data = torch.reshape(sys_op_pred_data, (sys_op_pred_data.shape[1], 1))

# Merege future data: (future environmental state data + future system operation data)
complete_future_data = torch.cat((Y_train_pred_data, sys_op_pred_data), 1)
complete_future_data = torch.reshape(complete_future_data,
                                     (complete_future_data.shape[0], 1, complete_future_data.shape[1]))
print("complete_future_data.shape:", complete_future_data.shape)
print()

# DAgger to generate new X and Y dataset
import time

# DAgger standard (search the most similar history to get new Y)
prev_time = time.time()
print("===DAgger standard===")
new_X_training_standard, new_Y_training_standard = DAgger(X_train, complete_future_data, X_train, Y_train)
print("new_X_training_standard.shape:", new_X_training_standard.shape, "new_Y_training_standard.shape:",
      new_Y_training_standard.shape)
cur_time = time.time()
print("consumed time:", cur_time - prev_time, 's')

# DAgger heuristic (reuse the given Y without searching the most similar history, based on a heuristic assumption)
prev_time = time.time()
print("===DAgger heuristic===")
new_X_training_heuristic, new_Y_training_heuristic = heuristic_DAgger(X_train, Y_train, complete_future_data)
print("new_X_training_heuristic.shape:", new_X_training_heuristic.shape, "new_Y_training_heuristic.shape:",
      new_Y_training_heuristic.shape)
cur_time = time.time()
print("consumed time:", cur_time - prev_time, 's')

# DAgger result comparison
print()
print("element-wise average of differences between new Ys made by standard and heuristic DAgger:",
      torch.sum(torch.abs(new_Y_training_standard - new_Y_training_heuristic))/new_Y_training_standard.shape[0],
      "over", 2)


