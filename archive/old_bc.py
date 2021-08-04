import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch #pytorch
import torch.nn as nn
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device available now:', device)

# Read files
raw_dfs = []
raw_dfs.append(pd.read_csv('data/ver1_fixed_interval/ver1_ft_60_30.csv', index_col = 'time')[:6000])
raw_dfs.append(pd.read_csv('data/ver1_fixed_interval/ver1_ft_60_30.csv', index_col = 'time')[6000:])

for df in raw_dfs:
  df[' color'] = df[' color'].rolling(window=5, center=True, min_periods=1).mean()


print("# of files: ", len(raw_dfs))
print("Head of first file")
raw_dfs[0].head()
for raw_df in raw_dfs:
  print(raw_df.shape)

plt.style.use('ggplot')
raw_dfs[0][' color'].plot(label='Color', title='Road observation', figsize =(30,5))

# Merge files
data_shape_list_for_each_file = [raw_df.shape for raw_df in raw_dfs]
#print(data_shape_list_for_each_file)
for data_shape in data_shape_list_for_each_file:
  print(data_shape)
raw_concat_df = pd.concat(raw_dfs)
#print(concat_df.shape)

# Normalize data
from sklearn.preprocessing import StandardScaler, MinMaxScaler
mm = MinMaxScaler(feature_range=(-1,1))
normalized_concat_nparray = mm.fit_transform(raw_concat_df)

# Split files
noramlized_nparrays = []
prev_index = 0
for shape in data_shape_list_for_each_file:
  noramlized_nparrays.append(normalized_concat_nparray[prev_index : prev_index + shape[0], :])
  prev_index = prev_index + shape[0]
#noramzliaed_data_shape_list_for_each_file = [normalized_nparray.shape for normalized_nparray in noramlized_nparrays]
#print(noramzliaed_data_shape_list_for_each_file)
print(len(noramlized_nparrays))


# Build dataset
# X shape = (#batch, #sequence, #feature), Y shape = (#batch, 1)
def Build_dataset(nparrays, seq_length):
    X_data = []
    Y_data = []

    for nparray in nparrays:
        for i in range(0, len(nparray) - seq_length):
            x = nparray[i: i + seq_length, :]
            y = nparray[i + seq_length, [0]]

            X_data.append(x)
            Y_data.append(y)
    return np.array(X_data), np.array(Y_data)


sequence_length = 10
X_train, Y_train = Build_dataset(noramlized_nparrays[:-1], sequence_length)
X_test, Y_test = Build_dataset(noramlized_nparrays[-1:], sequence_length)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

# Change nparray to Tensor
X_train_tensor = torch.from_numpy(X_train).type(torch.Tensor).to(device=device)
Y_train_tensor = torch.from_numpy(Y_train).type(torch.Tensor).to(device=device)
X_test_tensor = torch.from_numpy(X_test).type(torch.Tensor).to(device=device)
Y_test_tensor = torch.from_numpy(Y_test).type(torch.Tensor).to(device=device)

print(X_train_tensor.shape)
print(Y_train_tensor.shape)
print(X_test_tensor.shape)
print(Y_test_tensor.shape)


# Build model
#####################
import torch.nn.functional as F
class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.fc_std = nn.Linear(hidden_dim, output_dim)
        self.fc_v = nn.Linear(hidden_dim, 1)

        self.device = device

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        out = torch.tanh(out)
        return out


# instantiate a moddel
input_dim = 2 * sequence_length
hidden_dim = 256
num_layers = 2
output_dim = 1

#model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
model = GRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
#model = MyNetwork(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

model.to(device=device)
print(model)
#print(next(model.parameters()).device)

# Virtual Line Tracer version 1
def Line_tracer_decision_making(color):
    black = 5
    white = 75
    threshold = (black + white) / 2
    turning_ratio = 30

    denormalized_color = mm.inverse_transform([[color, 0]])[0][0]  # todo
    # print('color:', color)
    # print('denormalized color:', denormalized_color)

    if denormalized_color > threshold:
        None
    elif denormalized_color < threshold:
        turning_ratio = -turning_ratio
    else:
        turning_ratio = 0

    normalized_turning_ratio = mm.transform([[0, turning_ratio]])[0][1]  # todo
    # print('turning:', turning_ratio)
    # print('normalized turning:', normalized_turning_ratio)
    return normalized_turning_ratio


# Train Model
num_epochs = 201
loss_fn = torch.nn.MSELoss()
loss_fn_test = torch.nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
hist = np.zeros(num_epochs)
hist_test = np.zeros(num_epochs)
simulation_init_idx = 300
simulation_time = 500
dagger_original_start_idx = 0
num_DAgger = 20
cur_DAgger = 0
original_num_training_data = X_train_tensor.shape[0]
import time


def input_distance(reference, target):
    # length-aware average diff
    diff = reference - target
    diff = torch.abs(diff)
    diff = torch.sum(diff) / reference.shape[0]
    return diff


for i in range(num_epochs):
    # Forward propagation
    print("train X:", X_train_tensor.shape)
    # print(X_train_tensor)
    y_train_pred = model(X_train_tensor)

    # Loss calculation
    loss = loss_fn(y_train_pred, Y_train_tensor)
    print("Epoch ", i, "train MSE: ", loss.item())
    hist[i] = loss.item()

    y_test_pred = model(X_test_tensor)  # test together
    loss_test = loss_fn_test(y_test_pred, Y_test_tensor)
    hist_test[i] = loss_test.item()

    # Back propagation
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    # #DAgger
    # if loss.item() < 0.015 and cur_DAgger < num_DAgger:
    #   plt.figure(figsize=(10, 5))
    #   plt.plot(X_train_tensor[cur_DAgger, :, [0]].cpu(), label="Original X")
    #   plt.plot(X_train_tensor[dagger_original_start_idx, :, [0]].cpu(), label="DAgger new X")
    #   plt.legend()
    #   plt.show()

    #   dagger_original_end_idx = X_train_tensor.shape[0]

    #   X_new_training_dataset = []
    #   Y_new_training_dataset = []
    #   file_start_idx = dagger_original_start_idx
    #   file_num = 0
    #   for data_shape in data_shape_list_for_each_file[:-1]:
    #     file_end_idx = file_start_idx + data_shape[0] - sequence_length - cur_DAgger
    #     X_prev_training_data = X_train_tensor[file_start_idx:file_end_idx]
    #     Y_prev_training_data = Y_train_tensor[file_start_idx:file_end_idx]
    #     Y_prev_pred_data = y_train_pred.detach()
    #     Y_prev_pred_data = Y_prev_pred_data[file_start_idx:file_end_idx]

    #     turn_ratio_dagger = np.zeros(file_end_idx - file_start_idx)
    #     for t in range(len(turn_ratio_dagger)):
    #       turn_ratio_dagger[t] = Line_tracer_decision_making(Y_prev_pred_data[t])

    #     turn_ratio_dagger = torch.Tensor([turn_ratio_dagger]).to(device=device)
    #     turn_ratio_dagger = torch.reshape(turn_ratio_dagger, (turn_ratio_dagger.shape[1], 1))
    #     error_data = torch.cat((Y_prev_pred_data[:-1], turn_ratio_dagger[:-1]), 1)
    #     error_data = torch.reshape(error_data, (error_data.shape[0], 1, error_data.shape[1]))

    #     X_new_training_data = X_prev_training_data[:-1, :, :]
    #     X_new_training_data = X_new_training_data[:, 1:, :]
    #     X_new_training_data = torch.cat([X_new_training_data, error_data], dim=1)
    #     X_new_training_dataset.append(X_new_training_data)

    #     # Option 1: X_new_training_dataset과 유사한 경험을 찾아서 그것의 Y를 가져다 쓰는 방식을 추가
    #     # prev_time = time.time()
    #     # Y_new_training_data1 = []
    #     # for X_new_training_data in X_new_training_dataset[file_num]:
    #     #   best_fit_idx = -1
    #     #   best_fit_diff = -1
    #     #   for ref_idx in range(X_train_tensor.shape[0]):
    #     #     cur_diff = input_distance(X_new_training_data, X_train_tensor[ref_idx])
    #     #     if best_fit_idx == -1 or best_fit_diff > cur_diff:
    #     #       best_fit_diff = cur_diff
    #     #       best_fit_idx = ref_idx
    #     #   Y_new_training_data1.append(Y_train_tensor[best_fit_idx])
    #     # Y_new_training_data1 = torch.Tensor(Y_new_training_data1).to(device=device)
    #     # Y_new_training_data1 = torch.reshape(Y_new_training_data1, (Y_new_training_data1.shape[0],1))
    #     # print(Y_new_training_data1.shape)
    #     # cur_time = time.time()
    #     # print(cur_time-prev_time)

    #     # Option 2: 기존의 Y를 그대로 사용하는 코드
    #     prev_time = time.time()
    #     Y_new_training_data2 = []
    #     Y_new_training_data2 = Y_prev_training_data[1:, :]
    #     #print(Y_new_training_data2.shape)
    #     #print(Y_new_training_data1.shape)
    #     cur_time = time.time()
    #     #print(cur_time-prev_time)

    #     #print(Y_new_training_data1 - Y_new_training_data2)
    #     #print('diff sum:', torch.sum(Y_new_training_data1 - Y_new_training_data2))

    #     Y_new_training_dataset.append(Y_new_training_data2)

    #     file_start_idx = file_end_idx
    #     file_num = file_num + 1

    #   new_X = torch.cat(X_new_training_dataset, dim=0)
    #   new_Y = torch.cat(Y_new_training_dataset, dim=0)

    #   X_train_tensor = torch.cat([X_train_tensor, new_X], dim=0)
    #   Y_train_tensor = torch.cat([Y_train_tensor, new_Y], dim=0)

    #   cur_DAgger = cur_DAgger + 1
    #   dagger_original_start_idx = dagger_original_end_idx

    # Simulation
    if (i % 1 == 0):
        simulation_data = X_test_tensor[[simulation_init_idx], :, :]
        simulated_environment = np.zeros(simulation_time)
        for j in range(simulation_time):
            Y_simulation_tensor_pred = model(simulation_data)
            next_env_value = Y_simulation_tensor_pred.cpu().detach().numpy()[0][0]
            simulated_environment[j] = next_env_value

            turning_ratio = Line_tracer_decision_making(next_env_value)

            next_data = torch.Tensor([[[next_env_value, turning_ratio]]])
            next_data = next_data.to(device=device)
            simulation_data = torch.cat([simulation_data, next_data], dim=1)
            simulation_data = simulation_data[:, 1:, :]

        plt.figure(figsize=(10, 5))
        plt.plot(Y_test_tensor[simulation_init_idx:simulation_init_idx + simulation_time].cpu(), label="Actual")
        plt.plot(simulated_environment, label="Simulation")
        plt.legend()
        plt.show()