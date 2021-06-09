import numpy as np
import pandas as pd
import torch  # pytorch
import torch.nn as nn
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd


# initialize argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--num_lines", type=int, help="Number of lines to read from the data file", default=None)
parser.add_argument("-s", "--max_shift", type=int, help="Length of a history window length", default=100)
parser.add_argument("-b", "--bin_size", type=int, help="Number of histogram bins", default=10)
args = parser.parse_args()

# Read files
raw_df = pd.read_csv('data/ver1_fixed_interval/ver1_ft_60_30.csv', index_col='time', nrows=args.num_lines)
print("raw_df.shape:", raw_df.shape)

# Normalize data
from sklearn.preprocessing import StandardScaler, MinMaxScaler

mm = MinMaxScaler(feature_range=(-1, 1))
normalized_nparray = mm.fit_transform(raw_df)
print("normalized_nparray.shape:", normalized_nparray.shape)

bin_size = args.bin_size
env_data = normalized_nparray[:, 0]
env_data_hist = np.histogram(env_data, bins=bin_size)
plt.figure(figsize=(5, 5))
plt.hist(env_data, bins=bin_size)
plt.xlabel("env data value")
plt.ylabel("num data")
plt.show()

min_value = np.min(env_data)
print("min value:", min_value)

mean_value = np.mean(env_data)
print("mean value:", mean_value)

max_value = np.max(env_data)
print("max value:", max_value)

diff_array = env_data[1:] - env_data[:-1]
diff_hist = np.histogram(diff_array, bins=bin_size)
plt.figure(figsize=(5, 5))
plt.hist(diff_array, bins=bin_size)
plt.xlabel("env 1-tick diff")
plt.ylabel("num diff")
plt.show()

max_shift = args.max_shift
auto_correlation_array = []
lag_arr = range(1, max_shift + 1)
for shift in lag_arr:
    auto_correlation_array.append(np.corrcoef(env_data[:-shift], env_data[shift:])[0, 1])
plt.figure(figsize=(5, 5))
plt.plot(auto_correlation_array)
plt.xlabel("lag")
plt.ylabel("pearson correlation")
plt.show()

time = np.array([i for i in range(len(env_data))])
p = np.polyfit(time, env_data, 1)
trend = np.polyval(p, time)
plt.figure(figsize=(5, 5))
plt.plot(time, trend)
plt.xlabel("time")
plt.ylabel("env trend value")
plt.show()

change_point_times = []
change_point_values = []

if env_data[1] > env_data[0]:
    prev_increasing = True
elif env_data[1] == env_data[0]:
    prev_increasing = None
else:
    prev_increasing = False

for i in range(1, env_data.shape[0]-1):
    if env_data[i+1] > env_data[i]:
        cur_increasing = True
    elif env_data[i+1] == env_data[i]:
        cur_increasing = prev_increasing
    else:
        cur_increasing = False

    if prev_increasing is not None:
        if prev_increasing != cur_increasing:
            change_point_times.append(i)
            change_point_values.append(env_data[i])
    prev_increasing = cur_increasing

change_point_times = np.array(change_point_times)
change_point_values = np.array(change_point_values)

cycle_period_array = change_point_times[1:] - change_point_times[:-1]
cycle_period_hist = np.histogram(cycle_period_array, bins=bin_size)
plt.figure(figsize=(5, 5))
plt.hist(cycle_period_array, bins=bin_size)
plt.xlabel("inflection point period")
plt.ylabel("num cycle")
plt.show()

amplitude_array = change_point_values[1:] - change_point_values[:-1]
amplitude_hist = np.histogram(amplitude_array, bins=bin_size)
plt.figure(figsize=(5, 5))
plt.hist(amplitude_array, bins=bin_size)
plt.xlabel("inflection point amplitude")
plt.ylabel("num cycle")
plt.show()

output_df = pd.DataFrame.from_dict({"env_data_hist": env_data_hist[0]/len(env_data), "env_data_edge": env_data_hist[1],
                          "env_data_min": [min_value],
                          "env_data_mean": [mean_value],
                          "env_data_max": [max_value],
                          "diff_hist": diff_hist[0]/len(diff_array), "diff_edge": diff_hist[1],
                          "auto_correlation_arr": auto_correlation_array, "auto_correlation_idx": lag_arr,
                          "trend": p,
                          "change_period_hist": cycle_period_hist[0] / len(cycle_period_array), "change_period_edge": cycle_period_hist[1],
                          "amplitude_hist": amplitude_hist[0] / len(amplitude_array), "amplitude_edge": amplitude_hist[1]}, orient='index')
output_df = output_df.transpose()
output_df.to_csv("output/environment characteristics.csv", index=False)