import numpy as np
import pandas as pd
import torch  # pytorch
import torch.nn as nn
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import math

# initialize argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-r", "--real_env_info_file", type=str, help="Environment characteristics file", default='output/environment characteristics.csv')
parser.add_argument("-s", "--standard_or_heuristic", type=str, help="standard or heuristic", default='heuristic')

args = parser.parse_args()

# Read real environment information files
real_env_info_file = args.real_env_info_file
env_info_df = pd.read_csv(real_env_info_file)

# Read reference environment information
env_data_hist = env_info_df['env_data_hist'].dropna().to_numpy()
env_data_edge = env_info_df['env_data_edge'].dropna().to_numpy()
env_data_min = env_info_df['env_data_min'].dropna().to_numpy()[0]
env_data_mean = env_info_df['env_data_mean'].dropna().to_numpy()[0]
env_data_max = env_info_df['env_data_max'].dropna().to_numpy()[0]
diff_hist = env_info_df['diff_hist'].dropna().to_numpy()
diff_edge = env_info_df['diff_edge'].dropna().to_numpy()
auto_correlation_arr = env_info_df['auto_correlation_arr'].dropna().to_numpy()
auto_correlation_idx = env_info_df['auto_correlation_idx'].dropna().to_numpy()
trend = env_info_df['trend'].dropna().to_numpy()
change_period_hist = env_info_df['change_period_hist'].dropna().to_numpy()
change_period_edge = env_info_df['change_period_edge'].dropna().to_numpy()
amplitude_hist = env_info_df['amplitude_hist'].dropna().to_numpy()
amplitude_edge = env_info_df['amplitude_edge'].dropna().to_numpy()

# Read simulation result environment time-series file
standard_or_heuristic = args.standard_or_heuristic


def get_histogram(data, edge):
    hist = [0] * (len(edge)-1)
    for value in data:
        for edge_idx in range(1, len(edge)):
            if ((edge_idx < len(edge)-1 and edge[edge_idx - 1] <= value < edge[edge_idx]) or
                    (edge_idx == len(edge)-1 and edge[edge_idx - 1] <= value <= edge[edge_idx])):
                hist[edge_idx - 1] = hist[edge_idx - 1] + 1
                break
    return np.array(hist)/len(data)

def KLD(reference_hist, target_hist):
    kld = []
    for kld_idx in range(len(reference_hist)):
        if reference_hist[kld_idx] == 0:
            kld.append(0)
        else:
            kld.append(math.log(reference_hist[kld_idx]/target_hist[kld_idx]) * reference_hist[kld_idx])
    return sum(kld)

def JSD(reference_hist, target_hist):
    jsd = 0.5 * KLD(target_hist, (target_hist + reference_hist) / 2) + 0.5 * KLD(reference_hist,
                                                                           (target_hist + reference_hist) / 2)
    if np.isnan(jsd):
        print("nan")
    return jsd



for trial in range(1, 4):
    value_hist_divergence = []  # done
    boundary_test = []  # done
    diff_hist_divergence = []  # done
    auto_correlation_mse = []  # done
    mean_abs_error = []  # done
    change_period_hist_divergence = []
    amplitude_hist_divergence = []

    for epoch in range(1, 300, 10):
        sim_file_name = 'data/sim_env/' + standard_or_heuristic + ' dagger 210520/trial ' + str(trial) + '/' + standard_or_heuristic + '_' + str(epoch) + 'th.csv'
        print(sim_file_name)
        sim_time_series_df = pd.read_csv(sim_file_name)

        # get simulation environment data
        sim_env_data = sim_time_series_df.to_numpy()[:, 0]
        sim_env_hist = get_histogram(sim_env_data, env_data_edge)
        value_hist_divergence.append(JSD(env_data_hist, sim_env_hist))


        # boudnary test
        if np.min(sim_env_data) >= env_data_min and np.max(sim_env_data) <= env_data_max:
            boundary_test.append(0)
        else:
            boundary_test.append(1)

        # sim env diff data
        sim_diff_array = sim_env_data[1:] - sim_env_data[:-1]
        sim_env_diff_hist = get_histogram(sim_diff_array, diff_edge)
        diff_hist_divergence.append(JSD(diff_hist, sim_env_diff_hist))


        # sim auto correlation
        sim_auto_correlation_array = []
        for shift in auto_correlation_idx:
            shift = int(shift)
            corr = np.corrcoef(sim_env_data[:-shift], sim_env_data[shift:])
            if np.isnan(corr[0, 1]):
                print("nan")
            sim_auto_correlation_array.append(corr[0, 1])
        autocorr_mse = np.square(np.array(sim_auto_correlation_array) - auto_correlation_arr).mean()


        auto_correlation_mse.append(autocorr_mse)

        # mean test
        mean_abs_error.append(abs(np.mean(sim_env_data) - env_data_mean))

        # change point
        sim_change_point_times = []
        sim_change_point_values = []

        if sim_env_data[1] > sim_env_data[0]:
            prev_increasing = True
        elif sim_env_data[1] == sim_env_data[0]:
            prev_increasing = None
        else:
            prev_increasing = False

        for i in range(1, sim_env_data.shape[0] - 1):
            if sim_env_data[i + 1] > sim_env_data[i]:
                cur_increasing = True
            elif sim_env_data[i + 1] == sim_env_data[i]:
                cur_increasing = prev_increasing
            else:
                cur_increasing = False

            if prev_increasing is not None:
                if prev_increasing != cur_increasing:
                    sim_change_point_times.append(i)
                    sim_change_point_values.append(sim_env_data[i])
            prev_increasing = cur_increasing

        sim_change_point_times = np.array(sim_change_point_times)
        sim_change_point_values = np.array(sim_change_point_values)

        sim_cycle_period_array = sim_change_point_times[1:] - sim_change_point_times[:-1]
        sim_amplitude_array = sim_change_point_values[1:] - sim_change_point_values[:-1]


        sim_env_change_period_hist = get_histogram(sim_cycle_period_array, change_period_edge)
        change_period_hist_divergence.append(JSD(change_period_hist, sim_env_change_period_hist))

        sim_amplitude_hist = get_histogram(sim_amplitude_array, amplitude_edge)
        amplitude_hist_divergence.append(JSD(amplitude_hist, sim_amplitude_hist))


    fig, axs = plt.subplots(7, figsize=(10, 20))
    fontsize = 12
    fig.suptitle(standard_or_heuristic + ' trial ' + str(trial), fontweight="bold", size=fontsize+3)

    plt.subplots_adjust(top=0.85)

    axs[0].plot(boundary_test)
    axs[0].set_ylabel("Boundary test result\n (0=True)", fontsize=fontsize)

    axs[1].plot(auto_correlation_mse)
    axs[1].set_ylabel("auto-correlation MSE", fontsize=fontsize)

    axs[2].plot(mean_abs_error)
    axs[2].set_ylabel("mean absolute error", fontsize=fontsize)

    axs[3].plot(value_hist_divergence)
    axs[3].set_ylabel("env value distribution \n JSD", fontsize=fontsize)

    axs[4].plot(diff_hist_divergence)
    axs[4].set_ylabel("1-tick env value diff \n distribution JSD", fontsize=fontsize)

    axs[5].plot(change_period_hist_divergence)
    axs[5].set_ylabel("inflection point period \n distribution JSD", fontsize=fontsize)

    axs[6].plot(amplitude_hist_divergence)
    axs[6].set_ylabel("inflection point \n env value diff JSD", fontsize=fontsize)

    plt.xlabel("epochs/10", fontsize=fontsize)
    fig.show()


