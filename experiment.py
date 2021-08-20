import pandas as pd
import numpy as np
import random
import copy
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

from exp.bc import BehaviorCloning1TickTrainer
from exp.bc_episode import BehaviorCloningEpisodeTrainer
from exp.bc_gail_ppo import BCGailPPOTrainer
from exp.bc_stochastic import BehaviorCloning1TickStochasticTrainer
from exp.evaluation import *
from exp.gail_actor_critic import GailACTrainer
from exp.gail_ppo import GailPPOTrainer
from exp.gail_reinforce import GailREINFORCETrainer
from exp.manual_model import LineTracerManualEnvironmentModelDNN
from exp.random_model import LineTracerRandomEnvironmentModelDNN
from exp.shallow_model import LineTracerShallowPREnvironmentModel, LineTracerShallowRFEnvironmentModel
from exp.util import episode_to_datapoints
from src.dataset_builder import *
from src.line_tracer import LineTracerVer1
from src.line_tracer_env_model import LineTracerEnvironmentModelDNN
from src.soft_dtw_cuda import SoftDTW

log_files = ["data/ver1_fixed_interval/ver1_ft_60_30.csv"]

state_length = 10
episode_length = 250
training_episode_length = 1
shuffle = True
num_training_episode = 10
num_testing_episode = 30

algorithms = ['bc_gail_ppo', 'gail_ppo', 'bc', 'gail_reinforce', 'gail_actor_critic', 'bc_episode', 'bc_stochastic']
max_epoch = 100

num_simulation_repeat = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sdtw = SoftDTW(use_cuda=torch.cuda.is_available(), gamma=0.1)


def read_log_file(log_files):
    # read data
    raw_dfs = []
    for file in log_files:
        raw_dfs.append(pd.read_csv(file, index_col='time'))

    # normalize data
    noramlized_nparrays, scaler = normalize_dataframes_to_nparrays(raw_dfs)

    return noramlized_nparrays, scaler


def data_preprocessing(log_dataset, state_length, episode_length, training_episode_length, num_training_episode, num_testing_episode, device, shuffle=True):
    # change data shape
    x_data = []
    y_data = []
    for log_data in log_dataset:
        x, y = build_nparray_dataset_gail(log_data, state_length, episode_length)
        x_data.append(x)
        y_data.append(y)
    x_data = np.concatenate(x_data, axis=0)
    y_data = np.concatenate(y_data, axis=0)

    # split data
    index_list = list(range(len(x_data)))
    if shuffle:
        random.shuffle(index_list)

    training_index_list = index_list[:num_training_episode]
    testing_idx = int(len(index_list) - num_testing_episode)
    testing_index_list = index_list[testing_idx:]

    x_train_tensor = torch.tensor(x_data[training_index_list], dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_data[training_index_list, :training_episode_length], dtype=torch.float32, device=device)
    x_testing_tensor = torch.tensor(x_data[testing_index_list], dtype=torch.float32, device=device)
    y_testing_tensor = torch.tensor(y_data[testing_index_list], dtype=torch.float32, device=device)

    return x_train_tensor, y_train_tensor, x_testing_tensor, y_testing_tensor


def environment_model_generation(env_model, sut, device, algorithm, training_dataset_x, training_dataset_y, max_epopch, testing_dataset_x, testing_dataset_y):
    learning_rate = 0.00005
    if algorithm == 'bc':
        trainer = BehaviorCloning1TickTrainer(device=device, sut=sut, lr=learning_rate)
    elif algorithm == 'bc_stochastic':
        trainer = BehaviorCloning1TickStochasticTrainer(device=device, sut=sut, lr=learning_rate)
    elif algorithm == 'bc_episode':
        trainer = BehaviorCloningEpisodeTrainer(device=device, sut=sut, lr=learning_rate)
    elif algorithm == 'gail_reinforce':
        trainer = GailREINFORCETrainer(device=device, sut=sut, state_dim=input_feature, action_dim=output_dim, history_length=input_length, lr=learning_rate)
    elif algorithm == 'gail_actor_critic':
        trainer = GailACTrainer(device=device, sut=sut, state_dim=input_feature, action_dim=output_dim, history_length=input_length, lr=learning_rate)
    elif algorithm == 'gail_ppo':
        trainer = GailPPOTrainer(device=device, sut=sut, state_dim=input_feature, action_dim=output_dim, history_length=input_length, lr=learning_rate)
    elif algorithm == 'bc_gail_ppo':
        trainer = BCGailPPOTrainer(device=device, sut=sut, state_dim=input_feature, action_dim=output_dim, history_length=input_length, lr=learning_rate)

    episode_length = testing_dataset_y.shape[1]
    evaluation_result = trainer.train(model=env_model, epochs=max_epopch, x=training_dataset_x, y=training_dataset_y, xt=testing_dataset_x, yt=testing_dataset_y, episode_length=episode_length)
    return evaluation_result


def simulation_testing(models, sut, algorithm, testing_dataset_x, simulation_length, num_simulation_repeat, device):
    sim_results = []
    for env_model in tqdm(models, desc="Testing"):
        env_model.eval()

        repeat_results = []
        for _ in range(num_simulation_repeat):
            y_pred = torch.zeros((testing_dataset_x.shape[0], simulation_length, testing_dataset_x.shape[2]), device=device)
            sim_x = testing_dataset_x
            for sim_idx in range(simulation_length):
                # action choice
                if algorithm == 'bc':
                    action = env_model(sim_x).detach()
                else:
                    action_distribution = env_model.get_distribution(sim_x)
                    action = action_distribution.sample().detach()

                # state transition
                sys_operations = sut.act_sequential(action.cpu().numpy())
                sys_operations = torch.tensor(sys_operations).to(device=device).type(torch.float32)
                next_x = torch.cat((action, sys_operations), dim=1)
                next_x = torch.reshape(next_x, (next_x.shape[0], 1, next_x.shape[1]))
                sim_x = sim_x[:, 1:]
                sim_x = torch.cat((sim_x, next_x), dim=1)
                y_pred[:, sim_idx] = sim_x[:, -1]
            repeat_results.append(y_pred)
        sim_results.append(repeat_results)
    return sim_results


def log_comparison(simulation_log_datasets, real_log_dataset):
    ed_similarity = []
    dtw_similarity = []
    for epoch_log_dataset in tqdm(simulation_log_datasets, desc="Log comparison"):
        epoch_ed_similarity = []
        epoch_dtw_similarity = []
        for single_trial_dataset in epoch_log_dataset:
            simulation_log = single_trial_dataset[:, :, [0]]
            real_log = real_log_dataset[:, :, [0]]
            batch_mean_ed = batch_mean_euclidean_distance(simulation_log, real_log)
            batch_mean_dtw = batch_mean_dynamic_time_warping(simulation_log, real_log)
            epoch_ed_similarity.append(batch_mean_ed)
            epoch_dtw_similarity.append(batch_mean_dtw)
        ed_similarity.append(np.mean(epoch_ed_similarity))
        dtw_similarity.append(np.mean(epoch_dtw_similarity))

        plt.figure(figsize=(10, 5))
        plt.plot(epoch_log_dataset[-1][-1, :, [0]].cpu().detach().numpy(), label="y_pred")
        plt.plot(real_log_dataset[-1, :, [0]].cpu().detach().numpy(), label="y")
        plt.legend()
        plt.show()
    return ed_similarity, dtw_similarity


def batch_mean_euclidean_distance(batch1, batch2):
    diffs = torch.pow(batch1 - batch2, 2)
    diffs = torch.sum(diffs, dim=(1, 2))
    diffs = torch.sqrt(diffs)
    return diffs.mean().item()


def batch_mean_dynamic_time_warping(batch1, batch2):
    diffs = sdtw(batch1, batch2)
    return diffs.mean().item()


def list_mean(input_list):
    return sum(input_list)/len(input_list)


def verification_property_metric_evaluation(log_dataset):
    return None


def verification_result_comparison(simulation_verificataion_result, real_verification_result):
    return None


log_dataset, scaler = read_log_file(log_files)

training_dataset_x, training_dataset_y, testing_dataset_x, testing_dataset_y = data_preprocessing(log_dataset, state_length, episode_length, training_episode_length, num_training_episode, num_testing_episode, device, shuffle)

line_tracer = LineTracerVer1(scaler)

testing_dl = DataLoader(dataset=TensorDataset(testing_dataset_x, testing_dataset_y), batch_size=516, shuffle=False)

# shallow learning model (polynomial regression)
print("shallow model (Polynomial regression)")
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

x_training_datapoints, y_training_datapoints = episode_to_datapoints(training_dataset_x, training_dataset_y)
flatted_training_x = torch.reshape(x_training_datapoints, (x_training_datapoints.shape[0], x_training_datapoints.shape[1] * x_training_datapoints.shape[2])).cpu()
np_training_y = y_training_datapoints[:, 0, 0].cpu().numpy()

poly_features = PolynomialFeatures() #degree=2, include_bias=False)
np_poly_training_x = poly_features.fit_transform(flatted_training_x)
linear_regressor = LinearRegression()
linear_regressor.fit(np_poly_training_x, np_training_y)
shallow_PR_model = LineTracerShallowPREnvironmentModel(linear_regressor, poly_features, device)
shallow_PR_result = simulation_and_comparison(shallow_PR_model, line_tracer, testing_dl, device)


# shallow learning model (Random forest)
print("shallow model (Random forest)")
from sklearn.ensemble import RandomForestRegressor

rf_regressor = RandomForestRegressor()
rf_regressor.fit(flatted_training_x, np_training_y)
shallow_RF_model = LineTracerShallowRFEnvironmentModel(rf_regressor, device)
shallow_RF_result = simulation_and_comparison(shallow_RF_model, line_tracer, testing_dl, device)


# random model
print("random model")
random_model = LineTracerRandomEnvironmentModelDNN(device)
random_model.to(device)
random_result = simulation_and_comparison(random_model, line_tracer, testing_dl, device)

# manual model
print("manual model")
manual_model_latency_list = list(range(1, state_length + 1, 2))
manual_model_unit_diff_list = list(range(1, 16, 2))
manual_results = []
for latency in manual_model_latency_list:
    for unit_diff in manual_model_unit_diff_list:
        print("manual model, latency =", latency, "unit_diff =", unit_diff)
        manual_model = LineTracerManualEnvironmentModelDNN(latency, unit_diff, scaler, device)
        manual_model.to(device)
        manual_results.append(simulation_and_comparison(manual_model, line_tracer, testing_dl, device))
manual_results = np.array(manual_results)
manual_result = np.nanmean(manual_results, axis=0)

# our model
input_length = training_dataset_x.shape[1]
input_feature = training_dataset_x.shape[2]
hidden_dim = 256
output_dim = 1
initial_env_model = LineTracerEnvironmentModelDNN(input_dim=input_length*input_feature, hidden_dim=hidden_dim, output_dim=output_dim, device=device)
initial_env_model.to(device)

evaluation_results_for_each_algo = []
for algo in algorithms:
    print(algo)
    env_model = copy.deepcopy(initial_env_model)
    evaluation_results = environment_model_generation(env_model, line_tracer, device, algo, training_dataset_x, training_dataset_y, max_epoch, testing_dataset_x, testing_dataset_y)
    evaluation_results_for_each_algo.append(evaluation_results)
    print()

    titles = ["Euclidian distance", "Dynamic Time Warping",

              "Metric1 time diff",
              "Metric1 count diff",
              "Metric2 undershoot time diff",
              "Metric2 undershoot count diff",
              "Metric2 overshoot time diff",
              "Metric2 overshoot count diff",
              "Metric3 undershoot amplitude diff",
              "Metric3 overshoot amplitude diff",
              "Metric diff average",

              "Metric1 time overlapped CI ratio",
              "Metric1 count overlapped CI ratio",
              "Metric2 undershoot time overlapped CI ratio",
              "Metric2 undershoot count overlapped CI ratio",
              "Metric2 overshoot time overlapped CI ratio",
              "Metric2 overshoot count overlapped CI ratio",
              "Metric3 undershoot amplitude overlapped CI ratio",
              "Metric3 overshoot amplitude overlapped CI ratio",
              "Metric overlapped CI ratio average"]

    # save & visualize evaluation results
    np_evaluation_results = np.array(evaluation_results_for_each_algo)
    for vis_idx in range(20):
        plt.figure(figsize=(10, 5))

        plt.title(titles[vis_idx])
        if vis_idx == 1:
            min_dtws = batch_dynamic_time_warping(testing_dataset_y[:, :, [0]], testing_dataset_y[:, :, [0]], 1000)
            min_dtws = min_dtws.mean()
            plt.plot([min_dtws.cpu().item()] * np_evaluation_results.shape[1], label="field_experiment")
        elif vis_idx >= 11:
            plt.plot([1.] * np_evaluation_results.shape[1], label="field_experiment")
        else:
            plt.plot([0.] * np_evaluation_results.shape[1], label="field_experiment")

        # random baseline
        plt.plot([random_result[vis_idx]] * np_evaluation_results.shape[1], label="random_model")

        # manual baseline
        plt.plot([manual_result[vis_idx]] * np_evaluation_results.shape[1], label="manual_model")

        # shallow learning baseline
        plt.plot([shallow_PR_result[vis_idx]] * np_evaluation_results.shape[1], label="shallow_PR_model")
        plt.plot([shallow_RF_result[vis_idx]] * np_evaluation_results.shape[1], label="shallow_RF_model")

        # algorithms
        for i in range(len(evaluation_results_for_each_algo)):
            plt.plot(np_evaluation_results[i, :, vis_idx], label=algorithms[i])

        plt.legend()
        plt.show()

# simulation_log_datasets = simulation_testing(models, line_tracer, algo, testing_dataset_x, episode_length, num_simulation_repeat, device)
#
# ed_similarity, dtw_similarity = log_comparison(simulation_log_datasets, testing_dataset_y)
#
# plt.figure(figsize=(10, 5))
# plt.plot(np.array(ed_similarity), label="ed")
# plt.plot(np.array(dtw_similarity), label="dtw")
# plt.legend()
# plt.show()
#
#
# property_metric_evaluation_result_sim = verification_property_metric_evaluation(simulation_log_datasets)
# property_metric_evaluation_result_real = verification_property_metric_evaluation(testing_dataset_y)
#
# verification_similarity = verification_result_comparison(property_metric_evaluation_result_sim, property_metric_evaluation_result_real)
#



