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

## experiment 1
training_log_files = ["data/ver1_fixed_interval/ver1_ft_60_30.csv"]
training_folder_title = "30"
testing_log_files = ["data/ver1_fixed_interval/ver1_ft_60_30.csv"]
testing_folder_title_list = ["30"]

# experiment 2
training_log_files = ["data/ver1_fixed_interval/ver1_ft_60_10 straight cut.csv",
                      "data/ver1_fixed_interval/ver1_ft_60_30.csv",
                      "data/ver1_fixed_interval/ver1_ft_60_50.csv"]
training_folder_title = "103050"
testing_log_files = ["data/ver1_fixed_interval/ver1_ft_60_10 straight cut.csv",
                     "data/ver1_fixed_interval/ver1_ft_60_20 straight cut.csv",
                     "data/ver1_fixed_interval/ver1_ft_60_30.csv",
                     "data/ver1_fixed_interval/ver1_ft_60_40.csv",
                     "data/ver1_fixed_interval/ver1_ft_60_50.csv"]
testing_folder_title_list = ["10", "20", "30", "40", "50"]

episode_lengths = [15, 50, 100]
trials = list(range(1, 11))
#training_episode_length = episode_length
num_training_episodes = list(range(1, 11))
num_testing_episode = 100

state_length = 10
max_epoch = 300

shuffle = True
algorithms = ['bc_gail_ppo', 'gail_ppo', 'bc']#, 'gail_reinforce', 'gail_actor_critic', 'bc_episode', 'bc_stochastic']


num_simulation_repeat = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sdtw = SoftDTW(use_cuda=torch.cuda.is_available(), gamma=0.1)


def read_log_file(training_log_files, testing_log_files):
    # read data
    raw_dfs = []
    for file in training_log_files + testing_log_files:
        raw_dfs.append(pd.read_csv(file, index_col='time'))

    # normalize data
    noramlized_nparrays, scaler = normalize_dataframes_to_nparrays(raw_dfs)

    return noramlized_nparrays[:len(training_log_files)], noramlized_nparrays[len(training_log_files):], scaler


def data_preprocessing(log_dataset, state_length, episode_length, training_episode_length, num_training_episode, num_testing_episode, device, shuffle=True):
    # change data shape
    x_train_data = []
    y_train_data = []
    x_testing_data = []
    y_testing_data = []
    for log_data in log_dataset:
        x, y = build_nparray_dataset_gail(log_data, state_length, episode_length)
        # x_data.append(x)
        # y_data.append(y)
        # x_data = np.concatenate(x_data, axis=0)
        # y_data = np.concatenate(y_data, axis=0)

        # split data
        index_list = list(range(len(x)))
        if shuffle:
            random.shuffle(index_list)

        training_index_list = index_list[:num_training_episode]
        testing_idx = int(len(index_list) - num_testing_episode)
        testing_index_list = index_list[testing_idx:]

        x_train_tensor = torch.tensor(x[training_index_list], dtype=torch.float32, device=device)
        y_train_tensor = torch.tensor(y[training_index_list, :training_episode_length], dtype=torch.float32, device=device)
        x_testing_tensor = torch.tensor(x[testing_index_list], dtype=torch.float32, device=device)
        y_testing_tensor = torch.tensor(y[testing_index_list], dtype=torch.float32, device=device)

        x_train_data.append(x_train_tensor)
        y_train_data.append(y_train_tensor)
        x_testing_data.append(x_testing_tensor)
        y_testing_data.append(y_testing_tensor)

    x_train_data = torch.cat(x_train_data, dim=0)
    y_train_data = torch.cat(y_train_data, dim=0)
    x_testing_data = torch.cat(x_testing_data, dim=0)
    y_testing_data = torch.cat(y_testing_data, dim=0)

    return x_train_data, y_train_data, x_testing_data, y_testing_data


def environment_model_generation(env_model, sut, device, algorithm, training_dataset_x, training_dataset_y, max_epopch, testing_dataset_x_list, testing_dataset_y_list):
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
    evaluation_result = trainer.train(model=env_model, epochs=max_epopch, x=training_dataset_x, y=training_dataset_y, xt=testing_dataset_x_list, yt=testing_dataset_y_list, episode_length=episode_length)
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


for episode_length in episode_lengths:
    training_episode_length = episode_length
    for trial in trials:
        for num_episode in num_training_episodes:
            print("episode length:", episode_length)
            print("num episode:", num_episode)
            print("trial:", trial)

            training_log_dataset, testing_log_dataset, scaler = read_log_file(training_log_files, testing_log_files)

            training_dataset_x, training_dataset_y, _, _ = data_preprocessing(training_log_dataset, state_length, episode_length, training_episode_length, num_episode, num_testing_episode, device, shuffle)
            testing_dataset_x_list = []
            testing_dataset_y_list = []
            for testing_data in testing_log_dataset:
                _, _, testing_dataset_x, testing_dataset_y = data_preprocessing([testing_data], state_length, episode_length, training_episode_length, num_episode, num_testing_episode, device, shuffle)
                testing_dataset_x_list.append(testing_dataset_x)
                testing_dataset_y_list.append(testing_dataset_y)

            line_tracer = LineTracerVer1(scaler)

            #testing_dl = DataLoader(dataset=TensorDataset(testing_dataset_x, testing_dataset_y), batch_size=516, shuffle=False)

            # random model
            print("random model")
            random_model = LineTracerRandomEnvironmentModelDNN(device)
            random_model.to(device)
            random_results = simulation_and_comparison_with_multiple_testing_dataset(random_model, line_tracer, testing_dataset_x_list, testing_dataset_y_list, device)

            field_eval_results = []
            random_eval_results = []
            random_comp_results = []
            for random_result in random_results:
                field_eval_results.append(random_result[0])
                random_eval_results.append(random_result[1])
                random_comp_results.append(random_result[2])


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
            shallow_PR_results = simulation_and_comparison_with_multiple_testing_dataset(shallow_PR_model, line_tracer, testing_dataset_x_list, testing_dataset_y_list, device)

            PR_eval_results = []
            PR_comp_results = []
            for shallow_PR_result in shallow_PR_results:
                PR_eval_results.append(shallow_PR_result[1])
                PR_comp_results.append(shallow_PR_result[2])

            # shallow learning model (Random forest)
            print("shallow model (Random forest)")
            from sklearn.ensemble import RandomForestRegressor

            rf_regressor = RandomForestRegressor(n_estimators=10)
            rf_regressor.fit(flatted_training_x, np_training_y)
            shallow_RF_model = LineTracerShallowRFEnvironmentModel(rf_regressor, device)
            shallow_RF_results = simulation_and_comparison_with_multiple_testing_dataset(shallow_RF_model, line_tracer, testing_dataset_x_list, testing_dataset_y_list, device)

            RF_eval_results = []
            RF_comp_results = []
            for shallow_RF_result in shallow_RF_results:
                RF_eval_results.append(shallow_RF_result[1])
                RF_comp_results.append(shallow_RF_result[2])


            #manual model
            print("manual model")
            manual_model_latency_list = list(range(1, state_length + 1, 2))
            manual_model_unit_diff_list = list(range(1, 16, 2))
            manual_eval_results_list = []
            manual_comp_results_list = []
            for latency in manual_model_latency_list:
                for unit_diff in manual_model_unit_diff_list:
                    print("manual model, latency =", latency, "unit_diff =", unit_diff)
                    manual_model = LineTracerManualEnvironmentModelDNN(latency, unit_diff, scaler, device)
                    manual_model.to(device)
                    manual_results = simulation_and_comparison_with_multiple_testing_dataset(manual_model, line_tracer, testing_dataset_x_list, testing_dataset_y_list, device)
                    temp_eval = []
                    temp_comp = []
                    for manual_result_each_testing_dataset in manual_results:
                        temp_eval.append(manual_result_each_testing_dataset[1])
                        temp_comp.append(manual_result_each_testing_dataset[2])
                    manual_eval_results_list.append(temp_eval)
                    manual_comp_results_list.append(temp_comp)
            manual_eval_results_list = np.array(manual_eval_results_list)
            manual_comp_results_list = np.array(manual_comp_results_list)
            manual_eval_results = np.nanmean(manual_eval_results_list, axis=0).tolist()
            manual_comp_results = np.nanmean(manual_comp_results_list, axis=0).tolist()

            # our model
            input_length = training_dataset_x.shape[1]
            input_feature = training_dataset_x.shape[2]
            hidden_dim = 256
            output_dim = 1
            initial_env_model = LineTracerEnvironmentModelDNN(input_dim=input_length*input_feature, hidden_dim=hidden_dim, output_dim=output_dim, device=device)
            initial_env_model.to(device)

            algo_eval_results_list = []
            algo_comp_results_list = []
            for algo in algorithms:
                print(algo)
                env_model = copy.deepcopy(initial_env_model)
                evaluation_results = environment_model_generation(env_model, line_tracer, device, algo, training_dataset_x, training_dataset_y, max_epoch, testing_dataset_x_list, testing_dataset_y_list,)
                #evaluation_results_for_each_algo.append(evaluation_results)
                print()


                for algo_result_of_epoch in evaluation_results:
                    temp_eval = []
                    temp_comp = []
                    for algo_result_each_testing_dataset in algo_result_of_epoch:
                        temp_eval.append(algo_result_each_testing_dataset[1])
                        temp_comp.append(algo_result_each_testing_dataset[2])
                    algo_eval_results_list.append(temp_eval)
                    algo_comp_results_list.append(temp_comp)

                # titles = ["Euclidian distance", "Dynamic Time Warping",
                #       "Metric1 time diff",
                #       "Metric1 count diff",
                #       "Metric2 undershoot time diff",
                #       "Metric2 undershoot count diff",
                #       "Metric2 overshoot time diff",
                #       "Metric2 overshoot count diff",
                #       "Metric3 undershoot amplitude diff",
                #       "Metric3 overshoot amplitude diff",
                #       "Metric diff average",
                #
                #       "Metric1 time overlapped CI ratio",
                #       "Metric1 count overlapped CI ratio",
                #       "Metric2 undershoot time overlapped CI ratio",
                #       "Metric2 undershoot count overlapped CI ratio",
                #       "Metric2 overshoot time overlapped CI ratio",
                #       "Metric2 overshoot count overlapped CI ratio",
                #       "Metric3 undershoot amplitude overlapped CI ratio",
                #       "Metric3 overshoot amplitude overlapped CI ratio",
                #       "Metric overlapped CI ratio average"]
                #
                # #save & visualize evaluation results
                # np_evaluation_results = np.array(evaluation_results_for_each_algo)
                # for vis_idx in range(20):
                #     plt.figure(figsize=(10, 5))
                #
                #     plt.title(titles[vis_idx])
                #     if vis_idx == 1:
                #         min_dtws = batch_dynamic_time_warping(testing_dataset_y[:, :, [0]], testing_dataset_y[:, :, [0]], 1000)
                #         min_dtws = min_dtws.mean()
                #         plt.plot([min_dtws.cpu().item()] * np_evaluation_results.shape[1], label="field_experiment")
                #     elif vis_idx >= 11:
                #         plt.plot([1.] * np_evaluation_results.shape[1], label="field_experiment")
                #     else:
                #         plt.plot([0.] * np_evaluation_results.shape[1], label="field_experiment")
                #
                #     # random baseline
                #     plt.plot([random_result[vis_idx]] * np_evaluation_results.shape[1], label="random_model")
                #
                #     # manual baseline
                #     plt.plot([manual_result[vis_idx]] * np_evaluation_results.shape[1], label="manual_model")
                #
                #     # shallow learning baseline
                #     plt.plot([shallow_PR_result[vis_idx]] * np_evaluation_results.shape[1], label="shallow_PR_model")
                #     plt.plot([shallow_RF_result[vis_idx]] * np_evaluation_results.shape[1], label="shallow_RF_model")
                #
                #     # algorithms
                #     for i in range(len(evaluation_results_for_each_algo)):
                #         plt.plot(np_evaluation_results[i, :, vis_idx], label=algorithms[i])
                #
                #     plt.legend()
                #     plt.show()


            for testing_dataset_idx in range(len(testing_folder_title_list)):
                # eval output
                field_eval_result = np.array([field_eval_results[testing_dataset_idx]])
                random_eval_result = np.array([random_eval_results[testing_dataset_idx]])
                manual_eval_result = np.array([manual_eval_results[testing_dataset_idx]])
                PR_eval_result = np.array([PR_eval_results[testing_dataset_idx]])
                RF_eval_result = np.array([RF_eval_results[testing_dataset_idx]])

                algo_eval_results_list_of_the_training_dataset = []
                for algo_eval in algo_eval_results_list:
                    algo_eval_results_list_of_the_training_dataset.append(algo_eval[testing_dataset_idx])
                algo_eval_results = np.array(algo_eval_results_list_of_the_training_dataset)

                eval_output_nparray = np.concatenate((field_eval_result, random_eval_result, manual_eval_result, PR_eval_result, RF_eval_result, algo_eval_results), axis=0)
                np.savetxt("output/data/rqAll/" + str(episode_length) + "tick episode/training_" + training_folder_title + "/testing_" +testing_folder_title_list[testing_dataset_idx] + "/eval/episode_" + str(num_episode) + "_trial_" + str(trial) + ".csv", eval_output_nparray, delimiter=",")

                # comp output
                field_dtw_mean = batch_dynamic_time_warping(testing_dataset_y_list[testing_dataset_idx][:, :, [0]], testing_dataset_y_list[testing_dataset_idx][:, :, [0]], 1000).mean()
                field_dtw_mean = field_dtw_mean.cpu().item()
                field_comp_result = np.array([[0, field_dtw_mean] + ([0] * 9)])

                random_comp_result = np.array([random_comp_results[testing_dataset_idx]])
                manual_comp_result = np.array([manual_comp_results[testing_dataset_idx]])
                PR_comp_result = np.array([PR_comp_results[testing_dataset_idx]])
                RF_comp_result = np.array([RF_comp_results[testing_dataset_idx]])

                algo_comp_results_list_of_the_training_dataset = []
                for algo_comp in algo_comp_results_list:
                    algo_comp_results_list_of_the_training_dataset.append(algo_comp[testing_dataset_idx])
                algo_comp_results = np.array(algo_comp_results_list_of_the_training_dataset)

                comp_output_nparray = np.concatenate((field_comp_result, random_comp_result, manual_comp_result, PR_comp_result, RF_comp_result, algo_comp_results), axis=0)
                np.savetxt("output/data/rqAll/" + str(episode_length) +"tick episode/training_" + training_folder_title +"/testing_" + testing_folder_title_list[testing_dataset_idx] +"/comp/episode_" + str(num_episode) +"_trial_" + str(trial) +".csv", comp_output_nparray, delimiter=",")

            print()

