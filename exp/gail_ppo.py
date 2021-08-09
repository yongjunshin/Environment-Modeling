import random

from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from exp.evaluation import *


class GailPPOTrainer:
    def __init__(self, device, sut, state_dim, action_dim, history_length, lr):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.history_length = history_length

        self.lr = lr

        self.device = device
        self.sut = sut
        self.discriminator = Discriminator(state_dim * history_length + action_dim).to(device=self.device)

        self.optimiser_d = torch.optim.Adam(self.discriminator.parameters(), lr=0.01)

        self.disc_iter = 10
        self.disc_loss = nn.MSELoss()

        self.ppo_iter = 10

        self.gamma = 0.99
        self.lmbda = 0.95
        self.eps_clip = 0.2

    def train(self, model: torch.nn.Module, epochs: int, x: torch.tensor, y: torch.tensor, xt: torch.tensor, yt: torch.tensor) -> list:
        self.optimiser_pi = torch.optim.Adam(model.parameters(), lr=self.lr)

        evaluation_results = []
        dl = DataLoader(dataset=TensorDataset(x, y), batch_size=512, shuffle=True)
        testing_dl = DataLoader(dataset=TensorDataset(xt, yt), batch_size=512, shuffle=True)

        # initial model
        evaluation_results.append(simulation_and_comparison(model, self.sut, testing_dl, self.device))

        for _ in tqdm(range(epochs), desc="Training"):
            ed_sum = torch.zeros((), device=self.device)
            dtw_sum = torch.zeros((), device=self.device)
            for batch_idx, (x_batch, y_batch) in enumerate(dl):
                # Discriminator training
                model.eval()
                self.discriminator.train()

                pi_states = []
                pi_actions = []
                sim_x = x_batch
                for sim_idx in range(y_batch.shape[1]):
                    pi_states.append(sim_x)
                    # action choice
                    action_prob = model.get_distribution(sim_x)
                    action = action_prob.sample().detach()
                    pi_actions.append(action)

                    # state transition
                    sys_operations = self.sut.act_sequential(action.cpu().numpy())
                    sys_operations = torch.tensor(sys_operations).to(device=self.device).type(torch.float32)
                    next_x = torch.cat((action, sys_operations), dim=1)
                    next_x = torch.reshape(next_x, (next_x.shape[0], 1, next_x.shape[1]))
                    time_col = torch.reshape(y_batch[:, sim_idx, 2], (next_x.shape[0], 1, 1))
                    next_x = torch.cat((next_x, time_col), dim=2)
                    sim_x = sim_x[:, 1:]
                    sim_x = torch.cat((sim_x, next_x), dim=1)

                pi_states = torch.cat(pi_states)
                pi_actions = torch.cat(pi_actions)
                if batch_idx == 0:
                    print("before D learn")
                    print("expert judge: ", self.discriminator(x_batch, y_batch[:, 0, [0]]).mean(), "model judge: ",
                          self.discriminator(pi_states, pi_actions).mean())
                    print("expert reward: ", self.get_reward(x_batch, y_batch[:, 0, [0]]).mean(), "model reward: ",
                          self.get_reward(pi_states, pi_actions).mean())
                self.train_discriminator(x_batch, y_batch[:, 0, [0]], pi_states, pi_actions)
                if batch_idx == 0:
                    print("after D learn")
                    print("expert judge: ", self.discriminator(x_batch, y_batch[:, 0, [0]]).mean(), "model judge: ",
                          self.discriminator(pi_states, pi_actions).mean())
                    print("expert reward: ", self.get_reward(x_batch, y_batch[:, 0, [0]]).mean(), "model reward: ",
                          self.get_reward(pi_states, pi_actions).mean())

                # Policy training
                model.train()
                self.discriminator.eval()

                y_pred = torch.zeros(y_batch.shape, device=self.device)
                sim_x = x_batch
                rewards = []
                probs = []
                states = []
                states_prime = []
                actions = []
                for sim_idx in range(y_batch.shape[1]):
                    # action choice
                    action_distribution = model.get_distribution(sim_x)
                    states.append(sim_x)
                    action = action_distribution.sample().detach()
                    actions.append(action)
                    prob = action_distribution.log_prob(action)
                    probs.append(prob)

                    # get reward
                    reward = self.get_reward(sim_x, action).detach()
                    # if sim_idx == 0:
                    #     print("0 epoch reward mean:", reward.mean())
                    rewards.append(reward)

                    # state transition
                    sys_operations = self.sut.act_sequential(action.cpu().numpy())
                    sys_operations = torch.tensor(sys_operations).to(device=self.device).type(torch.float32)
                    next_x = torch.cat((action, sys_operations), dim=1)
                    next_x = torch.reshape(next_x, (next_x.shape[0], 1, next_x.shape[1]))
                    time_col = torch.reshape(y_batch[:, sim_idx, 2], (next_x.shape[0], 1, 1))
                    next_x = torch.cat((next_x, time_col), dim=2)
                    sim_x = sim_x[:, 1:]
                    sim_x = torch.cat((sim_x, next_x), dim=1)
                    y_pred[:, sim_idx] = sim_x[:, -1]
                    states_prime.append(sim_x)

                self.train_policy_value_net(model, states, states_prime, actions, probs, rewards)

                # if batch_idx == 0:
                #     plt.figure(figsize=(10, 5))
                #     plt.plot(y_pred[0, :, [0]].cpu().detach().numpy(), label="y_pred")
                #     plt.plot(y_batch[0, :, [0]].cpu().detach().numpy(), label="y")
                #     plt.legend()
                #     plt.show()
            evaluation_results.append(simulation_and_comparison(model, self.sut, testing_dl, self.device))
        return evaluation_results

    def train_discriminator(self, exp_state, exp_action, pi_states, pi_actions):
        index_list = list(range(len(pi_states)))
        random.shuffle(index_list)
        index_list = index_list[:len(exp_state)]
        pi_states = pi_states[index_list]
        pi_actions = pi_actions[index_list]

        states = torch.cat([exp_state, pi_states], dim=0)
        actions = torch.cat([exp_action, pi_actions], dim=0)
        exp_trajectory_label = torch.zeros(len(exp_action))
        pi_trajectory_label = torch.ones(len(pi_actions))
        labels = torch.cat((exp_trajectory_label, pi_trajectory_label)).to(device=self.device).type(torch.float32)
        labels = torch.reshape(labels, (labels.shape[0], 1))

        index_list = list(range(len(states)))
        l = 0
        for i in range(self.disc_iter):
            random.shuffle(index_list)
            judges = self.discriminator(states[index_list], actions[index_list])
            loss = self.disc_loss(judges, labels[index_list])
            l = l + loss.item()

            self.optimiser_d.zero_grad()
            loss.backward()
            print("disc loss", loss)
            self.optimiser_d.step()
        # print("loss: ", l)

    def train_policy_value_net(self, model, states, states_prime, actions, probs, rewards):
        steps = len(states)
        batches = len(states[0])
        probs = torch.cat(probs, dim=1).detach()
        probs = torch.reshape(probs, (probs.shape[0] * probs.shape[1], 1))

        # reducing dimension for parallel calculation
        t_states = torch.stack(states)
        t_states = torch.reshape(t_states,
                                 (t_states.shape[0] * t_states.shape[1], t_states.shape[2], t_states.shape[3]))
        t_states_prime = torch.stack(states_prime)
        t_states_prime = torch.reshape(t_states_prime, (
        t_states_prime.shape[0] * t_states_prime.shape[1], t_states_prime.shape[2], t_states_prime.shape[3]))
        t_rewards = torch.stack(rewards)
        t_rewards = torch.reshape(t_rewards, (t_rewards.shape[0] * t_rewards.shape[1], 1))
        t_actions = torch.stack(actions)
        t_actions = torch.reshape(t_actions, (t_actions.shape[0] * t_actions.shape[1], 1))

        for _ in range(self.ppo_iter):
            # calculating advantage
            td_target = t_rewards + self.gamma * model.v(t_states_prime)
            v = model.v(t_states)
            delta = td_target - v
            delta = torch.reshape(delta, (steps, batches, 1)).detach()
            deltas = [delta[i] for i in range(len(delta))]

            advantage_list = []
            advantage = torch.zeros((len(deltas[0]), 1), device=self.device)
            for delta in deltas[::-1]:
                advantage = delta + self.gamma * self.lmbda * advantage
                advantage_list.append(advantage)
            advantage_list.reverse()
            advantage_list = torch.stack(advantage_list)
            advantage_list = torch.reshape(advantage_list, (advantage_list.shape[0] * advantage_list.shape[1], 1))


            # calculating action probability ratio
            cur_distribution = model.get_distribution(t_states)
            cur_probs = cur_distribution.log_prob(t_actions)
            ratio = torch.exp(cur_probs - probs)

            #print("advantage:", advantage_list.mean())
            surr1 = ratio * advantage_list
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage_list
            loss_clip = -torch.min(surr1, surr2)
            loss_value = F.smooth_l1_loss(td_target, v)
            loss = loss_clip + loss_value

            if torch.isnan(loss).any():
                print(loss)
                None

            print("policy loss", loss.mean())
            self.optimiser_pi.zero_grad()
            loss.mean().backward()
            self.optimiser_pi.step()

    def get_reward(self, state, action):
        reward = self.discriminator.forward(state, action)
        reward = -reward.log()
        reward = torch.nan_to_num(reward)
        reward = torch.tanh(reward) #되나?
        return reward.detach()


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim

        self.fc1 = nn.Linear(self.input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        state = torch.reshape(state, (state.shape[0], state.shape[1] * state.shape[2]))
        input = torch.cat([state, action], dim=1)

        out = torch.nan_to_num(self.fc1(input))
        out = torch.relu(out)
        out = torch.nan_to_num(self.fc2(out))
        out = torch.relu(out)
        out = torch.nan_to_num(self.fc3(out))
        out = torch.sigmoid(out)

        return out
