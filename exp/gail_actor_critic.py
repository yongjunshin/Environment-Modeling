import random

from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchviz import make_dot

from exp.evaluation import *


class GailACTrainer:
    def __init__(self, device, sut, state_dim, action_dim, history_length, lr):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.history_length = history_length

        self.lr = lr

        self.device = device
        self.sut = sut
        self.discriminator = Discriminator(state_dim * history_length + action_dim).to(device=self.device)

        self.optimiser_d = torch.optim.Adam(self.discriminator.parameters(), lr=0.001)

        self.disc_iter = 1
        self.disc_loss = nn.MSELoss()

        self.gamma = 0.99

    def train(self, model: torch.nn.Module, epochs: int, x: torch.tensor, y: torch.tensor, xt: torch.tensor, yt: torch.tensor) -> list:
        self.optimiser_pi = torch.optim.Adam(model.parameters(), lr=self.lr)

        evaluation_results = []
        dl = DataLoader(dataset=TensorDataset(x, y), batch_size=516, shuffle=True)
        testing_dl = DataLoader(dataset=TensorDataset(xt, yt), batch_size=516, shuffle=True)

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
                    sim_x = sim_x[:, 1:]
                    sim_x = torch.cat((sim_x, next_x), dim=1)

                pi_states = torch.cat(pi_states)
                pi_actions = torch.cat(pi_actions)
                # if batch_idx == 0:
                #     print("expert judge: ", self.discriminator(x_batch, y_batch[:, 0, [0]]).mean(), "model judge: ", self.discriminator(x_batch, action).mean())
                #     print("expert reward: ", self.get_reward(x_batch, y_batch[:, 0, [0]]).mean(), "model reward: ", self.get_reward(x_batch, action).mean())
                self.train_discriminator(x_batch, y_batch[:, 0, [0]], pi_states, pi_actions)


                # Policy training
                model.train()
                self.discriminator.eval()

                y_pred = torch.zeros(y_batch.shape, device=self.device)
                sim_x = x_batch
                losses = []
                for sim_idx in range(y_batch.shape[1]):
                    # value estimate
                    cur_v = model.v(sim_x)

                    # action choice
                    action_prob = model.get_distribution(sim_x)
                    action = action_prob.sample().detach()
                    prob = action_prob.log_prob(action)

                    # get reward
                    reward = self.get_reward(sim_x, action).detach()

                    delta = reward + self.gamma * model.v(sim_x) - cur_v
                    loss = -prob * delta.detach() + torch.abs(delta)
                    losses.append(loss)


                    # state transition
                    sys_operations = self.sut.act_sequential(action.cpu().numpy())
                    sys_operations = torch.tensor(sys_operations).to(device=self.device).type(torch.float32)
                    next_x = torch.cat((action, sys_operations), dim=1)
                    next_x = torch.reshape(next_x, (next_x.shape[0], 1, next_x.shape[1]))
                    sim_x = sim_x[:, 1:]
                    sim_x = torch.cat((sim_x, next_x), dim=1)
                    y_pred[:, sim_idx] = sim_x[:, -1]

                self.train_policy_value_net(model, losses)

                # if batch_idx == 0:
                #     plt.figure(figsize=(10, 5))
                #     plt.plot(y_pred[0, :, [0]].cpu().detach().numpy(), label="y_pred")
                #     plt.plot(y_batch[0, :, [0]].cpu().detach().numpy(), label="y")
                #     plt.legend()
                #     plt.show()

            evaluation_results.append(simulation_and_comparison(model, self.sut, testing_dl, self.device))
        return evaluation_results

    def train_discriminator(self, exp_state, exp_action, pi_states, pi_actions):
        index_list = list(range(len(exp_state)))
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
            self.optimiser_d.step()
        #print("loss: ", l)

    def train_policy_value_net(self, model, losses):
        self.optimiser_pi.zero_grad()
        for loss in losses:
            loss.mean().backward()
            #make_dot(loss.mean(), params=dict(model.named_parameters())).render(
             #   "graph", format="png")
        self.optimiser_pi.step()

    def get_reward(self, state, action):
        reward = self.discriminator.forward(state, action)
        reward = -reward.log()
        return reward.detach()


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, state, action):
        state = torch.reshape(state, (state.shape[0], state.shape[1] * state.shape[2]))
        input = torch.cat([state, action], dim=1)
        return self.model(input)
