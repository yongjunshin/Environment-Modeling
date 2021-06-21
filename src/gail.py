from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class GailTrainer:
    def __init__(self, device, sut, dim):
        self.device = device
        self.sut = sut
        self.discriminator = Discriminator(dim).to(device=self.device)
        self.value = ValueNet(dim).to(device=self.device)

        self.optimiser_d = torch.optim.Adam(self.discriminator.parameters(), lr=0.005)
        self.optimiser_v = torch.optim.Adam(self.value.parameters(), lr=0.00001)

        self.disc_iter = 2
        self.disc_loss = nn.MSELoss()
        self.ppo_iter = 3

        self.gamma = 0.98
        self.lmbda = 0.95
        self.eps_clip = 0.1

    def train(self, model: torch.nn.Module, epochs: int, train_dataloaders: list, validation_dataloaders: list):
        self.optimiser_pi = torch.optim.Adam(model.parameters(), lr=0.005)

        num_batch = sum([len(dl) for dl in train_dataloaders])
        for i in tqdm(range(epochs), desc="Epochs:"):
            loader_idx = 0
            for dataloader in train_dataloaders:
                for batch_idx, (x, y) in enumerate(dataloader):
                    sim_length = y.shape[1]
                    rand_idx = np.random.randint(x.shape[0])
                    state_action = x[rand_idx]
                    y_pred = torch.zeros(y.shape, device=self.device)
                    sim_x = x
                    exp_states_actions = []
                    pi_states_actions = []
                    pi_states_actions_prime = []
                    pi_action_prop = []
                    pi_rewards = []
                    values = []
                    # simulation of current policy
                    for sim_idx in range(sim_length):
                        exp_states_actions.append(torch.cat((x[rand_idx][sim_idx + 1:], y[rand_idx][:sim_idx + 1])))

                        model.train()
                        pi_states_actions.append(state_action)
                        action_prop = model.get_distribution(torch.reshape(state_action, (1, state_action.shape[0], state_action.shape[1])))
                        action = action_prop.sample().detach()
                        pi_action_prop.append(action_prop.log_prob(action))

                        action = action.cpu().detach().numpy()
                        state = self.sut.act(action)
                        next_state_action = np.concatenate((action, np.array([[state]])), axis=1)
                        next_state_action = torch.tensor(next_state_action).to(device=self.device).type(torch.float32)
                        state_action = torch.cat((state_action[1:], next_state_action))
                        pi_states_actions_prime.append(state_action)

                        # get Discriminator reward
                        reward = self.get_reward(torch.reshape(state_action, (1, state_action.shape[0], state_action.shape[1])))
                        pi_rewards.append(reward)

                        # get Value
                        value = self.value(torch.reshape(state_action, (1, state_action.shape[0], state_action.shape[1])))
                        values.append(value)
                    # training
                    self.train_discriminator(exp_states_actions, pi_states_actions_prime)
                    self.train_policy_value_net(model, pi_states_actions, pi_rewards, pi_states_actions_prime, pi_action_prop)

                    if batch_idx == 0 and loader_idx == 0:
                        plt.figure(figsize=(10, 5))
                        plt.plot(pi_states_actions[-1][:, [0]].cpu().detach().numpy(), label="y_pred")
                        plt.plot(y[0, :, [0]].cpu().detach().numpy(), label="y")
                        plt.legend()
                        plt.show()
                        #plt.savefig('output/imgs/episode_pcc/fig' + str(i) + '.png', dpi=300)

    def train_discriminator(self, exp_trajectory, pi_trajectory):
        trajectories = torch.stack(exp_trajectory + pi_trajectory)
        exp_trajectory_label = torch.ones(len(exp_trajectory))
        pi_trajectory_label = torch.zeros(len(pi_trajectory))
        labels = torch.cat((exp_trajectory_label, pi_trajectory_label)).to(device=self.device).type(torch.float32)
        #labels = torch.tensor(exp_trajectory_label + pi_trajectory_label).to(device=self.device).type(torch.float32)
        labels = torch.reshape(labels, (labels.shape[0], 1))

        for i in range(self.disc_iter):
            judges = self.discriminator(trajectories)
            loss = self.disc_loss(judges, labels)
            #print(f"D loss: {loss.mean()}")

            self.optimiser_d.zero_grad()
            loss.mean().backward()
            self.optimiser_d.step()

    def train_policy_value_net(self, model, states_actions, rewards, states_actions_prime, probs):
        states_actions = torch.stack(states_actions)
        rewards = torch.cat(rewards)
        states_actions_prime = torch.stack(states_actions_prime)
        old_probs = torch.cat(probs)

        for i in range(self.ppo_iter):
            td_target = rewards + self.gamma * self.value(states_actions_prime)
            delta = td_target - self.value(states_actions)
            delta = delta.cpu().detach().numpy()

            advantage_list = []
            advantage = 0
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                advantage_list.append([advantage])
            advantage_list.reverse()
            advantage = torch.tensor(advantage_list, dtype=torch.float32, device=self.device)

            new_dist = model.get_distribution(states_actions)
            old_actions = states_actions_prime[:, -1, [1]]  #실제 했던 액션
            new_probs = new_dist.log_prob(old_actions)
            ratio = torch.exp(new_probs - old_probs.detach())

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
            pi_loss = -torch.min(surr1, surr2)
            #value_loss = F.smooth_l1_loss(td_target.detach(), self.value(states_actions))
            value_loss = (td_target.detach() - self.value(states_actions)).pow(2)
            # print(f"pi loss: {pi_loss.mean()}")
            # print(f"value loss: {value_loss.mean()}")

            self.optimiser_pi.zero_grad()
            self.optimiser_v.zero_grad()
            pi_loss.mean().backward()
            value_loss.mean().backward()
            self.optimiser_pi.step()
            self.optimiser_v.step()


    def get_reward(self, state_action):
        reward = self.discriminator.forward(state_action)
        reward = - reward.log()
        return reward.detach()



class ValueNet(nn.Module):
    def __init__(self, input_dim):
        super(ValueNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = 1

        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.output_dim)

    def forward(self, input):
        input = torch.reshape(input, (input.shape[0], input.shape[1] * input.shape[2]))
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        input = torch.reshape(input, (input.shape[0], input.shape[1] * input.shape[2]))
        return self.model(input)

