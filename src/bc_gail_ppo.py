from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchviz import make_dot


class BCGailPPOTrainer:
    def __init__(self, device, sut, state_dim, action_dim, history_length):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.history_length = history_length

        self.bc_loss_fn = torch.nn.MSELoss()

        self.device = device
        self.sut = sut
        self.discriminator = Discriminator((state_dim + action_dim) * history_length).to(device=self.device)

        self.optimiser_d = torch.optim.Adam(self.discriminator.parameters(), lr=0.001)

        self.disc_iter = 4
        self.disc_loss = nn.MSELoss()

        self.ppo_iter = 3

        self.gamma = 0.99
        self.lmbda = 0.95
        self.eps_clip = 0.2

    def train(self, model: torch.nn.Module, epochs: int, train_dataloaders: list, validation_dataloaders: list):
        self.optimiser_pi = torch.optim.Adam(model.parameters(), lr=0.001)


        num_batch = sum([len(dl) for dl in train_dataloaders])
        for i in tqdm(range(epochs), desc="Epochs:"):
            loader_idx = 0
            for dataloader in train_dataloaders:
                for batch_idx, (x, y) in enumerate(dataloader):
                    # Discriminator training
                    model.eval()
                    self.discriminator.train()

                    y_pred = torch.zeros(y.shape, device=self.device)
                    sim_x = x
                    for sim_idx in range(self.history_length):
                        # action choice
                        action_distribution = model.get_distribution(sim_x)
                        action = action_distribution.sample().detach()

                        # state transition
                        sys_operations = self.sut.act_sequential(action.cpu().numpy())
                        sys_operations = torch.tensor(sys_operations).to(device=self.device).type(torch.float32)
                        next_x = torch.cat((action, sys_operations), dim=1)
                        next_x = torch.reshape(next_x, (next_x.shape[0], 1, next_x.shape[1]))
                        sim_x = sim_x[:, 1:]
                        sim_x = torch.cat((sim_x, next_x), dim=1)
                        y_pred[:, sim_idx] = sim_x[:, -1]

                    self.train_discriminator(y[:,:self.history_length,:], y_pred.detach()[:,:self.history_length,:])

                    # Policy training
                    model.train()
                    self.discriminator.eval()

                    bc_dist = model.get_distribution(x)
                    y_pred = model(x)
                    bc_target_y = y[:, 0, [0]]
                    bc_loss = -bc_dist.log_prob(bc_target_y)
                    #bc_loss = self.bc_loss_fn(y_pred, bc_target_y)
                    self.optimiser_pi.zero_grad()
                    bc_loss.mean().backward()
                    self.optimiser_pi.step()


                    y_pred = torch.zeros(y.shape, device=self.device)
                    sim_x = x
                    rewards = []
                    probs = []
                    deltas = []
                    states = []
                    states_prime = []
                    actions = []
                    targets = []
                    for sim_idx in range(y.shape[1]):
                        # value estimate
                        cur_v = model.v(sim_x)

                        # action choice
                        action_distribution = model.get_distribution(sim_x)
                        if sim_idx < self.history_length:
                            states.append(sim_x)
                        action = action_distribution.sample().detach()
                        if sim_idx < self.history_length:
                            actions.append(action)
                        prob = action_distribution.log_prob(action)
                        if sim_idx < self.history_length:
                            probs.append(prob)

                        # state transition
                        sys_operations = self.sut.act_sequential(action.cpu().numpy())
                        sys_operations = torch.tensor(sys_operations).to(device=self.device).type(torch.float32)
                        next_x = torch.cat((action, sys_operations), dim=1)
                        next_x = torch.reshape(next_x, (next_x.shape[0], 1, next_x.shape[1]))
                        sim_x = sim_x[:, 1:]
                        sim_x = torch.cat((sim_x, next_x), dim=1)
                        y_pred[:, sim_idx] = sim_x[:, -1]
                        if sim_idx < self.history_length:
                            states_prime.append(sim_x)

                        # get reward
                        reward = self.get_reward(sim_x).detach()
                        if sim_idx < self.history_length:
                            rewards.append(reward)

                        target = reward + self.gamma * model.v(sim_x)
                        targets.append(target.detach())
                        delta = target - cur_v
                        deltas.append(delta.detach())
                    self.train_policy_value_net(model, states, states_prime, actions, probs, rewards)

                    if batch_idx == 0 and loader_idx == 0:
                        plt.figure(figsize=(10, 5))
                        plt.plot(y_pred[0, :, [0]].cpu().detach().numpy(), label="y_pred")
                        plt.plot(y[0, :, [0]].cpu().detach().numpy(), label="y")
                        plt.legend()
                        plt.show()
                        #plt.savefig('output/imgs/bc_gail_ppo/fig' + str(i) + '.png', dpi=300)
                loader_idx = loader_idx + 1

    def train_discriminator(self, exp_trajectory, pi_trajectory):
        trajectories = torch.cat([exp_trajectory, pi_trajectory], dim=0)
        exp_trajectory_label = torch.zeros(len(exp_trajectory))
        pi_trajectory_label = torch.ones(len(pi_trajectory))
        labels = torch.cat((exp_trajectory_label, pi_trajectory_label)).to(device=self.device).type(torch.float32)
        labels = torch.reshape(labels, (labels.shape[0], 1))

        for i in range(self.disc_iter):
            judges = self.discriminator(trajectories)
            loss = self.disc_loss(judges, labels)
            # print(f"D loss: {loss.item()}")

            self.optimiser_d.zero_grad()
            loss.backward()
            self.optimiser_d.step()

    def train_policy_value_net(self, model, states, states_prime, actions, probs, rewards):
        steps = len(states)
        batches = len(states[0])
        probs = torch.cat(probs, dim=1).detach()
        probs = torch.reshape(probs, (probs.shape[0] * probs.shape[1], 1))

        # reducing dimension for parallel calculation
        t_states = torch.stack(states)
        t_states = torch.reshape(t_states, (t_states.shape[0] * t_states.shape[1], t_states.shape[2], t_states.shape[3]))
        t_states_prime = torch.stack(states_prime)
        t_states_prime = torch.reshape(t_states_prime, (t_states_prime.shape[0] * t_states_prime.shape[1], t_states_prime.shape[2], t_states_prime.shape[3]))
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
            surr1 = ratio * advantage_list
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage_list
            loss_clip = -torch.min(surr1, surr2)
            loss_value = F.smooth_l1_loss(td_target, v)
            loss =  loss_clip + loss_value

            self.optimiser_pi.zero_grad()
            loss.mean().backward()
            self.optimiser_pi.step()

    def get_reward(self, state_action):
        reward = self.discriminator.forward(state_action)
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

    def forward(self, input):
        input = torch.reshape(input, (input.shape[0], input.shape[1] * input.shape[2]))
        return self.model(input)

