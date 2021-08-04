from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchviz import make_dot


class GailActorCriticTrainer:
    def __init__(self, device, sut, state_dim, action_dim, history_length):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.history_length = history_length

        self.device = device
        self.sut = sut
        self.discriminator = Discriminator((state_dim + action_dim) * history_length).to(device=self.device)

        self.optimiser_d = torch.optim.Adam(self.discriminator.parameters(), lr=0.001)

        self.disc_iter = 4
        self.disc_loss = nn.MSELoss()

        self.gamma = 0.99

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
                        action_prob = model.get_distribution(sim_x)
                        action = action_prob.sample().detach()

                        # state transition
                        sys_operations = self.sut.act_sequential(action.cpu().numpy())
                        sys_operations = torch.tensor(sys_operations).to(device=self.device).type(torch.float32)
                        next_x = torch.cat((action, sys_operations), dim=1)
                        next_x = torch.reshape(next_x, (next_x.shape[0], 1, next_x.shape[1]))
                        sim_x = sim_x[:, 1:]
                        sim_x = torch.cat((sim_x, next_x), dim=1)
                        y_pred[:, sim_idx] = sim_x[:, -1]

                    self.train_discriminator(y[:,:self.history_length,:], y_pred.detach()[:,:self.history_length,:])
                    print("expert judge: ", self.discriminator(y[:, :self.history_length, :]).mean(), "model judge: ",
                          self.discriminator(y_pred.detach()[:, :self.history_length, :]).mean())
                    print("expert reward: ", self.get_reward(y[:, :self.history_length, :]).mean(), "model reward: ",
                          self.get_reward(y_pred.detach()[:, :self.history_length, :]).mean())


                    # Policy training
                    model.train()
                    self.discriminator.eval()

                    y_pred = torch.zeros(y.shape, device=self.device)
                    sim_x = x
                    rewards = []
                    probs = []
                    losses = []
                    for sim_idx in range(y.shape[1]):
                        # value estimate
                        cur_v = model.v(sim_x)

                        # action choice
                        action_prob = model.get_distribution(sim_x)
                        action = action_prob.sample().detach()
                        prob = action_prob.log_prob(action)
                        probs.append(prob)

                        # state transition
                        sys_operations = self.sut.act_sequential(action.cpu().numpy())
                        sys_operations = torch.tensor(sys_operations).to(device=self.device).type(torch.float32)
                        next_x = torch.cat((action, sys_operations), dim=1)
                        next_x = torch.reshape(next_x, (next_x.shape[0], 1, next_x.shape[1]))
                        sim_x = sim_x[:, 1:]
                        sim_x = torch.cat((sim_x, next_x), dim=1)
                        y_pred[:, sim_idx] = sim_x[:, -1]

                        # get reward
                        if sim_idx < self.history_length:
                            reward = self.get_reward(sim_x).detach()
                            rewards.append(reward)

                            delta = reward + self.gamma * model.v(sim_x) - cur_v
                            loss = -prob * delta.detach() + torch.abs(delta)
                            losses.append(loss)
                        if sim_idx == 0 or sim_idx == self.history_length-1:
                            print(reward.mean())
                    self.train_policy_value_net(model, losses)

                    if batch_idx == 0 and loader_idx == 0:
                        plt.figure(figsize=(10, 5))
                        plt.plot(y_pred[0, :, [0]].cpu().detach().numpy(), label="y_pred")
                        plt.plot(y[0, :, [0]].cpu().detach().numpy(), label="y")
                        plt.legend()
                        plt.show()
                        #plt.savefig('output/imgs/gail_actor_critic/fig' + str(i) + '.png', dpi=300)
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

    def train_policy_value_net(self, model, losses):
        self.optimiser_pi.zero_grad()
        for loss in losses:
            loss.mean().backward()
            #make_dot(loss.mean(), params=dict(model.named_parameters())).render(
             #   "graph", format="png")
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

