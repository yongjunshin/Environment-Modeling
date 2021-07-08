import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F


# class LineTracerEnvironmentModelGRU(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_layers, output_dim, device):
#         super(LineTracerEnvironmentModelGRU, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#
#         self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, output_dim)
#
#         self.device = device
#
#     def forward(self, x):
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device=self.device)
#         out, (hn) = self.gru(x, (h0.detach()))
#         out = self.fc(out[:, -1, :])
#         return out
#
#     def get_distribution(self, x):
#         mu = self.forward(x)
#         sigma = torch.ones_like(mu) * 0.01
#         dist = Normal(mu, sigma)
#         return dist
#
#     def act(self, x):
#         dist = self.get_distribution(x)
#         action = dist.sample()
#         action = action.detach()
#         return action


class LineTracerEnvironmentModelDNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, device):
        super(LineTracerEnvironmentModelDNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.fc_std = nn.Linear(hidden_dim, output_dim)
        self.fc_v = nn.Linear(hidden_dim, 1)

        self.device = device

    def forward(self, x):
        """
        Deterministic (mean) action
        :param x: state
        :return: action
        """
        x = torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out

    def get_distribution(self, x):
        """
        Distribution of non-deterministic Normal(mean, std) action
        :param x: state
        :return: distribution
        """
        x = torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))
        out = F.relu(self.fc1(x))
        mu = self.fc2(out)

        sigma = torch.sigmoid(self.fc_std(out)) * 0.1

        dist = Normal(mu, sigma)
        return dist

    def act(self, x):
        """
        sampled non-deterministic action
        :param x: state
        :return: action
        """
        dist = self.get_distribution(x)
        action = dist.sample()
        action = action.detach()
        return action

    def v(self, x):
        """
        value (sum discounted reward) estimation function
        :param x: state
        :return: value
        """
        x = torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))
        out = F.relu(self.fc1(x))
        v = self.fc_v(out)
        return v

