import torch
import torch.nn as nn
from torch.distributions import Normal


class LineTracerEnvironmentModelGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, device):
        super(LineTracerEnvironmentModelGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.device = device

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device=self.device)
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :])
        return out

    def get_distribution(self, x):
        mu = self.forward(x)
        sigma = torch.ones_like(mu) * 0.01
        dist = Normal(mu, sigma)
        return dist

    def act(self, x):
        dist = self.get_distribution(x)
        action = dist.sample()
        action = action.detach()
        return action

