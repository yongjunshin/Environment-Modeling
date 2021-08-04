import torch
import torch.nn as nn
import numpy as np
import random
from torch.distributions import Normal
import torch.nn.functional as F


class LineTracerRandomEnvironmentModelDNN(nn.Module):
    def __init__(self, device):
        super(LineTracerRandomEnvironmentModelDNN, self).__init__()
        self.device = device

    def forward(self, x):
        """
        Deterministic (mean) action
        :param x: state
        :return: action
        """

        out = (torch.rand(len(x), 1, device=self.device) - 0.5) * 2
        return out
