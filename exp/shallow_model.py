import torch
import torch.nn as nn
import numpy as np
import random
from torch.distributions import Normal
import torch.nn.functional as F


class LineTracerShallowPREnvironmentModel(nn.Module):
    def __init__(self, regressor, poly_feature, device):
        super(LineTracerShallowPREnvironmentModel, self).__init__()
        self.regressor = regressor
        self.poly_feature = poly_feature
        self.device = device

    def forward(self, x):
        """
        Deterministic (mean) action
        :param x: state
        :return: action
        """
        flatted_x = torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2])).cpu()
        np_poly_x = self.poly_feature.transform(flatted_x)

        predicted_color = self.regressor.predict(np_poly_x)
        predicted_color = torch.tensor(predicted_color, device=self.device)
        predicted_color = torch.reshape(predicted_color, (predicted_color.shape[0], 1))
        predicted_color = torch.nan_to_num(predicted_color)
        predicted_color = torch.clamp(predicted_color, -1, 1)

        return predicted_color


class LineTracerShallowRFEnvironmentModel(nn.Module):
    def __init__(self, regressor, device):
        super(LineTracerShallowRFEnvironmentModel, self).__init__()
        self.regressor = regressor
        self.device = device

    def forward(self, x):
        """
        Deterministic (mean) action
        :param x: state
        :return: action
        """
        flatted_x = torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2])).cpu()

        predicted_color = self.regressor.predict(flatted_x)
        predicted_color = torch.tensor(predicted_color, device=self.device)
        predicted_color = torch.reshape(predicted_color, (predicted_color.shape[0], 1))
        predicted_color = torch.nan_to_num(predicted_color)
        predicted_color = torch.clamp(predicted_color, -1, 1)

        return predicted_color
