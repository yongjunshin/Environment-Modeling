import torch
import torch.nn as nn
import numpy as np
import random
from torch.distributions import Normal
import torch.nn.functional as F


class LineTracerManualEnvironmentModelDNN(nn.Module):
    def __init__(self, latency, unit_diff, scaler, device):
        super(LineTracerManualEnvironmentModelDNN, self).__init__()
        self.device = device
        self.latency = latency
        self.unit_diff = unit_diff
        self.scaler = scaler
        self.normalized_unit_diff = torch.tensor((scaler.transform(np.array([[2 * unit_diff, 0]])) - scaler.transform(np.array([[unit_diff, 0]])))[:, 0], device=self.device)

    def forward(self, x):
        """
        Deterministic (mean) action
        :param x: state
        :return: action
        """
        cur_color = x[:, -1, 0]
        ref_angle = x[:, x.shape[1] - self.latency, 1]

        true_mask_value = torch.tensor([1.], device=self.device)
        false_mask_value = torch.tensor([0.], device=self.device)

        positive_mask = torch.where(ref_angle > 0, true_mask_value, false_mask_value)
        zero_mask = torch.where(ref_angle == 0, true_mask_value, false_mask_value)
        negative_mask = torch.where(ref_angle < 0, true_mask_value, false_mask_value)
        new_color = positive_mask * (cur_color - self.normalized_unit_diff) + zero_mask * cur_color + negative_mask * (cur_color + self.normalized_unit_diff)
        new_color = torch.clamp(new_color, -1, 1)
        new_color = torch.reshape(new_color, (new_color.shape[0], 1))


        return new_color
