import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from sklearn.preprocessing import MinMaxScaler
import numpy as np
from src import line_tracer


def test_line_tracer_ver1_act_sequential_input_output_shape_check():
    mm = MinMaxScaler(feature_range=(-1, 1))
    temp = np.array([[-1, -1], [1, 1]])
    mm.fit_transform(temp)

    lt = line_tracer.LineTracerVer1(mm)
    input_colors = np.zeros(100)
    input_colors = np.reshape(input_colors, (input_colors.shape[0], 1))
    assert input_colors.shape == lt.act_sequential(input_colors).shape
