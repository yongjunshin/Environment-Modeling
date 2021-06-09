import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
from src import behavior_cloning

def test_behavior_cloning_1Tick_trainer_input_distances_input_output_shape_check():
    batch_size = 1000
    seq_length = 100
    num_features = 2
    target = torch.zeros((seq_length, num_features))
    references = torch.zeros((batch_size, seq_length, num_features))

    weights = [[pow(0.9, len(target[0]) - i)] for i in range(len(target))]
    weights = torch.tensor(weights)
    weights_sum = torch.sum(weights)

    trainer = behavior_cloning.BehaviorCloning1TickTrainer(None, None)
    distances = trainer.input_distances(target, references, 'ed', weights, weights_sum)
    assert distances.shape[0] == batch_size

    distances = trainer.input_distances(target, references, 'wed', weights, weights_sum)
    assert distances.shape[0] == batch_size

    distances = trainer.input_distances(target, references, 'md', weights, weights_sum)
    assert distances.shape[0] == batch_size

    distances = trainer.input_distances(target, references, 'wmd', weights, weights_sum)
    assert distances.shape[0] == batch_size

    distances = trainer.input_distances(target, references, 'dtw', weights, weights_sum)
    assert distances.shape[0] == batch_size

