import torch


def episode_to_datapoints(x: torch.tensor, y: torch.tensor):
    state_length = x.shape[1]
    episode = torch.cat((x, y), dim=1)
    episode_length = episode.shape[1]

    x_datapoints = []
    y_datapoints = []
    for i in range(episode_length - state_length):
        x_datapoints.append(episode[:, i:i+state_length])
        y_datapoints.append(episode[:, [i+state_length]])

    x_datapoints = torch.cat(x_datapoints)
    y_datapoints = torch.cat(y_datapoints)

    return x_datapoints, y_datapoints