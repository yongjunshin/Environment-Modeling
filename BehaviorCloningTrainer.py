from tqdm import tqdm
import torch
import numpy as np

class BehaviorCloningTrainer():
    def __init__(self):
        None

    def train(self, model, epochs, train_dataloaders, test_dataloaders, dagger=False, max_dagger=50):
        loss_fn = torch.nn.MSELoss()
        optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
        training_loss = np.zeros(epochs)

        for i in tqdm(range(epochs)):
            for dataloader in train_dataloaders:
                for x, y in iter(dataloader):
                    model.train()
                    y_pred = model(x)
                    loss = loss_fn(y, y_pred)
                    training_loss[i] = loss.item()

                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()

        return training_loss
