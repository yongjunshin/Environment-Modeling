from torch.utils.data import Dataset


class FieldTestDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        super(FieldTestDataset, self).__init__()
        self.x = x_tensor
        self.y = y_tensor

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]