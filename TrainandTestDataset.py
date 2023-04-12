# -*- coding: utf-8 -*-
import torch.utils.data as data

class TrainandTestDataset(data.Dataset):
    def __init__(self, x, y):
        super(TrainandTestDataset, self).__init__()
        assert x.shape[0] == y.shape[0]
        self.x = x
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]