import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np

class BPModel(nn.Module):
    def __init__(self):
        super(BPModel, self).__init__()
        # set customize layer
        # set hidden layer
        self.layer1 = nn.Linear(
            in_features=13,
            out_features=30,
            bias=True
        )
        # set activation
        self.activate1 = nn.ReLU()
        self.output = nn.Linear(
            in_features=30,
            out_features=1,
            bias=True
        )
        self.activate2 = nn.ReLU()



    def forward(self, x):
        x = self.layer1(x)
        x = self.activate1(x)
        x = self.output(x)
        x = self.activate2(x)
        return x
