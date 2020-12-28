import torch
import torch.nn as nn

class LinearUnit(nn.Module):
    def __init__(self):
        super(LinearUnit, self).__init__()
        self.linear = nn.Linear(4, 2, bias=False)
        self.activation = nn.Tanh()

    def forward(self, z, x):
        inter = torch.cat((z, x), axis=0)
        inter = self.linear(inter)
        return self.activation(inter)
