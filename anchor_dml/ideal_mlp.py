import torch
from torch import nn, optim
from torch.nn import functional as F
import ext.mish as mish


class IdealMLP(torch.nn.Module):
    _d_in = 0

    def __init__(self, d_in, h1, d_out):
        self._d_in = d_in
        super(IdealMLP, self).__init__()
        self.trans = nn.Sequential(
            nn.Linear(d_in, d_in),
            mish.Mish(),
            nn.Linear(d_in, d_in),
            mish.Mish(),
            nn.Linear(d_in, h1)
        )
        self.perceptron = nn.Linear(h1, d_out)

    def forward(self, x):
        z = self.trans(x)
        z = self.perceptron(z)  #
        z = torch.nn.functional.log_softmax(z)
        return z

    def transform(self, x):
        y = self.trans(x)
        return y

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, X, label, z):
        # print(label)
        # print(z)
        loss = torch.nn.functional.cross_entropy(z, label)
        return loss
