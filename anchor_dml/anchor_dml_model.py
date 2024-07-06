import torch
from torch import nn, optim
from torch.nn import functional as F
import dl_model.rere_config as cnf
import ext.mish as mish
import dl_model.svdd.kf as kf
import numpy as np
from dl_model.rere_dml import Triplet as Trip


# _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AnchorDML(torch.nn.Module):
    # the feature number of the data points fed into the net
    _d_in = 0
    # The sampled data points used in Nystrom
    _samples = None
    # The number of classes
    _d_out = 0

    def __init__(self, d_in, samples, d_out):
        super(AnchorDML, self).__init__()
        self._d_in = d_in
        # 将从外部获得的samples转换为torch的相应格式
        self._samples = samples
        sample_size = len(samples)  # calculate the number of samples
        self._sample_size = sample_size
        self._d_out = d_out  #
        self.dml_module = nn.Sequential(
            nn.Linear(d_in, d_in, bias=True),
            mish.Mish(),
            nn.Linear(d_in, d_in, bias=True),
            mish.Mish()
        )
        self.perceptron = nn.Sequential(
            nn.Linear(sample_size, d_out),
            nn.Tanh()
        )

    def encode_dml(self, x):
        xs = self.dml_module(x)
        return xs

    # 计算数据的DML损失，适合mini-batch或者batch数据
    def compute_dml_loss(self, x, label):
        xs = self.encode_dml(x)
        loss = Trip.compute_dml_loss(xs, label)
        return loss

    def compute_whole_anchor(self, xx, sam_new):
        width = sam_new.shape[1]
        dis = (xx.unsqueeze(1) - sam_new.unsqueeze(0)).view(-1, width)
        re = torch.norm(dis, dim=1)
        # re = torch.exp(-re)
        re = re.view(xx.shape[0], sam_new.shape[0])
        return re

    # 进行数据的NYSTROM步变换
    def cross_over(self, x):
        sam_new = self.encode_dml(self._samples)
        z = self.compute_whole_anchor(x, sam_new)
        return z

    def percept(self, anchor):
        z = self.perceptron(anchor)
        z = F.log_softmax(z, 1)
        return z

    def forward(self, x):
        # print(x.dtype)
        x_dml = self.encode_dml(x)
        cross = self.cross_over(x_dml)  #
        z = self.percept(cross)
        # print(z)
        return z

    def transform(self, x):
        x_dml = self.encode_dml(x)
        return x_dml

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, x, label, output):
        label = label.long()
        # print(label)
        # print(output)
        loss1 = F.cross_entropy(output, label)
        loss2 = 0
        # if self.if_dml_reg:
        #     loss2 = self.compute_dml_loss(x, label) / x.shape[0]
        #     print("Classification loss:", loss1, ",DML loss", loss2)
        # loss = F.nll_loss(output, label)
        loss = loss1 + loss2
        return loss
