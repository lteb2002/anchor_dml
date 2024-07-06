import torch
from torch import nn, optim
from torch.nn import functional as F
import dl_model.rere_config as cnf
import ext.mish as mish
import dl_model.svdd.kf as kf
import numpy as np
from dl_model.rere_dml import Triplet as Trip


# _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AnchorMlgDml(torch.nn.Module):
    # the feature number of the data points fed into the net
    _d_in = 0
    # The sampled data points used in Nystrom
    _samples = None
    # The number of classes
    _d_out = 0
    _nystrom_matrix = None

    def __init__(self, d_in, mlg, sample_size, d_out):
        super(AnchorMlgDml, self).__init__()
        self._d_in = d_in
        # 将从外部获得的samples转换为torch的相应格式
        self.mlg = mlg  # 极大线性无关组
        self.mlg_size = len(mlg)  # calculate the number of samples
        self.sample_size = sample_size
        print("anchor points size:",sample_size)
        self._d_out = d_out  #
        self.dml_module = nn.Sequential(
            nn.Linear(d_in, d_in, bias=True),
            mish.Mish(),
            nn.Linear(d_in, d_in, bias=True),
            mish.Mish()
        )
        self.anchor_representation = nn.Sequential(
            nn.Linear(self.mlg_size, self.sample_size, bias=False),
            nn.Tanh()
        )
        self.classifier = nn.Sequential(
            nn.Linear(sample_size, d_out)
        )

    def encode_dml(self, x):
        xs = self.dml_module(x)
        return xs

    # 计算数据的DML损失，适合mini-batch或者batch数据
    def compute_dml_loss(self, x, label):
        xs = self.encode_dml(x)
        loss = Trip.compute_dml_loss(xs, label)
        return loss

    def compute_whole_anchor(self, xx, sam_new,dis_fun='eud'):
        width = sam_new.shape[1]
        if dis_fun == 'cos':
            a = xx.unsqueeze(1)
            b = sam_new.unsqueeze(0)
            ab = a * b
            ab_dis = torch.norm(a, dim=1)*torch.norm(a, dim=1)
            re = ab/ab_dis
        else:
            dis = (xx.unsqueeze(1) - sam_new.unsqueeze(0)).reshape(-1, width)
            re = torch.norm(dis, dim=1)
        # print(norms)
        # re = torch.exp(-re)
        re = re.view(xx.shape[0], sam_new.shape[0])
        # print(re)
        return re

    # 进行数据的NYSTROM步变换
    def cross_over(self, x):
        anchors = self.anchor_representation(self.mlg.t()).t()
        anchors = self.encode_dml(anchors)
        # print(anchors.shape)
        z = self.compute_whole_anchor(x, anchors)
        return z

    def percept(self, anchor):
        z = self.classifier(anchor)
        z = F.log_softmax(z, 1)
        return z

    def forward(self, x):
        # print(x.dtype)
        x_dml = self.encode_dml(x)
        x_nys = self.cross_over(x_dml)  #
        z = self.percept(x_nys)
        # print(z)
        return z

    def predict(self, x):
        x_dml = self.encode_dml(x)
        x_nys = self.encode_nys(x_dml)  #
        z = self.percept(x_nys)
        return torch.argmax(z)

    def transform(self, x):
        x_dml = self.encode_dml(x)
        return x_dml

    def export_anchors(self):
        anchors = self.anchor_representation(self.mlg.t()).t()
        anchors = self.encode_dml(anchors)
        return anchors

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
