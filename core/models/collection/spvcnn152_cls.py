import torch
import torch.nn as nn

import torchsparse.nn as spnn
from torchsparse.tensor import *

from core.models.utils import *


__all__ = ['SPVCNN152_cls']


def global_avg_pool(inputs):
    batch_index = inputs.C[:, -1]
    max_index = torch.max(batch_index).item()
    outputs = []
    for i in range(max_index + 1):
        cur_inputs = torch.index_select(inputs.F, 0,
                                        torch.where(batch_index == i)[0])
        cur_outputs = cur_inputs.mean(0).unsqueeze(0)
        outputs.append(cur_outputs)
    outputs = torch.cat(outputs, 0)
    return outputs


def global_max_pool(inputs):
    batch_index = inputs.C[:, -1]
    max_index = torch.max(batch_index).item()
    outputs = []
    for i in range(max_index + 1):
        cur_inputs = torch.index_select(inputs.F, 0,
                                        torch.where(batch_index == i)[0])
        cur_outputs = cur_inputs.max(0)[0].unsqueeze(0)
        outputs.append(cur_outputs)
    outputs = torch.cat(outputs, 0)
    return outputs


def make_layers(num, inc, outc, midc, ks, stride, dilation):
    blocks = []
    for _ in range(num):
        blocks.append(ResidualBlock(inc, outc, midc, ks=ks, stride=stride, dilation=dilation))
    return blocks


class ResidualBlock(nn.Module):
    def __init__(self, inc, outc, midc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                                 midc,
                                 kernel_size=1,
                                 dilation=dilation,
                                 stride=1), spnn.BatchNorm(midc),
            spnn.ReLU(True),
            spnn.Conv3d(midc,
                                 midc,
                                 kernel_size=ks,
                                 dilation=dilation,
                                 stride=stride), spnn.BatchNorm(midc),
            spnn.ReLU(True),
            spnn.Conv3d(midc,
                                 outc,
                                 kernel_size=1,
                                 dilation=dilation,
                                 stride=1), spnn.BatchNorm(outc))

        self.downsample = nn.Sequential() if (inc == outc and stride == 1) else \
            nn.Sequential(
                spnn.Conv3d(inc, outc, kernel_size=2, dilation=1, stride=stride),
                spnn.BatchNorm(outc)
            )

        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


class SPVCNN152_cls(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        cr = kwargs.get('cr', 1.0)
        cs = [64, 256, 512, 1024, 2048]
        midcs = [64, 128, 256, 512]
        cs = [int(cr * x) for x in cs]

        if 'pres' in kwargs and 'vres' in kwargs:
            self.pres = kwargs['pres']
            self.vres = kwargs['vres']

        self.stem = nn.Sequential(
            spnn.Conv3d(kwargs['input_dim'], cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0])
        )


        self.stage1 = nn.Sequential(
            ResidualBlock(cs[0], cs[1], midcs[0], ks=3, stride=1, dilation=1),
            *make_layers(2, cs[1], cs[1], midcs[0], ks=3, stride=1, dilation=1)
        )

        self.stage2 = nn.Sequential(
            ResidualBlock(cs[1], cs[2], midcs[1], ks=3, stride=2, dilation=1),
            *make_layers(7, cs[2], cs[2], midcs[1], ks=3, stride=1, dilation=1)
        )

        self.stage3 = nn.Sequential(
            ResidualBlock(cs[2], cs[3], midcs[2], ks=3, stride=2, dilation=1),
            *make_layers(35, cs[3], cs[3], midcs[2], ks=3, stride=1, dilation=1)
        )

        self.stage4 = nn.Sequential(
            ResidualBlock(cs[3], cs[4], midcs[3], ks=3, stride=2, dilation=1),
            *make_layers(2, cs[4], cs[4], midcs[3], ks=3, stride=1, dilation=1)
        )

        self.classifier = nn.Sequential(
        	                            nn.Linear(cs[4], kwargs['num_classes']),
                                        nn.BatchNorm1d(kwargs['num_classes']))

        self.weight_initialization()
        self.dropout = nn.Dropout(0.3, True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: SparseTensor z: PointTensor
        z = PointTensor(x.F, x.C.float())
        x0 = initial_voxelize(z, self.pres, self.vres)
        x0 = self.stem(x0)
        z0 = voxel_to_point(x0, z, nearest=False)
        z0.F = z0.F

        x1 = point_to_voxel(x0, z0)
        x1 = self.stage1(x1)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = global_avg_pool(x4)
        out = self.classifier(x5)

        return out
