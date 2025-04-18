import torch
from torch import nn
import math
import numpy as np
import torch.nn.functional as F



class BaseBlock(torch.nn.Module):
    def __init__(self, in_planes):
        super(BaseBlock, self).__init__()

        self.bottleneck = torch.nn.Conv1d(in_planes, in_planes // 4, kernel_size=1, stride=1, bias=False)
        self.conv4 = torch.nn.Conv1d(in_planes // 4, in_planes // 4, kernel_size=7, stride=1, padding=3, bias=False)
        self.conv3 = torch.nn.Conv1d(in_planes // 4, in_planes // 4, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv2 = torch.nn.Conv1d(in_planes // 4, in_planes // 4, kernel_size=3, stride=1, padding=1, bias=False)

        self.maxpool = torch.nn.MaxPool1d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
        self.conv1 = torch.nn.Conv1d(in_planes, in_planes // 4, kernel_size=1, stride=1, bias=False)

        self.bn = torch.nn.BatchNorm1d(in_planes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.bottleneck(x)
        output4 = self.conv4(output)
        output3 = self.conv3(output)
        output2 = self.conv2(output)

        output1 = self.maxpool(x)
        output1 = self.conv1(output1)

        x_out = self.relu(self.bn(torch.cat((output1, output2, output3, output4), dim=1)))
        return x_out

class Downsample(torch.nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, bias=False),
                                  nn.BatchNorm1d(out_planes),
                                  nn.ReLU(inplace=True))

    def forward(self, x):
        output = self.conv(x)

        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5120):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.T  # 转换为 (d_model, max_len)
        self.register_buffer('pe', pe.unsqueeze(0))  # 形状 (1, d_model, max_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :, :x.size(2)]
        return x


class stem_inceptiontime(torch.nn.Module):
    def __init__(self, input_channels = 1, num_classes = 4, channels=[64, 96, 128]):
        super(stem_inceptiontime, self).__init__()

        self.stem = nn.Sequential(nn.Conv1d(input_channels, channels[0], kernel_size=7, stride=2, padding=3, bias=False),
                                  nn.BatchNorm1d(channels[0]),
                                  nn.ReLU(inplace=True),
                                  nn.MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False))


        self.block1 = nn.ModuleList([])
        for i in range(3):
            self.block1.append(BaseBlock(channels[0]))

        self.block2 = nn.ModuleList([])
        for i in range(3):
            self.block2.append(BaseBlock(channels[1]))

        self.downsample = nn.ModuleList([])
        for i in range(len(channels) - 1):
            self.downsample.append(Downsample(channels[i], channels[i + 1]))

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[-1], num_classes)

        self.transformer1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=2, dim_feedforward=64 * 4, dropout=0.1),
            num_layers=1
        )
        self.transformer2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=96, nhead=3, dim_feedforward=96 * 4, dropout=0.1),
            num_layers=1
        )
        self.pe1 = PositionalEncoding(d_model=64)
        self.pe2 = PositionalEncoding(d_model=96)
        self.segment_pos_embedding = nn.Embedding(num_embeddings = 20, embedding_dim=64)
        self.inter_segment_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=64,
                nhead=4,
                dim_feedforward=64*4
            ),
            num_layers=1
        )



    def forward(self, x):

        x = self.stem(x)


        output1 = self.block1[0](x)
        output1 = self.block1[1](output1)
        output1 = self.block1[2](output1)
        output1 = x + output1
        output1 = self.downsample[0](output1)
        ##5120 4972

        output2 = self.block2[0](output1)
        output2 = self.block2[1](output2)
        output2 = self.block2[2](output2)
        output2 = output1 + output2
        output2 = self.downsample[1](output2)


        output2 = self.avgpool(output2)
        output2 = output2.view(output2.size(0), -1)
        output2= self.fc(output2)

        return output2

def My_model(args):
    return stem_inceptiontime(args.input_channels, args.num_classes)