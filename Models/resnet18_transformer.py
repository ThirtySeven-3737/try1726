import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import math

# 获取当前运行脚本的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))

# 将当前工作目录设置为脚本所在的目录
os.chdir(script_dir)

# 定义一维残差块（Residual Block）
class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # 如果输入和输出通道不同，或者stride不是1，则使用1x1卷积调整通道
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 添加shortcut连接
        out += self.shortcut(x)
        out = self.relu(out)

        return out
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10240):
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

# 定义一维ResNet-18
class resnet18(nn.Module):
    def __init__(self, input_channels, num_classes, max_len, device):
        super(resnet18, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.device = device
        self.conv1 = nn.Conv1d(self.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # 定义四个stage，每个stage包含多个BasicBlock
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        # 全局平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, self.num_classes)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=2, dim_feedforward=64 * 4, dropout=0.1),
            num_layers=1
        )
        self.pe = PositionalEncoding(d_model=64, max_len=max_len)


    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(BasicBlock1D(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        print(x.shape)
        #9943
        pad_nums = (x.size(2) // 512 + 1) * 512 - x.size(2)
        chunks_num = x.size(2) // 512 + 1  ###段数
        padded_x = F.pad(x, (0, pad_nums), mode='constant', value=0)
        padded_x = self.pe(padded_x)
        data_reshaped = padded_x.view(padded_x.size(0), padded_x.size(1), 512, chunks_num)

        # 段内处理 + 位置编码
        outputs = []
        for i in range(chunks_num):
            # 获取当前段 [8, 256, 512]
            segment = data_reshaped[:, :, :, i]

            # 调整维度供Transformer使用 [512, 8, 256]
            segment = segment.permute(2, 0, 1)

            # # 添加段位置编码（关键修正点）
            # segment_pos = self.segment_pos_embedding(
            #     torch.tensor(i).to(x.device)
            # ).view(1, 1, -1)  # [1, 1, 256]
            # segment = segment + segment_pos  # 广播到 [512, 8, 256]

            # 段内Transformer处理
            output = self.transformer(segment)  # [512, 8, 256]
            outputs.append(output)

        # 拼接所有段 [78*512, 8, 256]
        final_output = torch.cat(outputs, dim=0)
        x = final_output.permute(1, 2, 0)




        x = self.layer1(x)
        #9943
        x = self.layer2(x)
        #4972
        x = self.layer3(x)
        #2486
        x = self.layer4(x)
        #1243

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    ##################################################################################################
def My_model(args):
    return  resnet18(args.input_channels, args.num_classes, args.max_len, args.device)
