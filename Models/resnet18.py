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


# 定义一维ResNet-18
class ResNet18(nn.Module):
    def __init__(self, input_channels, num_classes, device):
        super(ResNet18, self).__init__()
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

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def My_model(args):
    return  ResNet18(args.input_channels, args.num_classes, args.device)