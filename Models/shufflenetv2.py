import torch
import torch.nn as nn


def channel_shuffle(x, groups):
    batchsize, num_channels, length = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, length)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, length)
    return x


class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ShuffleUnit, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 通道分配逻辑修正
        if stride > 1:
            branch_channels = out_channels // 2
            self.branch1 = nn.Sequential(
                nn.Conv1d(in_channels, in_channels, kernel_size=3,
                          stride=stride, padding=1, groups=in_channels, bias=False),
                nn.BatchNorm1d(in_channels),
                nn.Conv1d(in_channels, branch_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(branch_channels),
                nn.ReLU(inplace=True)
            )
            self.branch2 = nn.Sequential(
                nn.Conv1d(in_channels, branch_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(branch_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(branch_channels, branch_channels, kernel_size=3,
                          stride=stride, padding=1, groups=branch_channels, bias=False),
                nn.BatchNorm1d(branch_channels),
                nn.Conv1d(branch_channels, branch_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(branch_channels),
                nn.ReLU(inplace=True)
            )
        else:
            mid_channels = out_channels // 2
            branch_channels = mid_channels // 1  # 通道分割修正
            assert in_channels == out_channels, "输入输出通道必须相等当stride=1"

            self.branch1 = nn.Sequential()
            self.branch2 = nn.Sequential(
                nn.Conv1d(mid_channels, mid_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(mid_channels, mid_channels, kernel_size=3,
                          stride=1, padding=1, groups=mid_channels, bias=False),
                nn.BatchNorm1d(mid_channels),
                nn.Conv1d(mid_channels, branch_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(branch_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        if self.stride > 1:
            x1 = self.branch1(x)
            x2 = self.branch2(x)
            out = torch.cat((x1, x2), dim=1)
        else:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        return channel_shuffle(out, 2)


class ShuffleNetV2(nn.Module):
    def __init__(self, input_channels=2, num_classes=4, scale_factor=1.0):  # 修改默认输入通道为2
        super(ShuffleNetV2, self).__init__()
        self.scale_factor = scale_factor
        self.stage_repeats = [4, 8, 4]
        self.stage_out_channels = self._get_stage_channels()

        # 初始层调整
        self.conv1 = nn.Conv1d(input_channels, 24, kernel_size=3,
                               stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(24)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # 分阶段构建
        self.stage2 = self._make_stage(24, self.stage_out_channels[0], self.stage_repeats[0], 2)
        self.stage3 = self._make_stage(self.stage_out_channels[0], self.stage_out_channels[1],
                                       self.stage_repeats[1], 2)
        self.stage4 = self._make_stage(self.stage_out_channels[1], self.stage_out_channels[2],
                                       self.stage_repeats[2], 2)

        # 输出层
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.stage_out_channels[-1], num_classes)
        self.relu = nn.ReLU(inplace=True)

    def _get_stage_channels(self):
        # 通道数配置表修正
        if self.scale_factor == 0.5:
            return [48, 96, 192]
        elif self.scale_factor == 1.0:
            return [116, 232, 464]
        elif self.scale_factor == 1.5:
            return [176, 352, 704]
        elif self.scale_factor == 2.0:
            return [244, 488, 976]
        else:
            raise ValueError("Invalid scale factor. Supported: 0.5, 1.0, 1.5, 2.0")

    def _make_stage(self, in_channels, out_channels, repeats, stride):
        layers = [ShuffleUnit(in_channels, out_channels, stride)]
        for _ in range(repeats - 1):
            layers.append(ShuffleUnit(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 前向传播跟踪
        x = self.relu(self.bn1(self.conv1(x)))  # [32, 24, 19886]
        x = self.maxpool(x)  # [32, 24, 9943]

        x = self.stage2(x)  # [32, 116, 4972]
        x = self.stage3(x)  # [32, 232, 2486]
        x = self.stage4(x)  # [32, 464, 1243]

        x = self.avgpool(x)  # [32, 464, 1]
        x = torch.flatten(x, 1)  # [32, 464]
        x = self.fc(x)  # [32, num_classes]
        return x


def My_model(args):
    return ShuffleNetV2(input_channels=args.input_channels,
                        num_classes=args.num_classes,
                        scale_factor=1.0)