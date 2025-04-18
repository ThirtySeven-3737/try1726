import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    """1D卷积+BN+ReLU6基础模块"""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    """1D倒置残差块（核心模块）"""

    def __init__(self, in_ch, out_ch, stride, expand_ratio):
        super().__init__()
        hidden_ch = int(round(in_ch * expand_ratio))
        self.use_res_connect = stride == 1 and in_ch == out_ch

        layers = []
        if expand_ratio != 1:
            # 扩展阶段 (1x1卷积)
            layers.append(ConvBNReLU(in_ch, hidden_ch, kernel_size=1))

        # 深度卷积
        layers.extend([
            ConvBNReLU(hidden_ch, hidden_ch, stride=stride, groups=hidden_ch),
            # 投影阶段 (1x1线性卷积)
            nn.Conv1d(hidden_ch, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class mobilenetv2(nn.Module):
    """1D版MobileNetV2 for序列处理"""

    def __init__(self,
                 input_channels,
                 num_classes,
                 width_mult=1.0,
                 inverted_residual_setting=None):
        super().__init__()

        # 基础配置 (扩展比, 输出通道, 重复次数, 步长)
        default_setting = [
            # t, c, n, s (与原论文配置对应)
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        inverted_residual_setting = inverted_residual_setting or default_setting

        # 首层卷积
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280

        features = [ConvBNReLU(input_channels, input_channel, stride=2)]

        # 构建倒置残差块
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, stride, t))
                input_channel = output_channel

        # 末尾层
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))

        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)  # 特征提取
        x = x.mean([2])  # 全局平均池化 [B, C]
        return self.classifier(x)

def My_model(args):
    return mobilenetv2(args.input_channels, args.num_classes)
