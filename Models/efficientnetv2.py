import torch
import torch.nn as nn
from math import ceil


# 激活函数保持不变
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# SE注意力模块（1D版本）
class SEBlock1D(nn.Module):
    def __init__(self, in_channels, se_ratio=0.25):
        super().__init__()
        reduced_channels = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # 改为1D池化
            nn.Conv1d(in_channels, reduced_channels, 1),
            Swish(),
            nn.Conv1d(reduced_channels, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


# 1D版本的MBConv模块
class MBConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, expand_ratio, se_ratio=None, fused=False):
        super().__init__()
        self.stride = stride
        self.use_res = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expand_ratio

        layers = []
        # Fused-MBConv结构（1D）
        if fused:
            if expand_ratio != 1:
                layers.extend([
                    nn.Conv1d(in_channels, hidden_dim, kernel_size,
                              stride=stride, padding=kernel_size // 2, bias=False),
                    nn.BatchNorm1d(hidden_dim),  # 改为1D BN
                    Swish()
                ])
            layers.extend([
                nn.Conv1d(hidden_dim, hidden_dim, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm1d(hidden_dim),
                Swish()
            ])
            if se_ratio is not None:
                layers.append(SEBlock1D(hidden_dim, se_ratio))

            layers.append(nn.Conv1d(hidden_dim, out_channels, 1, bias=False))
            layers.append(nn.BatchNorm1d(out_channels))

        # 标准MBConv结构（1D）
        else:
            if expand_ratio != 1:
                layers.extend([
                    nn.Conv1d(in_channels, hidden_dim, 1, bias=False),
                    nn.BatchNorm1d(hidden_dim),
                    Swish()
                ])

            # 深度卷积调整为1D
            layers.extend([
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size,
                          stride=stride, padding=kernel_size // 2,
                          groups=hidden_dim, bias=False),
                nn.BatchNorm1d(hidden_dim),
                Swish()
            ])

            if se_ratio is not None:
                layers.append(SEBlock1D(hidden_dim, se_ratio))

            layers.extend([
                nn.Conv1d(hidden_dim, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels)
            ])

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res:
            return x + self.block(x)
        else:
            return self.block(x)


class efficientnet(nn.Module):
    def __init__(self,
                 input_channels,
                 num_classes,
                 width_coeff=1.0,
                 depth_coeff=1.0,
                 dropout_rate=0.2,
                 ):
        super().__init__()

        # 调整后的配置（更适合序列处理）
        config = [
            # (type, kernel, stride, channels, expansion, se_ratio, layers)
            ('fused', 3, 1, 24, 1, None, 2),
            ('fused', 3, 2, 48, 4, None, 4),
            ('fused', 5, 2, 64, 4, None, 4),
            ('mbconv', 3, 2, 128, 4, 0.25, 6),
            ('mbconv', 3, 1, 160, 6, 0.25, 9),
            ('mbconv', 5, 2, 256, 6, 0.25, 15),
        ]

        scaled_config = []
        for t, k, s, c, e, se, n in config:
            scaled_config.append((
                t,
                k,
                s,
                self._scale_channels(c, width_coeff),
                e,
                se,
                self._scale_depth(n, depth_coeff)
            ))

        # 输入层（1D卷积）
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(32),
            Swish()
        )

        layers = []
        in_channels = 32
        for t, k, s, c, e, se, n in scaled_config:
            for i in range(n):
                stride = s if i == 0 else 1
                fused = True if t == 'fused' else False
                layers.append(
                    MBConv1D(in_channels, c, k, stride, e, se, fused)
                )
                in_channels = c

        self.blocks = nn.Sequential(*layers)

        # 输出层调整
        self.head = nn.Sequential(
            nn.Conv1d(in_channels, self._scale_channels(1280, width_coeff), 1),
            nn.BatchNorm1d(self._scale_channels(1280, width_coeff)),
            Swish(),
            nn.AdaptiveAvgPool1d(1),  # 1D全局平均池化
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(self._scale_channels(1280, width_coeff), num_classes)
        )

    # 保持原有缩放方法
    def _scale_depth(self, n, depth_coeff):
        return int(ceil(n * depth_coeff))

    def _scale_channels(self, c, width_coeff):
        c *= width_coeff
        new_c = max(8, int(c + 8 / 2) // 8 * 8)
        return new_c

    def forward(self, x):

        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


# 变体配置
def My_model(args):
    return efficientnet(args.input_channels, args.num_classes)