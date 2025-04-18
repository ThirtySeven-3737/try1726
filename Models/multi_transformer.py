import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# class MergingBlock(nn.Module):
#     def __init__(self, in_channel, out_channel, stride):
#         super(MergingBlock, self).__init__()
#



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



class multi_transformer(nn.Module):
    def __init__(self, input_channels, num_classes, max_len):
        super(multi_transformer, self).__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.conv1 = nn.Conv1d(self.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.merging1 = nn.Sequential(
            # 第一次降采样
            nn.Conv1d(in_channels=self.input_channels, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            # 第二次降采样
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        self.merging2 = nn.Sequential(
            # 第一次降采样
            nn.Conv1d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),

            # 第二次降采样
            nn.Conv1d(96, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )

        self.pe = PositionalEncoding(d_model=64, max_len=max_len)
        self.local_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=2, dim_feedforward=64 * 4, dropout=0.1),
            num_layers=1
        )
        self.global_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=2, dim_feedforward=64 * 4, dropout=0.1),
            num_layers=1
        )

        # 全局平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, self.num_classes)

    def _preprocess_segments(self, x):
        """数据预处理和分块"""
        # 填充到能被512整除的长度
        pad_nums = (x.size(2) // 512 + 1) * 512 - x.size(2)
        padded_x = F.pad(x, (0, pad_nums), mode='constant', value=0)
        shortcut = padded_x
        padded_x = self.pe(padded_x)

        # 分块处理
        chunks_num = padded_x.size(2) // 512
        data_reshaped = padded_x.view(padded_x.size(0), padded_x.size(1), 512, chunks_num)
        return shortcut, data_reshaped, chunks_num

    def _preprocess_segments_shifted(self, x):
        """数据预处理和分块"""
        # 填充到能被512整除的长度
        padded_x = x
        shortcut = padded_x
        #第二次不再进行位置嵌入
        # 分块处理
        chunks_num = padded_x.size(2) // 512
        data_reshaped = padded_x.view(padded_x.size(0), padded_x.size(1), 512, chunks_num)
        return shortcut, data_reshaped, chunks_num

    def _process_intra_segment(self, data_reshaped, chunks_num):
        """段内注意力处理"""
        outputs = []
        segment_embeddings = []

        for i in range(chunks_num):
            segment = data_reshaped[:, :, :, i]
            segment = segment.permute(2, 0, 1)  # [512, B, C]
            # 段内Transformer
            local_output = self.local_transformer(segment)
            outputs.append(local_output)

            # 段嵌入提取
            segment_embedding = torch.mean(local_output, dim=0)  # [B, C]
            segment_embeddings.append(segment_embedding)

        return outputs, torch.stack(segment_embeddings, dim=0)

    def _process_inter_segment(self, segment_embeddings):
        """段间注意力处理"""
        return self.global_transformer(segment_embeddings)

    def _fuse_features(self, local_outputs, global_output, chunks_num):
        """特征融合"""
        enhanced_outputs = []
        for i in range(chunks_num):
            # 全局特征扩展
            global_feat = global_output[i].unsqueeze(0)  # [1, B, C]
            global_feat = global_feat.expand_as(local_outputs[i])

            # 特征相加融合
            enhanced = local_outputs[i] + global_feat
            enhanced_outputs.append(enhanced)

        return torch.cat(enhanced_outputs, dim=0)

    def _shift_processing(self, x):
        """仅包含段内注意力的移位处理"""
        # 向右填充256个零 [B, C, L] -> [B, C, L+256]
        shifted = F.pad(x, (256, 0), mode='constant', value=0)
        # 截取有效部分 [B, C, L]
        shifted = shifted[:, :, 256:-256] if shifted.size(2) > shifted.size(2) + 256 else shifted[:, :, 256:]
        # 重新分块处理 (保持512长度)
        _, data_reshaped_shift, chunks_num_shift = self._preprocess_segments_shifted(shifted)
        # 仅进行段内注意力处理
        shift_outputs = []
        for i in range(chunks_num_shift):
            segment = data_reshaped_shift[:, :, :, i]
            segment = segment.permute(2, 0, 1)  # [512, B, C]
            local_output = self.local_transformer(segment)  # 使用独立的段内Transformer
            shift_outputs.append(local_output)

        return torch.cat(shift_outputs, dim=0)  # [N*512, B, C]

    def forward(self, x):
        # 初始卷积处理
        x = self.merging1(x)
        # 预处理和分块
        shortcut, data_reshaped, chunks_num = self._preprocess_segments(x)
        # 段内处理
        local_outputs, segment_embeddings = self._process_intra_segment(data_reshaped, chunks_num)
        # 段间处理
        global_output = self._process_inter_segment(segment_embeddings)
        # 特征融合
        final_output = self._fuse_features(local_outputs, global_output, chunks_num)
        # 移位处理
        shift_output = self._shift_processing(final_output.permute(1, 2, 0))

        final_output = shift_output
        # 后续处理
        x = final_output.permute(1, 2, 0)
        x = x + shortcut


        x = self.merging2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def My_model(args):
    return  multi_transformer(args.input_channels, args.num_classes, args.max_len)

