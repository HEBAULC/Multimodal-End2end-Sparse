"""
Ref paper: Tensor Fusion Network for Multimodal Sentiment Analysis
Ref url: https://github.com/Justin1904/TensorFusionNetworks
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['SubNet', 'TextSubNet']
# 子网络
class SubNet(nn.Module):
    '''
    The subnetwork that is used in TFN for video and audio in the pre-fusion stage
    TFN 中用于预融合阶段(融合之前的阶段)的视频和音频的子网络
    '''

    def __init__(self, in_size, hidden_size, dropout):
        '''
        Args:
            in_size: input dimension
            in_size: 输入维度

            hidden_size: hidden layer dimension
            hidden_size: 隐藏层维度

            dropout: dropout probability
            dropout: 丢弃率
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        输出：
            (返回一个前向传播后的值) 一个具有(batch_size, hidden_size)形状的张量
        '''
        super(SubNet, self).__init__()
        # 对 2D 或 3D 输入应用批量标准化（1D 输入的小批量，具有可选的附加通道维度）
        self.norm = nn.BatchNorm1d(in_size)
        # 随机失去活性
        self.drop = nn.Dropout(p=dropout)
        # 线型层1: in_size -> hidden_size
        self.linear_1 = nn.Linear(in_size, hidden_size)

        # 线型层2: hidden_size -> hidden_size
        self.linear_2 = nn.Linear(hidden_size, hidden_size)

        # 线型层3: hidden_size -> hidden_size
        self.linear_3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        # 批量归一化
        normed = self.norm(x)

        # 随机失去活性
        dropped = self.drop(normed)

        # 线型层1处理 + relu激活函数
        y_1 = F.relu(self.linear_1(dropped))

        # 线型层2处理 + relu激活函数
        y_2 = F.relu(self.linear_2(y_1))

        # 线型层3处理 + relu激活函数
        y_3 = F.relu(self.linear_3(y_2))

        # 返回y_3
        return y_3


class TextSubNet(nn.Module):
    '''
    The LSTM-based subnetwork that is used in TFN for text
    在 TFN 中用于文本的基于 LSTM 的子网络
    '''

    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension

            hidden_size: hidden layer dimension

            num_layers: specify the number of layers of LSTMs.
            num_layers: 指定 LSTM 的层数 默认1层

            dropout: dropout probability

            bidirectional: specify usage of bidirectional LSTM
            bidirectional: 指定双向 LSTM 的用法 默认不是双向

        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
            返回值的形状 (batch_size, out_size)
        '''
        super(TextSubNet, self).__init__()
        if num_layers == 1:
            dropout = 0.0
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, sequence_len, in_size)
        '''
        _, final_states = self.rnn(x)
        # 经过lstm后 形状由(batch_size, sequence_len, in_size)变成(batch_size, hidden_size) 序列长度消失了 序列的in_size变成了hidden_size


        # output, (hn, cn) = rnn(input, (h0, c0))
        # final_states[0] = hn
        # .squeeze() 删除所有维度中等于1的维度
        # 在默认参数条件下，如果input的某些维度大小为1，经过压缩后会把相应维度给删除；
        # 如果给定了dim的值且对应维度上的值不为1，则不对原tensor进行降维，只能是对应维度的值为1时才会降维。
        # 提取hn
        h = self.dropout(final_states[0].squeeze())

        # 经过线性层映射后 (batch_size, hidden_size) -> (batch_size, out_size)
        y_1 = self.linear_1(h)
        return y_1
