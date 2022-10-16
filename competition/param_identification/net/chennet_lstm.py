# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F

class LSTM_Net(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=32, n_layer=2, n_class=1):
        super(LSTM_Net, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
        self.linear = nn.Linear(hidden_dim, n_class)
 
    def forward(self, x):                  # x's shape (batch_size, 序列长度, 序列中每个数据的长度)
        # print(x.shape)
        out, _ = self.lstm(x)              # out's shape (batch_size, 序列长度, hidden_dim)
        out = out[:, -1, :]                # 中间的序列长度取-1，表示取序列中的最后一个数据，这个数据长度为hidden_dim，
                                           # 得到的out的shape为(batch_size, hidden_dim)
        out = self.linear(out)             # 经过线性层后，out的shape为(batch_size, n_class)
        # print(out.shape)
        return out