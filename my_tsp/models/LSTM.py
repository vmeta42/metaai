import torch
from torch import nn
import torch.nn.functional as F
import argparse
from progressbar import *
from torch.optim import Adam
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
from datetime import date


class LSTM(nn.Module):

    def __init__(self, input_size, predict_seq_len, hidden_size, num_obs_to_train, n_layers, use_cls_loss=False):
        '''
        Args:
            input_size: 1
            output_horizon: args.seq_len
            hidden_size:  args.hidden_size
            obs_len: args.num_obs_to_train
            n_layers: args.n_layers
        '''
        super(LSTM, self).__init__()
        # args.seq_len:432,  args.hidden_size:24, a rgs.num_obs_to_train:4320,  args.n_layers:1

        # 转换维度  input_size 1 → hidden_size 24
        self.input_size = input_size
        self.hidden_size = hidden_size   # 24
        self.predict_seq_len = predict_seq_len
        self.num_obs_to_train = num_obs_to_train
        self.use_cls_loss = use_cls_loss
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()


        # lstm参数
        # input_size:    x的特征维度
        # hidden_size：   隐藏层的特征维度
        # num_layers：    隐藏层的层数，默认为1
        # bias：          默认为True
        # batch_first:    True则输出的数据格式为(batch, seq, feature)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers,
                            bias=True, batch_first=True)  # output (batch_size, obs_len, hidden_size)


        self.conv = nn.Conv2d(1, self.predict_seq_len, (self.num_obs_to_train, 1))  # (1, 32, )

        # 最后的输出： hidden_size 24 → x的特征维度
        self.linear = nn.Linear(hidden_size, self.input_size)

        # 用于分类  hidden_size 24 → 是否处于故障状态的类别，二分类
        self.classifier = nn.Linear(hidden_size, 1)

        # 隐藏层的层数
        self.n_layers = n_layers

    def forward(self, x):
        batch_size, num_obs, num_features = x.shape
        x = self.relu(self.hidden(x))

        # ht: [n_layers, num_col, hidden_size]  [1, 2, 24]
        ht = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        ct = ht.clone()
        # out: [batch_size, num_obs_to_train, num_features ]
        out, (ht, ct) = self.lstm(x, (ht, ct))
        out = out.view(batch_size, -1, num_obs, self.hidden_size)
        out = self.conv(out)
        out = out.view(batch_size, self.predict_seq_len, self.hidden_size)
        out = self.relu(out)
        # ypred: [B, predict_seq_len, num_features]
        ypred = self.linear(out)
        # cls_ypred: [B, predict_seq_len, 1]
        cls_ypred = self.classifier(out)
        return ypred, cls_ypred


