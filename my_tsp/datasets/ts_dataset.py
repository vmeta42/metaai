from __future__ import print_function, absolute_import
import os.path as osp
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
import random
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import copy
from torch.autograd import Variable

class TimeSeriesDataset(object):
    def __init__(self, args):
        '''
        Args:
            dSet: time series dataset (numpy)
            num_obs_to_train: 单个时间序列训练的样本长度
            horizon: 取训练序列标签时，往后延 horizon 个单位开始取 (int)
            normalize: 选择数据归一化的方式
            use_split: 选择是否将原始数据集分割为训练集、测试集
            use_cuda: 选择是否用 cuda
        '''
        # n = number of periods , m = number of TS
        self.args = args
        self.use_cuda = args.use_cuda
        self.num_obs_to_train = args.num_obs_to_train
        self.predict_seq_len = args.predict_seq_len
        self.horizon = args.horizon
        self.normalize = args.normalize

    def get_loaders(self, data_path):
        '''
        将数据生成pytorch的dataloader
        Args:
            data_path:
        Returns:
        '''
        data_set = pd.read_csv(data_path)[self.args.kpi_list].values
        if len(data_set) <= (self.num_obs_to_train + self.predict_seq_len - 1 + self.horizon):
            return None, None
        else:
            # 数据归一化等
            data_set, scaler = self.data_normalize(data_set)
            # 获得训练集和对应的标签结合
            data_X, data_Y = self.train_label_generator(data_set)

            data_X = torch.FloatTensor(data_X)
            data_Y = torch.FloatTensor(data_Y)
            dataset_X_Y = TensorDataset(data_X, data_Y)
            data_iter = DataLoader(dataset_X_Y, batch_size=self.args.batch_size, shuffle=self.args.shuffle, num_workers=0, drop_last=False)
        return data_iter, scaler

    def data_normalize(self, dataset):
        scaler = MinMaxScaler()
        dataset = scaler.fit_transform(dataset)  # 后续 self.scaler可用于 inverse_transform
        return dataset, scaler

    def train_label_generator(self, dataset_X_Y):
        n, m = dataset_X_Y.shape
        num_seq = n - self.num_obs_to_train - self.predict_seq_len + 1 - self.horizon

        dataset_X = np.zeros((num_seq, self.num_obs_to_train, m))         # train set
        dataset_Y = np.zeros((num_seq, self.predict_seq_len, m))         # label

        for i in range(num_seq):
            start = i
            end = self.num_obs_to_train + i
            dataset_X[i, :, :] = dataset_X_Y[start:end, :]
            dataset_Y[i, :, :] = dataset_X_Y[end:end + self.predict_seq_len, :]

        return dataset_X, dataset_Y