from __future__ import print_function, absolute_import
import os.path as osp
import os
import random
import torch
import numpy as np
import copy
from torch.autograd import Variable

class Data_utility(object):
    def __init__(self, dSet, num_obs_to_train, predict_seq_len, horizon=0, normalize='max', use_split=True, use_cuda=False):
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
        self.dataset = dSet
        self.use_cuda = use_cuda
        self.num_obs_to_train = num_obs_to_train
        self.predict_seq_len = predict_seq_len
        self.horizon = horizon
        self.normalize = normalize
        self.use_split = use_split

        self.train_dataset = None
        self.test_dataset = None

        self.train_dataset_X = None
        self.train_dataset_Y = None
        self.test_dataset_X = None
        self.test_dataset_Y = None
    # 残差标准误差(residual standard error), 衡量总体离散程度的统计量
    # numpy.std() 内部函数是除以n的(有偏的); pandas.std()默认是除以n-1的(无偏的)
    def rse(self, dataset):
        tmp = copy.deepcopy(dataset)
        tmp = torch.from_numpy(tmp)
        data_rse = tmp.std() * np.sqrt((len(tmp) - 1.)/(len(tmp)))
        return data_rse

    # 残差绝对值误差
    def rae(self, dataset):
        tmp = copy.deepcopy(dataset)
        tmp = torch.from_numpy(tmp)
        data_rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))
        return data_rae

    # 数据标准化、归一化等
    def data_normalize(self, dataset):
        self.scaler = self._normalize_func()
        self.dataset = self.scaler.fit_transform(dataset)  # 后续 self.scaler可用于 inverse_transform
        return self.scaler

    # 数据归一化的方式
    def _normalize_func(self):
        if self.normalize == 'standar':
            scaler = StandardScaler()

        elif self.normalize == 'max':
            scaler = MaxScaler()

        elif self.normalize == 'mean':
            scaler = MeanScaler()

        elif self.normalize == 'log':
            scaler = LogScaler()
        else:
            print('There is no {} normalize function, so use the max scaler instead! '.format(self.normalize))
            scaler = MaxScaler()

        return scaler


    # 分割训练集和测试集
    def train_test_split(self, dataset, train_ratio=0.7):
        '''
        划分训练集、测试集
        Args:
            dataset: (归一化后的)数据集 (numpy)
            train_ratio: 训练集分割比例
        Returns:
            train_dataset: 训练集 (numpy)
            test_dataset: 测试集  (numpy)
        '''
        # 时间序列长度，特征数量
        num_periods, num_kpi = dataset.shape
        # 通过切分时间序列长度，分割训练集测试集
        train_periods = int(num_periods * train_ratio)
        train_dataset = dataset[:train_periods, :]
        test_dataset = dataset[train_periods:, :]

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset



    def train_label_generator(self, dataset_X_Y):

        n, m = dataset_X_Y.shape
        # 计算一条完整训练集里有多少条子训练集序列
        # eg. 完整训练集:1-20, 子训练集长度为5，需要预测未来2个点，则X:[1-5, 2-6,...], Y:[[6,7], [8, 9]..)
        # 则[X, Y]的个数为 20 - 5 - 2 + 1
        num_seq = n - self.num_obs_to_train - self.predict_seq_len + 1 - self.horizon
        dataset_X = np.zeros((num_seq, self.num_obs_to_train, m))         # train set
        dataset_Y = np.zeros((num_seq, self.predict_seq_len, m))         # label
        # print('idx_set:{}, n:{}'.format(idx_set, n))
        # print('X.shape:{}, Y.shape:{}'.format(X.shape, Y.shape))

        # 获得训练集X和对应的标签Y
        for i in range(num_seq):
            start = i
            end = self.num_obs_to_train + i
            dataset_X[i, :, :] = dataset_X_Y[start:end, :]
            dataset_Y[i, :, :] = dataset_X_Y[end:end+self.predict_seq_len, :]
        return dataset_X, dataset_Y

def batch_generator(dataset_X, dataset_Y, batch_size, rand_choice=True):
    # print('type(dataset_Y): ', type(dataset_Y))
    # 序列个数， 一条序列的长度(一条序列的观测点个数)， 特征维度数
    num_seq, num_periods, num_ts = dataset_X.shape
    # print('num_seq: ', num_seq)
    # if num_ts < batch_size:
    #     batch_size = num_ts

    # 将时子序列随机打散训练
    if rand_choice:
        t = random.choice(range(num_seq))  # 随机选择哪一个时间序列训练

    # 训练和测试集打乱顺序
    X = dataset_X[t, :, :]
    Y = dataset_Y[t, :, :]
    # print('type(Y): ', type(Y))
    return X, Y

def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))

class StandardScaler:

    def fit_transform(self, y):
        self.mean = np.mean(y, axis=0)
        self.std = np.std(y) + 1e-4
        return (y - self.mean) / self.std

    def inverse_transform(self, y):
        return y * self.std + self.mean

    def transform(self, y):
        return (y - self.mean) / self.std

class MaxScaler:

    def fit_transform(self, y):
        self.max = np.max(y, axis=0)
        return y / self.max

    def inverse_transform(self, y):
        # print('self.max:', self.max)
        return y * self.max

    def transform(self, y):
        return y / self.max

class MeanScaler:

    def fit_transform(self, y):
        self.mean = np.mean(y, axis=0)
        return y / self.mean

    def inverse_transform(self, y):
        return y * self.mean

    def transform(self, y):
        return y / self.mean

class LogScaler:

    def fit_transform(self, y):
        return np.log1p(y)

    def inverse_transform(self, y):
        return np.expm1(y)

    def transform(self, y):
        return np.log1p(y)