# -*- coding: utf-8 -*-
'''
@Time    : 4/6/2022 11:10 AM
@Author  : dong.yachao
'''

'''
用于数据处理，如归一化、标准化、聚类解决类别不平衡问题、特征工程等
'''
import os
from multiprocessing.pool import ThreadPool, Pool
import pandas as pd
import numpy as np
from scipy.optimize import leastsq
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler




# 用于归一化、标准化等操作
def preprocess(data, mode=0):
    final_data = None
    if mode == 0:
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler

    final_data = scaler.fit_transform(data)

    return final_data


'''
k-means聚类欠采样解决类比不平衡问题
'''
def cluster_undersampling(data, n_clusters=50, n=1000):
    '''
    正样本降采样，聚类，对于每个类选取距离质心距离最近的n个样本作为代表
    Args:
        data: [n_samples, n_feats], 输入到模型中训练的正样本数据
        n_clusters: 聚类类别的个数
        n: 每个类别选出n个样本

    Returns: final_data
    '''
    # 经过降采样后的数据
    final_data = None
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
    # 聚类的类别
    labels = kmeans.labels_
    # 距离矩阵 [n_samples, n_clusters]
    dst = kmeans.transform(data)

    # 按照类别筛选出数据 {类别:矩阵距离， 1:距离矩阵...}
    # 遍历每个类别，得到每个类别的距离聚类质心最近的n个样本
    for label in np.unique(labels):
        # 先将距离矩阵按照类别分类，然后对距离(根据聚类类别定位)进行从小到大排序，得到当前类别 距离当前类别质心距离最近的n个样本的index
        sort_index = dst[labels == label][:, int(label)].argsort()[:n]
        # cur_label_data.shape: [n_samples, n_features]
        cur_label_data = data[labels == label][sort_index]
        if final_data is None:
            final_data = cur_label_data
        else:
            final_data = np.concatenate((final_data, cur_label_data), axis=0)
        # print('label: {}\n X:{}\n'.format(label, cur_label_data))

    return final_data

DIV = 1e-5

'''
特征工程
将原始的三维特征：电压、内阻、温度，扩展到14维度
'''
def get_mutil_features(raw_data, t, Vt_mean_pack, Vt_std_pack, Rt_mean_pack, Rt_std_pack, Tc, Tg, D=288):
    '''
    返回当前时刻 t 的 14维特征
    Args:
        raw_data: 原始数据，需要当前时刻 t 以前的值 用于计算特征，
        t: 当前时刻，即在时间跨度中属于第几个数据
        Vt_mean_pack: 当前时刻电池组平均电压
        Vt_std_pack:  当前时刻电池组电压的标准差
        Rt_mean_pack: 当前时刻电池组平均内阻
        Rt_std_pack:  当前时刻电池组内阻的标准差
        Tc:  计算电压内阻 变化率 的时间点
        Tg:  计算电压内阻 梯度 的时间点
        D:  计算 电压内阻 变化率用到的值，为一天测量的数目

    Returns: final_feats 当前时刻t的14维特征
    '''

    # raw_data: [N, n_batteries, time_period, Vt, Rt, Tt, ...]
    # raw_data: [N个电池组， 一个电池组包含n_batteries个电池，该的电池时间序列长度，测点值]

    # raw_data: [N, n_values]  [N代表time_period, n_values]

    # TODO 根据真实的 raw_data 数据的顺序以及是否为多维数据修改
    # 电池的基本特征
    Vt = raw_data[t, 0]
    Rt = raw_data[t, 1]
    Tt = raw_data[t, 2]

    # 电池组的相关特征
    relative_Vt = Vt - Vt_mean_pack
    relative_Rt = Rt - Rt_mean_pack

    # 电池时间序列特征
    V_change_rate = Vt - np.mean(raw_data[t-Tc:t-Tc+D, 0])
    R_change_rate = Vt - np.mean(raw_data[t-Tc:t-Tc+D, 1])

    V_Tg = raw_data[t-Tg:t+1, 0]  # Yi
    R_Tg = raw_data[t-Tg:t+1, 1]

    V_gradient = get_gradient(np.arange(t-Tg, t+1), V_Tg)[0][0]
    R_gradient = get_gradient(np.arange(t-Tg, t+1), R_Tg)[0][0]

    # 电池组合特征
    combined_feats = Vt / (Rt+DIV)

    # 返回t时刻的14维特征
    final_feats = np.array((Vt, Rt, Tt,
                            Vt_mean_pack, Vt_std_pack, relative_Vt, Rt_mean_pack, Rt_std_pack, relative_Rt,
                            V_change_rate, V_gradient, R_change_rate, R_gradient,
                            combined_feats)).reshape(1, -1)
    return final_feats


'''
特征工程
没有 电池组的信息
'''
def get_mutil_features_v1(raw_data, t, Tc, Tg, D=288):
    '''
    返回当前时刻 t 的 14维特征
    Args:
        raw_data: 原始数据，需要当前时刻 t 以前的值 用于计算特征，
        t: 当前时刻，即在时间跨度中属于第几个数据
        Tc:  计算电压内阻 变化率 的时间点
        Tg:  计算电压内阻 梯度 的时间点
        D:  计算 电压内阻 变化率用到的值，为一天测量的数目

    Returns: final_feats 当前时刻t的14维特征
    '''

    # raw_data: [N, n_batteries, time_period, Vt, Rt, Tt, ...]
    # raw_data: [N个电池组， 一个电池组包含n_batteries个电池，该的电池时间序列长度，测点值]

    # raw_data: [N, n_values]  [N代表time_period, n_values]

    # TODO 根据真实的 raw_data 数据的顺序以及是否为多维数据修改
    # 电池的基本特征
    Vt = raw_data[t, 0]
    Rt = raw_data[t, 1]
    Tt = raw_data[t, 2]

    status = raw_data[t, 3]


    # 电池时间序列特征
    V_change_rate = Vt - np.mean(raw_data[t-Tc:t-Tc+D, 0])
    R_change_rate = Vt - np.mean(raw_data[t-Tc:t-Tc+D, 1])

    V_Tg = raw_data[t-Tg:t+1, 0]  # Yi
    R_Tg = raw_data[t-Tg:t+1, 1]

    V_gradient = get_gradient(np.arange(t-Tg, t+1), V_Tg)[0][0]
    R_gradient = get_gradient(np.arange(t-Tg, t+1), R_Tg)[0][0]

    # 电池组合特征
    combined_feats = Vt / (Rt+DIV)

    # 返回t时刻的14维特征
    final_feats = np.array((Vt, Rt, Tt,
                            V_change_rate, V_gradient, R_change_rate, R_gradient,
                            combined_feats, status)).reshape(1, -1)
    return final_feats

# 计算电压内阻的梯度 (用在 get_mutil_features() 函数中)
def get_gradient(Xi, Yi):
    def fun(p, x):
        a1, a0 = p
        return a1*x + a0

    def error(p, x, y):
        return fun(p, x) - y

    p0 = np.array([1, 3])
    para = leastsq(error, p0, args=(Xi, Yi))
    return para



# 获得t时刻的电压和内阻的均值、方差
# TODO
def get_pack_value():
    pass




# 暂时没有用到，直接使用sklearn中的数据预处理库
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