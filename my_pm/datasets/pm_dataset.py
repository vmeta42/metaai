# -*- coding: utf-8 -*-
'''
@Time    : 4/6/2022 11:10 AM
@Author  : dong.yachao
'''

'''
用于读取数据、调用 data_processing 中的数据归一化、特征工程等
生成能够训练的 data_loader
'''
import os
from itertools import repeat  # 复制模块
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from my_pm.datasets.data_processing import cluster_undersampling, get_mutil_features_v1, get_mutil_features, preprocess
from my_pm.utils.util import gen_labels

from multiprocessing.pool import ThreadPool, Pool
num_threads = min(8, os.cpu_count())  # 定义多线程个数




# **使用多线程读取处理数据** 从文件夹 读取每一个csv数据, 对每一个csv文件数据进行处理，最后合成一起返回最终需要的data, 并分割训练测试集
def get_csv_files_datasets_v2(csv_files_path, args):
    '''

    Args:
        csv_files_path:  存放所有csv数据的文件夹的绝对路径
        args: 参数，用于传递 Tc、Tg等

    Returns: all_data: csv文件夹内所有被处理过的数据 shape=[n_samples, n_feats]
            all_labels: csv文件夹内所有被处理过数据的标签 shape=[n_samples, ]
    '''
    csv_paths = os.listdir(csv_files_path)
    all_data = None
    all_labels = None

    # TODO 转换为多线程读取处理
    with Pool(num_threads) as pool:
        pbar = tqdm(pool.imap(get_csv_datasets, zip(csv_paths, repeat(csv_files_path), repeat(args)), chunksize=4))
        for cur_final_data, cur_labels in pbar:
            if all_data is None:
                all_data = cur_final_data
            else:
                all_data = np.concatenate((all_data, cur_final_data), axis=0)

            if all_labels is None:
                all_labels = cur_labels
            else:
                all_labels = np.concatenate((all_labels, cur_labels))
    pbar.close()  # 关闭进度条
    return all_data, all_labels.ravel()



# 从文件夹 读取每一个csv数据, 对每一个csv文件数据进行处理，最后合成一起返回最终需要的data, 并分割训练测试集
def get_csv_files_datasets(csv_files_path, args):
    '''

    Args:
        csv_files_path:  存放所有csv数据的文件夹的绝对路径
        args: 参数，用于传递 Tc、Tg等

    Returns: all_data: csv文件夹内所有被处理过的数据 shape=[n_samples, n_feats]
            all_labels: csv文件夹内所有被处理过数据的标签 shape=[n_samples, ]
    '''
    csv_paths = os.listdir(csv_files_path)[:50]
    all_data = None
    all_labels = None
    # TODO 转换为多线程读取处理
    for csv_path in tqdm(csv_paths):
        # 获得每个csv文件的特征和labels [n_samples, n_feats], [n_samples]
        # [24063, 8] [24063,]
        cur_final_data, cur_labels = get_csv_datasets(zip(csv_path, csv_files_path, args))
        if all_data is None:
            all_data = cur_final_data
        else:
            all_data = np.concatenate((all_data, cur_final_data), axis=0)

        if all_labels is None:
            all_labels = cur_labels
        else:
            all_labels = np.concatenate((all_labels, cur_labels))

    return all_data, all_labels.ravel()




# 读取 **单个** csv文件数据，并经过特征工程转换为可训练的数据
def get_csv_datasets(cfgs):
    csv_path, csv_files_path, args = cfgs
    # 用于选取原始数据的特征列
    kpi = args.kpi_list
    # Tc 用于计算变化率特征
    Tc = args.Tc
    # Tg 用于计算梯度特征
    Tg = args.Tg
    # init_t 可以计算特征的时刻，由Tc和Tg确定，因为开始的数据长度不够计算
    init_t = np.max((Tc, Tg))

    if kpi is None:
        kpi = ['monomer_voltage', 'monomer_resistance', 'temperature']

    # 从csv单个文件获取 原始选取的特征(字段)数据
    csv_data = pd.read_csv(os.path.join(csv_files_path, csv_path), encoding='gbk')[kpi].dropna().values

    # TODO: 1.正样本数据降采样

    # 2.做特征工程，特征扩维
    # 获取当前t时刻的新特征
    n_samples, n_feats = csv_data.shape
    final_data = None
    for t in range(init_t, n_samples):
        # cur_t_data.shape: [new_n_feats, ] np.array  8(expand_feats) + 1(status) = 9
        cur_t_data = get_mutil_features_v1(raw_data=csv_data, t=t, Tc=Tc, Tg=Tg, D=288)
        if final_data is None:
            final_data = cur_t_data
        else:
            final_data = np.concatenate((final_data, cur_t_data), axis=0)

    labels = final_data[:, -1].astype(int).ravel()
    # 这里测试，随机生成labels
    # labels = gen_labels(n_samples, fault_rate=0.3)

    # 3.数据归一化、标准化等
    final_data = final_data[:, :-1]
    final_data = preprocess(final_data)

    return (final_data, labels)























