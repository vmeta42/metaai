# -*- coding: utf-8 -*-
'''
@Time    : 4/6/2022 10:58 AM
@Author  : dong.yachao
'''
import os
import random
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
# 保存模型
import joblib

from my_pm.datasets import auto_labeling, data_processing, pm_dataset
from my_pm.models import cls_model

def train(args):

    if not args.use_local_data:
        csv_files_path = args.csv_files_path
        all_data, all_labels = pm_dataset.get_csv_files_datasets_v2(csv_files_path, args=args)
        X_train, X_test, y_train, y_test = train_test_split(all_data, all_labels, train_size=0.8)

        # 保存到本地.npy文件，方便读取，相当于cache
        # train dataset
        train_cache = {}
        train_cache['data'] = X_train
        train_cache['labels'] = y_train
        # test dataset
        test_cache = {}
        test_cache['data'] = X_test
        test_cache['labels'] = y_test
        # save
        np.save(args.train_cache_path, train_cache)
        np.save(args.test_cache_path, test_cache)

    else:
        # 读取本地保存的训练集测试集
        train_cache = np.load(args.train_cache_path, allow_pickle=True).item()
        test_cache = np.load(args.test_cache_path, allow_pickle=True).item()

        X_train = train_cache['data']
        y_train = train_cache['labels']

        X_test = test_cache['data']
        y_test = test_cache['labels']


    # 训练模型
    model = cls_model.train_model(use_model=args.model, X_train=X_train, y_train=y_train, args=args)

    # 保存模型到本地
    joblib.dump(model, os.path.join(args.model_path, args.model) + 'v2.lib')

    # 测试模型
    pred_report = cls_model.test_model(model=model, X_test=X_test, y_test=y_test)

    return pred_report