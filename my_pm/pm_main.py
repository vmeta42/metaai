# -*- coding: utf-8 -*-
'''
@Time    : 4/13/2022 9:58 AM
@Author  : dong.yachao
'''

#!/usr/bin/python 3.7
#-*-coding:utf-8-*-
import os
from torch import nn
import argparse
from progressbar import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from my_pm.datasets import auto_labeling, data_processing, pm_dataset
from my_trainer import train
from my_pm.utils import util


def main(args):
    #  读取所有的csv数据，并进行处理
    pred_report = train(args)

    print('results:', pred_report)



if __name__ == "__main__":
    # TODO 每种机器学习模型配置一个模型参数文件
    parser = argparse.ArgumentParser()

    parser.add_argument('--use_local_data', type=bool, default=True, help='初始使用原始csv数据训练则设置为False，'
                                                                           '如果已经将训练集和测试集保存到本地了则设置为True，'
                                                                           '方便每一次训练测试加载、处理数据的速度')
    # 初始化读取处理所有的csv文件
    parser.add_argument('--csv_files_path', type=str, default='D:\DeskFiles\metaAI相关资料\\battery_datasets\BatteryfaultMarkingData-logic_NO2\\', help='存放csv文件夹的路径')
    #保存成本地文件(已处理好、可直接训练的训练集、测试集文件)
    parser.add_argument('--train_cache_path', type=str, default='D:\CodeFiles\data\metaAI_data\data\pm_datasets\\train_cachev2.npy', help='存放到本地的训练集文件')
    parser.add_argument('--test_cache_path', type=str, default='D:\CodeFiles\data\metaAI_data\data\pm_datasets\\test_cachev2.npy', help='存放到本地的测试集文件')

    parser.add_argument('--use_split', type=bool, default=True)
    parser.add_argument("--kpi_list", "-kpi", type=list, default=['monomer_voltage', 'monomer_resistance', 'temperature', 'battery_status'])
    parser.add_argument('--normalize', type=str, default='max', help='数据归一化的方式')
    parser.add_argument('--Tc', type=int, default=288*7, help='变化率特征')
    parser.add_argument('--Tg', type=int, default=288*7, help='梯度特征')

    parser.add_argument('--model', type=str, default='gbdt', help='')
    parser.add_argument('--model_path', type=str, default='D:\DeskFiles\meta-AI\model_checkpoints\pm_checkpoints', help='保存训练好模型的路径')


    args = parser.parse_args()


    class Logger(object):
        def __init__(self, fileN='Default.log'):
            self.terminal = sys.stdout
            self.log = open(fileN, 'a+')

        def write(self, message):
            ''' print实际相当于sys.stdout.write '''
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    # log_path = os.path.join('..', 'model_checkpoints', args.model)
    # util.mkdirs(log_path)
    # sys.stdout = Logger(os.path.join(log_path, args.save.split('.')[0]) + '.txt')
    # print('--------args----------\n')
    # for k in list(vars(args).keys()):
    #     print('%s: %s' % (k, vars(args)[k]))
    # print('--------args----------\n')

    main(args=args)







