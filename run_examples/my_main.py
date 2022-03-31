#!/usr/bin/python 3.7
#-*-coding:utf-8-*-
import os
from torch import nn
import argparse
from progressbar import *
from torch.optim import Adam
import pandas as pd
import torch
import numpy as np
from my_tsp.datasets import data_processing, ts_dataset
from my_tsp.my_trainer import train
from my_tsp.test import test
from my_tsp.utils import util


def main(args):
    #  读取数据
    train_path = '../data/m6_train/'
    test_path = '../data/m6_test/'

    train_files = os.listdir(train_path)
    test_files = os.listdir(test_path)

    train_list = [train_path+i for i in train_files]
    test_list = [test_path+i for i in test_files]

    # dSet = pd.read_csv(data_path)[['InternalVol', 'InternalIR']].values
    # print("dSet.shape", dSet.shape)

    Data_utils = ts_dataset.TimeSeriesDataset(args)

    # 训练
    if args.train:
        print("start train mode......")
        train(args, Data_utils, train_list, test_list)

    # 测试
    if args.test:
        print("start test mode......")
        test(args, Data_utils, test_list)


if __name__ == "__main__":
    # todo 使用yaml参数配置文件，没训练一个模型一个yaml文件
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='',
                        help='location of the data file')
    parser.add_argument('--use_split', type=bool, default=True)
    parser.add_argument("--kpi_list", "-kpi", type=list, default=['InternalVol'])
    parser.add_argument('--normalize', type=str, default='max', help='数据归一化的方式')
    parser.add_argument('--horizon', type=int, default=0, help='取训练序列标签时，往后延 horizon 个单位开始取')
    parser.add_argument("--predict_seq_len", "-psl", type=int, default=10)  # 预测未来5天，,144*5
    parser.add_argument("--num_obs_to_train", "-not", type=int, default=60)   # 单个训练序列的长度 (4320)
    parser.add_argument('--v_threshold', type=float, default=12.6, help='电池电压阈值界限，小于12.6为异常状态')

    parser.add_argument('--model', type=str, default='tpaLSTM', help='')
    parser.add_argument("-lr", type=float, default=1e-4)
    parser.add_argument("--n_layers", "-nl", type=int, default=1)
    parser.add_argument("--hidden_size", "-hs", type=int, default=24)

    # 不支持 bool 类型
    parser.add_argument('--L1Loss', type=bool, default=False)
    parser.add_argument('--use_cuda', type=bool, default=False)

    parser.add_argument("--batch_size", "-b", type=int, default=144)
    parser.add_argument("--num_epoches", "-e", type=int, default=60)
    # load_model_path
    parser.add_argument('--model_path', type=str, default='lstm_model_7.tar', help='')
    # save model path
    parser.add_argument('--save', type=str,  default='lstm_model_7.tar',
                        help='path to save the final model')

    parser.add_argument("--eval_epoch", "-ee", type=int, default=1)
    parser.add_argument("--min_thre", "-min", type=float, default=12.6)
    parser.add_argument("--max_thre", "-max", type=float, default=15.0)
    parser.add_argument("--num_results_to_sample", "-nrs", type=int, default=10)
    parser.add_argument('--shuffle', type=str, choices=['True', 'False'], default='True')

    parser.add_argument("--train", "-tr", action="store_true", help='train mode')
    parser.add_argument("--test", "-te", action="store_true", help='test mode')
    parser.add_argument("--use_cls_loss", "-ucl", action="store_true", help='是否使用class loss')
    parser.add_argument("--show_plot", "-sp", action="store_true")

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


    log_path = os.path.join('..', 'model_checkpoints', args.model)
    util.mkdirs(log_path)
    sys.stdout = Logger(os.path.join(log_path, args.save.split('.')[0]) + '.txt')
    print('--------args----------\n')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('--------args----------\n')

    main(args=args)







