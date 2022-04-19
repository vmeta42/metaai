#!/usr/bin/python 3.7
#-*-coding:utf-8-*-
from torch import nn
import argparse
from progressbar import *
from torch.optim import Adam
import pandas as pd
import numpy as np
from my_tsp.datasets import data_processing
from my_tsp.trainer import train


def main(args):
    #  读取数据
    data_path = '../data/2-2.csv'
    dSet = pd.read_csv(data_path)[['InternalVol', 'InternalIR']].values
    print("dSet.shape", dSet.shape)

    Data_utils = data_processing.Data_utility(dSet, args.num_obs_to_train, args.predict_seq_len,
                                        args.horizon, args.normalize, args.use_split, args.use_cuda)
    # Data = Data_utility(dSet, 0.6, 0.2, args.cuda, args.horizon, args.window, args.normalize);

    # 数据归一化
    data_scaler = Data_utils.data_normalize(Data_utils.dataset)


    # 分割训练集和测试集
    # if args.use_split:
    Data_utils.train_test_split(Data_utils.dataset)




    # 训练
    train(args, Data_utils)

    # print(Data.rse)
    # train(Data, args)






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='',
                        help='location of the data file')
    parser.add_argument('--horizon', type=int, default=0, help='取训练序列标签时，往后延 horizon 个单位开始取')

    parser.add_argument('--normalize', type=str, default='max', help='数据归一化的方式')
    parser.add_argument('--model', type=str, default='tpaLSTM', help='')
    parser.add_argument('--L1Loss', type=bool, default=False)
    parser.add_argument('--save', type=str,  default='../model_checkpoints/bt_model.pt',
                        help='path to save the final model')

    parser.add_argument("--num_epoches", "-e", type=int, default=100)
    parser.add_argument("--step_per_epoch", "-spe", type=int, default=100)
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("--n_layers", "-nl", type=int, default=1)
    parser.add_argument("--hidden_size", "-hs", type=int, default=24)
    parser.add_argument('--use_split', type=bool, default=True)
    parser.add_argument('--use_cuda', type=bool, default=False)

    parser.add_argument("--predict_seq_len", "-psl", type=int, default=720)  # 预测未来5天，,144*5
    parser.add_argument("--num_obs_to_train", "-not", type=int, default=144*30)   # 单个训练序列的长度

    parser.add_argument("--num_results_to_sample", "-nrs", type=int, default=10)
    parser.add_argument("--show_plot", "-sp", action="store_true")
    parser.add_argument("--batch_size", "-b", type=int, default=64)
    args = parser.parse_args()


    main(args=args)







