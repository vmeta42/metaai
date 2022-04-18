# -*- coding: utf-8 -*-
'''
@Time    : 4/14/2022 11:15 AM
@Author  : dong.yachao
'''

import os
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchmetrics

# 创建文件
def mkdirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)



# 随机生成labels，用于跑通代码、测试
def gen_labels(n_samples, fault_rate=0.3):
    n_fault = int(n_samples * fault_rate)
    n_normal = n_samples - n_fault

    fault = np.array([1] * n_fault)
    normal = np.array([1] * n_normal)

    labels = np.append(fault, normal)

    np.random.shuffle(labels)
    return labels














def plot_whole_battery(args, result_dict):
    # 比如预测未来10天的值，将所有预测第一天的值串联起来组成单个电池样本总天数(365天)的预测曲线，
    # 则每个单个电池样本有10个预测曲线图，分别采取的是第一天的点、第二天的点...
    for key, values in result_dict.items():
        train_seqs = values[0]
        label_seqs = values[1]
        pred_seqs = values[2]

        label_seqs = np.array(label_seqs)
        pred_seqs = np.array(pred_seqs)
        for i in range(label_seqs.shape[1]):
            # 定义一张图
            plt.figure(1, figsize=(25, 10))
            # 画坐标轴名称
            plt.xlabel("Periods")
            plt.ylabel("Y")
            # 画图的标题
            plt.title(str(key) + '_' + str(i))

            # 画真实值序列
            plt.plot(range(label_seqs.shape[0]), label_seqs[:, i, 0], "k-")
            # 画预测值
            plt.plot(range(pred_seqs.shape[0]), pred_seqs[:, i, 0], "r-")

            # plt.vlines(args.predict_seq_len*(i+1), plt.ylim()[0], plt.ylim()[1], color="blue", linestyles="dashed", linewidth=2)
            # 保存
            plt_save_path = os.path.join('..', 'outputs_pics', args.model,
                                         args.model_path.split('.')[0],
                                         os.path.basename(key).split('.')[0])
            # plt_save_path = '../outputs_pics/' + os.path.basename(args.save).split('.')[0]\
            #                 + '/' + os.path.basename(key).split('.')[0] + '/'
            mkdirs(plt_save_path)
            # if not os.path.exists(plt_save_path):
            #     os.makedirs(plt_save_path)
            plt.savefig(os.path.join(plt_save_path, os.path.basename(key).split('.')[0] + '_' + str(i) + '.png'))
            plt.clf()


def plot_predict_battery(args, result_dict):
    # 比如输入训练序列长度为60， 预测长度为10，将训练的60、预测的10、真实的10画出来
    # 只挑选60中没有故障点 且 真实的10中有故障点
    for j, (key, values) in enumerate(result_dict.items()):
        # key: 单个电池的名称
        train_seqs = values[0]
        label_seqs = values[1]
        pred_seqs = values[2]

        # 定义一张图
        plt.figure(1, figsize=(25, 10))
        ax_i = 0
        for i in range(len(label_seqs)):
            train_seq = train_seqs[i]
            label_seq = label_seqs[i]
            pred_seq = pred_seqs[i]

            # 只挑选60中没有故障点 且 真实的10中有故障点
            if (label_seq.min() <= args.min_thre or label_seq.max() >= args.max_thre) and \
                    (train_seq.min() > args.min_thre and train_seq.max() < args.max_thre):
                ax_i = ax_i + 1
                if ax_i > 12:
                    continue
                ax = plt.subplot(3, 4, ax_i)
                # 画坐标轴名称
                # plt.xlabel("Periods")
                # plt.ylabel("Y")
                # 画图的标题
                plt.title(str(key) + '_' + str(i))
                # 画输入训练值序列
                plt.plot(range(len(train_seq)), train_seq, "k-")
                # 画真实值序列
                plt.plot(range(len(train_seq), len(train_seq)+len(label_seq)), label_seq, "k-")
                # 画预测值
                plt.plot(range(len(train_seq), len(train_seq)+len(label_seq)), pred_seq, "r-")

                # plt.vlines(args.predict_seq_len*(i+1), plt.ylim()[0], plt.ylim()[1], color="blue", linestyles="dashed", linewidth=2)
        # 保存
        plt_save_path = os.path.join('..', 'outputs_pics', args.model,
                                     args.model_path.split('.')[0])
        # plt_save_path = '../outputs_pics/' + os.path.basename(args.save).split('.')[0]\
        #                 + '/' + os.path.basename(key).split('.')[0] + '/'
        mkdirs(plt_save_path)
        # if not os.path.exists(plt_save_path):
        #     os.makedirs(plt_save_path)
        plt.savefig(os.path.join(plt_save_path, os.path.basename(key).split('.')[0] + '_' + str(j) + '.png'))
        plt.clf()
