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

# 计算一组点中 每组 两个相邻点的斜率，最后得出最大斜率差
def cal_cls_eva_thre(np_array):
    slope_dif_list = []
    for i, cur_val in enumerate(np_array):
        if i+2 <= len(np_array)-1:
            mid_val = np_array[i+1]
            third_val = np_array[i+1]
            slope_1 = math.fabs(mid_val - cur_val)
            slope_2 = math.fabs(third_val - mid_val)
            slope_dif = math.fabs(slope_2 - slope_1)
            slope_dif_list.append(slope_dif)
    # 当前阈值/最大斜率差/拐点
    max_slope_dif = max(slope_dif_list)
    # 异常点/拐点对应的index
    abnormal_index = slope_dif_list.index(max_slope_dif) + 1
    cur_cls_thre = np_array[abnormal_index]
    return cur_cls_thre, abnormal_index

# 将预测的具体值转换为one-hot标签(二分类标签)
def reg2cls(train_Y_labels, scaler, min_thre=12.6, max_thre=15.0):
    CLS_Labels = []
    train_Y_cls_label = train_Y_labels.clone().detach()
    train_Y_cls_label = train_Y_cls_label.numpy()

    for i in range(train_Y_cls_label.shape[0]):
        cls_labels = scaler.inverse_transform(train_Y_cls_label[i])
        one_hot_labels = []
        for label in cls_labels[:, 0]:
            if label <= min_thre or label >= max_thre:
                # print('label:', label)
                one_hot_labels.append(int(1))
            elif min_thre < label < max_thre:
                one_hot_labels.append(int(0))
            else:
                print('one hot label error!!!')

        CLS_Labels.append(one_hot_labels)
    return torch.Tensor(CLS_Labels)

#
def cls_pred2label(cls_ypred, posi_thre=0.5):
    one_hot_pred_labels = []
    for i in range(cls_ypred.shape[0]):
        one_hot_pred_label = []
        for v in cls_ypred[i]:
            if v >= posi_thre:
                # print('v:', v)
                one_hot_pred_label.append(int(1))
            else:
                one_hot_pred_label.append(int(0))

        one_hot_pred_labels.append(one_hot_pred_label)

    return torch.Tensor(one_hot_pred_labels)




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
