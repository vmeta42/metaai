import numpy as np
import torch
from torch import nn
from progressbar import *
import random
from ..datasets import data_processing
from .loss import RMSELoss, CLSLoss, cal_cls_eval_loss
from torch.optim import Adam
from ..utils.util import reg2cls, cls_pred2label
from sklearn.metrics import classification_report, confusion_matrix


def evaluate(args, Data_utils, model, test_list):
    # evaluation
    reg_criterion = RMSELoss()
    cls_criterion = CLSLoss()
    model.eval()
    progress = ProgressBar()
    epoch_start_time = time.time()
    total_reg_loss = 0
    total_cls_loss = 0
    total_final_loss = 0
    n_samples = 0
    result_dict = {}

    # 用于sklearn库里的评价指标库
    all_preds = np.array([], dtype=int)
    all_labels = np.array([], dtype=int)

    for data_path in test_list:
        data_iter, scaler = Data_utils.get_loaders(data_path)
        if data_iter is None:
            continue
        Labels = []
        Predicts = []
        Trains = []
        for i, (train_X_seqs, train_Y_labels) in enumerate(data_iter):
            train_Y_preds, cls_ypred = model(train_X_seqs)

            # loss
            reg_loss = reg_criterion(train_Y_preds, train_Y_labels)
            if args.use_cls_loss:
                # 将预测的真实标签值 按照正常、故障阈值转换为分类的one-hot标签
                train_Y_cls_label = reg2cls(train_Y_labels, scaler)
                # 计算分类损失
                cls_loss = cls_criterion(cls_ypred.squeeze(), train_Y_cls_label)
                # 计算分类的评估指标
                # 将预测的分类概率转换为0,1的标签，概率大于0.5的记为正样本，反之记为负样本
                cls_ypred_one_hot = cls_pred2label(cls_ypred, posi_thre=0.5)
                # 使用sklearn计算评估指标
                all_labels = np.append(all_labels, train_Y_cls_label.numpy())
                all_preds = np.append(all_preds, cls_ypred_one_hot.numpy())
                final_loss = reg_loss + cls_loss
            else:
                final_loss = reg_loss

            total_reg_loss += reg_loss.item()
            total_cls_loss += cls_loss.item()
            total_final_loss += final_loss.item()
            n_samples += 1

            train_X_seqs = train_X_seqs.detach().numpy()
            train_Y_labels = train_Y_labels.detach().numpy()
            train_Y_preds = train_Y_preds.detach().numpy()
            for i in range(train_Y_labels.shape[0]):
                Trains.append(scaler.inverse_transform(train_X_seqs[i]))
                Labels.append(scaler.inverse_transform(train_Y_labels[i]))
                Predicts.append(scaler.inverse_transform(train_Y_preds[i]))

        result_dict[data_path] = [Trains, Labels, Predicts]

    eval_reg_loss = total_reg_loss / n_samples
    eval_cls_loss = total_cls_loss / n_samples
    eval_final_loss = total_final_loss / n_samples

    # 计算平均精度值(只包含预测的label存在故障点的情况，
    # 通过将预测回归到的参数值通过最大斜率差的动态阈值方法转换为分类标签)
    evl_reg_loss_penalty, avg_precision, avg_recall, avg_accuracy, avg_f1 = cal_cls_eval_loss(result_dict)

    # 使用sklearn的方法   计算平均精度值(包括预测的label没有故障点，
    # 通过分类的概率得到分类标签)
    all_labels = all_labels.astype(int)
    all_preds = all_preds.astype(int)

    eval_cls_report = classification_report(all_labels, all_preds, output_dict=True)
    # 混淆矩阵
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    print('tn:{}, fp:{}, fn:{}, tp:{}'.format(tn, fp, fn, tp))
    print('eval_cls_report:', eval_cls_report)
    return eval_reg_loss, evl_reg_loss_penalty, eval_cls_loss, eval_final_loss, \
           avg_precision, avg_recall, avg_accuracy, avg_f1, \
           result_dict, eval_cls_report









