import numpy as np
import torch
from torch import nn
from ..utils.util import cal_cls_eva_thre

DIV_CONSTANT = 1e-5

# 回归损失-均方方差的平方根
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))

# 分类损失函数
class CLSLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_criterion = nn.BCEWithLogitsLoss()

    def forward(self, cls_ypred, train_Y_labels, min_thre=12.6, max_thre=15.0):
        train_Y_cls_labels = torch.zeros(size=train_Y_labels.shape)
        # 将预测的值(回归)转换为二分类是否是异常的的标签
        for i in range(train_Y_cls_labels.shape[0]):
            for j in range(train_Y_cls_labels.shape[1]):
                # 如果处于异常阈值范围内，label为1
                if (train_Y_labels[i, j] <= min_thre) or (train_Y_labels[i, j] >= max_thre):
                    train_Y_cls_labels[i, j] = 1
                # 如果处于正常阈值范围内，label为0
                elif (min_thre < train_Y_labels[i, j] < max_thre):
                    train_Y_cls_labels[i, j] = 0
                else:
                    print('error！！！ if语句判断情况没有考虑完全，请检查！！！！')
                    train_Y_cls_labels[i, j] = 0
        # 计算分类损失
        cls_loss = self.cls_criterion(cls_ypred, train_Y_cls_labels)
        return cls_loss

# 评估方法
def cal_cls_eval_loss(result_dict, min_thre=12.6, max_thre=15, penalty=1.2):
    rmse_loss = RMSELoss()
    total_reg_loss_penalty = 0
    n_samples = 0
    precision_list = []
    recall_list = []
    accuracy_list = []
    f1_score_list = []
    for key, values in result_dict.items():
        train_seqs = values[0]
        label_seqs = values[1]
        pred_seqs = values[2]
        for i in range(len(label_seqs)):
            n_samples = n_samples + 1
            # print('pred_seqs[i]:', pred_seqs[i])
            loss = rmse_loss(torch.from_numpy(pred_seqs[i]), torch.from_numpy(label_seqs[i])).item()
            # print('loss:', loss)
            train_seq = train_seqs[i]
            label_seq = label_seqs[i]
            pred_seq = pred_seqs[i]
            # 判断label_seq是否处于异常阈值范围
            # 输入训练序列没有故障状态 但是label_seq有故障状态
            if (label_seq.min() <= min_thre or label_seq.max() >= max_thre):
            # if (label_seq.min() <= min_thre or label_seq.max() >= max_thre) and (train_seq.min() > min_thre and train_seq.max() < max_thre):
                cur_cls_thre, abnormal_index = cal_cls_eva_thre(pred_seq)
                # print('cur_cls_thre:', cur_cls_thre)
                TP = 0
                FP = 0
                TN = 0
                FN = 0
                for j in range(len(label_seq)):

                    # (TP: 实际为故障，预测为故障)
                    # 实际小于 最低阈值，预测也小于 最低阈值
                    if (label_seq[j] <= min_thre) and (pred_seq[j] <= cur_cls_thre):
                        TP = TP + 1
                        # print('label_seq[j]:{}, pred_seq[j]:{} '.format(label_seq[j], pred_seq[j]))
                    # 实际大于 最高阈值，预测也大于 最高阈值
                    elif (label_seq[j] >= max_thre) and (pred_seq[j] >= max_thre):
                        TP = TP + 1
                        # print('label_seq[j]:{}, pred_seq[j]:{} '.format(label_seq[j], pred_seq[j]))

                    # (TN: 实际为正常，预测为正常)
                    # 实际处于 正常阈值，预测也处于 正常阈值
                    elif (min_thre < label_seq[j] < max_thre) and (cur_cls_thre < pred_seq[j] < max_thre):
                        TN = TN + 1

                    # (FP: 实际为正常，预测为故障)
                    # 实际处于 正常阈值，预测小于 最低阈值 或  大于最高阈值
                    elif (min_thre < label_seq[j] < max_thre) and \
                            ((pred_seq[j] <= cur_cls_thre) or (pred_seq[j] >= max_thre)):
                        FP = FP + 1

                    # (FN: 实际为故障，预测为正常)
                    # 实际大于 最高阈值，预测小于 最高阈值
                    elif (label_seq[j] >= max_thre) and (cur_cls_thre < pred_seq[j] < max_thre):
                        FN = FN + 1
                    # 实际小于 最低阈值，预测大于 最低阈值
                    elif (label_seq[j] <= max_thre) and (cur_cls_thre < pred_seq[j] < max_thre):
                        FN = FN + 1
                    else:
                        print('出现错误，情况没有考虑完全，请检查！！！！！！！！！！')
                        FN = FN + 1
                # 得出混淆矩阵 计算各种评估指标
                precision = float(TP/(TP+FP+DIV_CONSTANT))
                recall = float(TP/(TP+FN+DIV_CONSTANT))
                accuracy = float((TP+TN)/(TP+TN+FP+FN+DIV_CONSTANT))
                f1_score = float(2.*(precision*recall)/(precision+recall+DIV_CONSTANT))
                precision_list.append(precision)
                recall_list.append(recall)
                accuracy_list.append(accuracy)
                f1_score_list.append(f1_score)
                # print('TP:{}, FP:{}, TN:{}, FN:{}, precision:{}, recall:{}, accuracy:{}, f1:{}'.
                #       format(TP, FP, TN, FN, precision, recall, accuracy, f1_score))
                if precision < 0.8:
                    total_reg_loss_penalty = total_reg_loss_penalty + penalty*loss
                else:
                    total_reg_loss_penalty = total_reg_loss_penalty + loss

            else:
                # TODO: 真实值label故障点一个都没出现，但是预测错误 预测中出现故障点
                total_reg_loss_penalty = total_reg_loss_penalty + loss
    return total_reg_loss_penalty / n_samples, \
           sum(precision_list)/len(precision_list), \
           sum(recall_list)/len(recall_list), \
           sum(accuracy_list)/len(accuracy_list), \
           sum(f1_score_list)/len(f1_score_list),

class loss:
    def __init__(self):
        super(loss, self).__init__()

    def MSELoss(self, ypred, ytrue):
        mse_loss = nn.MSELoss(size_average=False)
        mse_loss_value = mse_loss(ypred, ytrue).item()
        return mse_loss_value

    def L1Loss(self, ypred, ytrue):
        l1_loss = nn.L1Loss(size_average=False)
        l1_loss_value = l1_loss(ypred, ytrue).item()
        return l1_loss_value

def RSE(ypred, ytrue):
    rse = np.sqrt(np.square(ypred - ytrue).sum()) / np.sqrt(np.square(ytrue - ytrue.mean()).sum())
    return rse

def quantile_loss(ytrue, ypred, qs):
    '''
    Quantile loss version 2
    Args:
    ytrue (batch_size, output_horizon)
    ypred (batch_size, output_horizon, num_quantiles)
    '''
    L = np.zeros_like(ytrue)
    for i, q in enumerate(qs):
        yq = ypred[:, :, i]
        diff = yq - ytrue
        L += np.max(q * diff, (q - 1) * diff)
    return L.mean()

def SMAPE(ytrue, ypred):
    ytrue = np.array(ytrue).ravel()
    ypred = np.array(ypred).ravel() + 1e-4
    mean_y = (ytrue + ypred) / 2.
    return np.mean(np.abs((ytrue - ypred) / mean_y))

def MAPE(ytrue, ypred):
    ytrue = np.array(ytrue).ravel() + 1e-4
    ypred = np.array(ypred).ravel()
    return np.mean(np.abs((ytrue - ypred) / ytrue))

def gaussian_likelihood_loss(z, mu, sigma):
    '''
    Gaussian Liklihood Loss
    Args:
    z (tensor): true observations, shape (num_ts, num_periods)
    mu (tensor): mean, shape (num_ts, num_periods)
    sigma (tensor): standard deviation, shape (num_ts, num_periods)

    likelihood:
    (2 pi sigma^2)^(-1/2) exp(-(z - mu)^2 / (2 sigma^2))

    log likelihood:
    -1/2 * (log (2 pi) + 2 * log (sigma)) - (z - mu)^2 / (2 sigma^2)
    '''
    negative_likelihood = torch.log(sigma + 1) + (z - mu) ** 2 / (2 * sigma ** 2) + 6
    return negative_likelihood.mean()

def negative_binomial_loss(ytrue, mu, alpha):
    '''
    Negative Binomial Sample
    Args:
    ytrue (array like)
    mu (array like)
    alpha (array like)

    maximuze log l_{nb} = log Gamma(z + 1/alpha) - log Gamma(z + 1) - log Gamma(1 / alpha)
                - 1 / alpha * log (1 + alpha * mu) + z * log (alpha * mu / (1 + alpha * mu))

    minimize loss = - log l_{nb}

    Note: torch.lgamma: log Gamma function
    '''
    batch_size, seq_len = ytrue.size()
    likelihood = torch.lgamma(ytrue + 1. / alpha) - torch.lgamma(ytrue + 1) - torch.lgamma(1. / alpha) \
        - 1. / alpha * torch.log(1 + alpha * mu) \
        + ytrue * torch.log(alpha * mu / (1 + alpha * mu))
    return - likelihood.mean()