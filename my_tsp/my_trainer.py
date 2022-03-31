#!/usr/bin/python 3.7
#-*-coding:utf-8-*-
import random
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import argparse
from progressbar import *
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from .datasets import data_processing
from .models.my_tpaLSTM import TPALSTM
from .models.LSTM import LSTM
from .evaluation_metrics import evaluate
from .evaluation_metrics.loss import RMSELoss, CLSLoss
from .utils.util import reg2cls


def train(args, Data_utils, train_list, test_list):

    # training
    # 回归损失函数
    if args.L1Loss:
        reg_criterion = nn.L1Loss(size_average=False)
    else:
        reg_criterion = RMSELoss()

    # 分类损失函数
    # cls_criterion = nn.BCEWithLogitsLoss()
    cls_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2]))

    # 模型创建
    model = LSTM(len(args.kpi_list), args.predict_seq_len, args.hidden_size, args.num_obs_to_train, args.n_layers)

    # 优化器
    optimizer = Adam(model.parameters(), lr=args.lr)
    # 动态调整学习率
    scheduler = MultiStepLR(optimizer, milestones=[45, 55], gamma=0.1)

    progress = ProgressBar()
    best_val = np.inf
    best_f1 = -1.0
    reg_losses = []
    # 开始训练
    for epoch in progress(range(args.num_epoches)):
        epoch_start_time = time.time()
        model.train()
        total_reg_loss = 0
        total_cls_loss = 0
        total_final_loss = 0
        n_samples = 0
        for data_path in train_list:
            # print('data_path:{}'.format(data_path))
            data_iter, scaler = Data_utils.get_loaders(data_path)
            if data_iter is None:
                continue
            for i, (train_X_seqs, train_Y_labels) in enumerate(data_iter):
                train_Y_preds, cls_ypred = model(train_X_seqs)
                # 回归损失函数
                reg_loss = reg_criterion(train_Y_preds, train_Y_labels)
                # 分类损失函数
                if args.use_cls_loss:
                    train_Y_cls_label = reg2cls(train_Y_labels, scaler)
                    # print('train_Y_cls_label:', train_Y_cls_label.shape)
                    # print('cls_ypred.squeeze().shape:', cls_ypred.squeeze().shape)
                    cls_loss = cls_criterion(cls_ypred.squeeze(), train_Y_cls_label)
                    final_loss = reg_loss + cls_loss
                else:
                    final_loss = reg_loss
                reg_losses.append(reg_loss.item())
                optimizer.zero_grad()
                # reg_loss.backward()
                final_loss.backward()
                optimizer.step()

                total_reg_loss += reg_loss.item()
                total_cls_loss += cls_loss.item()
                total_final_loss += final_loss.item()
                n_samples += 1
        train_reg_loss = total_reg_loss / n_samples
        train_cls_loss = total_cls_loss / n_samples
        train_final_loss = total_final_loss / n_samples

        print('| epoch: {:3d}/{:3d}   | time: {:5.2f}s | train_reg_loss: {:5.8f} | train_cls_loss: {:5.8f} | '.
              format(epoch+1, args.num_epoches, (time.time() - epoch_start_time), train_reg_loss, train_cls_loss))


        if (epoch+1) % (args.eval_epoch) == 0:
            eval_reg_loss, evl_reg_loss_penalty, eval_cls_loss, eval_final_loss, \
            avg_precision, avg_recall, avg_accuracy, avg_f1, result_dict, \
            eval_cls_report = evaluate.evaluate(args, Data_utils, model, test_list)

            cur_f1 = eval_cls_report['1']['f1-score']
            # Save the model if the validation loss is the best we've seen so far.
            if cur_f1 > best_f1:
                best_f1 = cur_f1
                with open(os.path.join('../model_checkpoints', args.model, args.save), 'wb') as f:
                    state_dict = {'model_state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch+1,
                                  'f1': best_f1, 'args': args}
                    torch.save(state_dict, f)

            print('| evaluate: {:3d}/{:3d}  | time: {:5.2f}s | eval_reg_loss: {:5.8f} |  eval_reg_loss_penalty:{:5.8f} | '
                  'eval_cls_loss: {:5.8f}  | eval_final_loss: {:5.8f} '
                  ' | avg_precision: {:5.8f} | avg_recall: {:5.8f}| avg_accuracy: {:5.8f}| avg_f1: {:5.8f}'
                  ' | best_f1: {:5.8f} '.
                  format(epoch + 1, args.num_epoches, (time.time() - epoch_start_time), eval_reg_loss, evl_reg_loss_penalty,
                         eval_cls_loss, eval_final_loss, avg_precision, avg_recall, avg_accuracy, avg_f1, best_f1))
            print('eval_cls_report:', eval_cls_report)

        scheduler.step()

