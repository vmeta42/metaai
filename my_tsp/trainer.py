# if self.use_split:
#     self.train_dataset, self.test_dataset = self.train_test_split(self.dataset)

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

from .datasets import data_processing
from .models.tpaLSTM import TPALSTM
from .evaluation_metrics import evaluate



def train(args, Data_utils):

    if args.L1Loss:
        criterion = nn.L1Loss(size_average=False)
    else:
        criterion = nn.MSELoss(size_average=False)

    model = TPALSTM(1, args.predict_seq_len, args.hidden_size, args.num_obs_to_train, args.n_layers)
    optimizer = Adam(model.parameters(), lr=args.lr)
    random.seed(2)

    # select sku with most top n quantities

    # 获取 data_X(训练的序列集) 和 data_Y(对应训练的标签集)
    # shape: data_X  (num_seq, num_obs_to_train, nts)
    # shape: data_Y  (num_seq, predict_seq_len, nts)
    # 生成训练集中的 X, Y， 和测试集的 X, Y

    # print('type(Data_utils.train_dataset): ', type(Data_utils.train_dataset))
    train_X, train_Y = Data_utils.train_label_generator(Data_utils.train_dataset)
    # print('type(train_Y)0 : ', type(train_Y))
    # print('type(train_X)0 : ', type(train_X))


    # 使用将nts作为batch_size训练的模型
    # data_X.shape：[nts, num_seq, num_obs_to_train]
    # data_Y.shape：[nts, num_seq, predict_seq_len]
    # data_X = np.asarray(data_X.permute(2, 0, 1))


    # training
    progress = ProgressBar()
    best_val = np.inf
    total_loss = 0
    n_samples = 0
    losses = []
    for epoch in progress(range(args.num_epoches)):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0
        n_samples = 0
        for step in tqdm(range(args.step_per_epoch)):
            # print(step)
            # 生成单个的训练样本和label

            train_X_seq, train_Y_label = data_processing.batch_generator(train_X, train_Y, args.batch_size)

            # 转换，把nts当成batch size  shape [nts, num_obs_to_train]

            train_X_seq = torch.from_numpy(train_X_seq).float().permute(1, 0)
            train_Y_label = torch.from_numpy(train_Y_label).float().permute(1, 0)

            # [nts, predict_seq_len]
            train_Y_pred = model(train_X_seq)


            # 转换
            # train_Y_pred = Data_utils.scaler.inverse_transform(train_Y_pred.permute(1, 0).detach().numpy())
            # train_Y_label = Data_utils.scaler.inverse_transform(train_Y_label)
            # #
            # train_Y_pred = torch.from_numpy(train_Y_pred).float().permute(1, 0)
            # train_Y_label = torch.from_numpy(train_Y_label).float().permute(1, 0)

            # print('max: ', Data_utils.scaler.max)
            for i in range(len(Data_utils.scaler.max)):
                train_Y_pred[i] = train_Y_pred[i] * Data_utils.scaler.max[i]
                train_Y_label[i] = train_Y_label[i] * Data_utils.scaler.max[i]


            # loss
            loss = criterion(train_Y_pred, train_Y_label)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_samples += (train_Y_label.size(0))
        train_loss = total_loss / n_samples

        test_loss, test_rae, test_corr = evaluate.evaluate(args, Data_utils, model, batch_size=None)
        print('| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | '
              'valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}'.
              format(epoch, (time.time() - epoch_start_time), train_loss, test_loss, test_rae, test_corr))

        # Save the model if the validation loss is the best we've seen so far.
        if test_loss < best_val:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val = test_loss
