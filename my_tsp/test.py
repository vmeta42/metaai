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
from .evaluation_metrics import evaluate, loss
import matplotlib.pyplot as plt
from .utils.util import mkdirs, plot_whole_battery, plot_predict_battery
from .models.LSTM import LSTM


def test(args, Data_utils, test_list):
    if args.test:
        args.shuffle = False

    check_point_path = os.path.join('..', 'model_checkpoints', args.model, args.model_path)
    print('load model checkpoint from {}'.format(check_point_path))
    checkpoint = torch.load(check_point_path)
    model = LSTM(len(args.kpi_list), args.predict_seq_len, args.hidden_size, args.num_obs_to_train, args.n_layers)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('test model info: epoch:{},  best f1:{},  args:{} '.format(checkpoint['epoch'], checkpoint['f1'], checkpoint['args']))

    eval_reg_loss, evl_reg_loss_penalty, eval_cls_loss, eval_final_loss, \
    avg_precision, avg_recall, avg_accuracy, avg_f1, result_dict, eval_cls_report\
        = evaluate.evaluate(args, Data_utils, model, test_list)

    print('eval_reg_loss:{:5.8f},  evl_reg_loss_penalty:{:5.8f}, '
          ' eval_cls_loss:{:5.8f}, eval_final_loss:{:5.8f}  '
          'avg_precision:{:5.8f},  avg_recall:{:5.8f}, avg_accuracy:{:5.8f}, avg_f1:{:5.8f}'
          .format(eval_reg_loss, evl_reg_loss_penalty,
                  eval_cls_loss, eval_final_loss,
                  avg_precision, avg_recall, avg_accuracy, avg_f1))



    if args.show_plot:

        # plot_whole_battery(args, result_dict)
        plot_predict_battery(args, result_dict)
