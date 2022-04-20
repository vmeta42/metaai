# -*- coding: utf-8 -*-
'''
@Time    : 4/14/2022 11:15 AM
@Author  : dong.yachao
'''

import os
import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchmetrics
from pathlib import Path
import re           # 用来匹配字符串（动态、模糊）的模块

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


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    """这是个用处特别广泛的函数 train.py、detect.py、test.py等都会用到
    递增路径 如 run/train/exp --> runs/train/exp{sep}0, runs/exp{sep}1 etc.
    :params path: window path   run/train/exp
    :params exist_ok: False
    :params sep: exp文件名的后缀  默认''
    :params mkdir: 是否在这里创建dir  False
    """
    path = Path(path)  # string/win路径 -> win路径
    # 如果该文件夹已经存在 则将路径run/train/exp修改为 runs/train/exp1
    if path.exists() and not exist_ok:
        # path.suffix 得到路径path的后缀  ''
        suffix = path.suffix
        # .with_suffix 将路径添加一个后缀 ''
        path = path.with_suffix('')
        # 模糊搜索和path\sep相似的路径, 存在一个list列表中 如['runs\\train\\exp', 'runs\\train\\exp1']
        # f开头表示在字符串内支持大括号内的python表达式
        dirs = glob.glob(f"{path}{sep}*")
        # r的作用是去除转义字符       path.stem: 没有后缀的文件名 exp
        # re 模糊查询模块  re.search: 查找dir中有字符串'exp/数字'的d   \d匹配数字
        # matches [None, <re.Match object; span=(11, 15), match='exp1'>]  可以看到返回span(匹配的位置) match(匹配的对象)
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        # i = [1]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        # 生成需要生成文件的exp后面的数字 n = max(i) + 1 = 2
        n = max(i) + 1 if i else 1  # increment number
        # 返回path runs/train/exp2
        path = Path(f"{path}{sep}{n}{suffix}")  # update path

    # path.suffix文件后缀   path.parent　路径的上级目录  runs/train/exp2
    dir = path if path.suffix == '' else path.parent  # directory
    if not dir.exists() and mkdir:  # mkdir 默认False 先不创建dir
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path  # 返回runs/train/exp2

