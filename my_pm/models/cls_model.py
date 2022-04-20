# -*- coding: utf-8 -*-
'''
@Time    : 4/6/2022 11:10 AM
@Author  : dong.yachao
'''

'''
各种机器学习分类模型，如随机森林、GBDT、逻辑回归等
'''
# 决策树
from sklearn.tree import DecisionTreeClassifier
# 随机森林
from sklearn.ensemble import RandomForestClassifier
# GBDT 分类模型
from sklearn.ensemble import GradientBoostingClassifier
# XGBoost
import xgboost as xgb     # 原生xgboost库
from xgboost.sklearn import XGBClassifier   # sklearn xgboost
# LightGBM
import lightgbm as lgb   # 原生lightgbm库
from lightgbm.sklearn import LGBMClassifier   # sklearn lightgbm

import yaml
# 保存模型
import joblib

# 评估指标
# precision (P)
from sklearn.metrics import precision_score
# recall (R)
from sklearn.metrics import recall_score
# F1 Score
from sklearn.metrics import f1_score
# 分类报告
from sklearn.metrics import classification_report



def train_model(use_model, X_train, y_train, args):
    '''

    Args:
        use_model:  使用的分类模型名字, eg."gbdt", "random_forest"
        X_train: 经过数据清洗以及特征工程 具有14维特征的训练数据。
        y_train: 对应的标签（自动标注生成的，0,1）
        cfg: 使用模型对应的参数文件

    Returns: 训练好的分类模型

    '''

    if use_model == 'gbdt':
        cls_model = GradientBoostingClassifier()
        cls_model.fit(X_train, y_train)

    elif use_model == 'random_forest':
        cls_model = RandomForestClassifier()
        cls_model.fit(X_train, y_train)

    elif use_model == 'xgboost':
        # 这里使用原生的xgboost库，可以选择是否使用gpu加速
        # 构建DMatrix
        dtrain = xgb.DMatrix(X_train, y_train)
        # 参数设置
        with open(args.xgboost_params, 'r', encoding='utf-8') as doc:
            xg_cfg = yaml.load(doc, Loader=yaml.Loader)
            params = xg_cfg['params']

        # params = {
        #     'tree_method': 'hist',  # if gpu, set 'gpu_hist'
        #     'booster': 'gbtree',
        #     'objective': 'multi:softmax',
        #     'num_class': 2,
        #     'max_depth': 6,
        #     'eval_metric': 'merror',
        #     'eta': 0.01,
        #     # 'gpi_id': 0
        # }
        # 训练
        cls_model = xgb.train(params, dtrain, num_boost_round=xg_cfg['num_boost_round'])

    elif use_model == 'lightgbm':
        cls_model = LGBMClassifier(num_leaves=31, learning_rate=0.1, n_estimators=20)
        cls_model.fit(X_train, y_train)

    else:
        cls_model = GradientBoostingClassifier()
        cls_model.fit(X_train, y_train)

    return cls_model


def test_model(model, X_test, y_test, use_xg=True):
    if use_xg:
        X_test = xgb.DMatrix(X_test)

    pred_labels = model.predict(X_test)
    pred_report = classification_report(y_test, pred_labels, labels=[0, 1])
    return pred_report












































