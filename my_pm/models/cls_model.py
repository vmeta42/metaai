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

    elif use_model == 'random_forest':
        cls_model = RandomForestClassifier()

    elif use_model == 'xgboost':
        cls_model = XGBClassifier(
            learning_rate=0.1,
            gamma=0.1,
            reg_alpha=2,
            reg_lambda=2,
            max_depth=6,
            min_child_weight=6,
            colsample_bytree=0.7,
            colsample_bylevel=0.7,
            verbosity=2)    # xgb.XGBClassifier() XGBoost分类模型

    elif use_model == 'lightgbm':
        cls_model = LGBMClassifier(num_leaves=31, learning_rate=0.1, n_estimators=20)

    else:
        cls_model = GradientBoostingClassifier()


    cls_model.fit(X_train, y_train)

    # 返回预测标签 shape:(n_smaples, )
    pred_labels = cls_model.predict(X_train)

    # 返回预测属于某标签的概率 shape:(n_samples, n_classes)
    pred_prob_labels = cls_model.predict_proba(X_train)

    return cls_model


def test_model(model, X_test, y_test):
    pred_labels = model.predict(X_test)
    pred_report = classification_report(y_test, pred_labels, labels=[0, 1])
    return pred_report












































