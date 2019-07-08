# -*- coding: utf-8 -*-

"""
    作者:     梁斌
    版本:     1.0
    日期:     2017/03/02
    项目名称：通过移动设备行为数据预测使用者的性别和年龄 (TalkingData Mobile User Demographics)
    Kaggle地址：https://www.kaggle.com/c/talkingdata-mobile-user-demographics
"""
from sklearn.model_selection import GridSearchCV


def get_best_model(model, X_train, y_train, params, cv=5):
    """
        交叉验证获取最优模型
        默认5折交叉验证
    """
    clf = GridSearchCV(model, params, cv=cv)
    clf.fit(X_train, y_train)
    return clf.best_estimator_
