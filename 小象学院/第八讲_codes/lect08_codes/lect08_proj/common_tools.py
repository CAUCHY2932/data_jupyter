# -*- coding: utf-8 -*-

"""
    作者:     梁斌
    版本:     1.0
    日期:     2017/03/01
    项目名称：识别Twitter用户性别 (Twitter User Gender Classification)
    Kaggle地址：https://www.kaggle.com/crowdflower/twitter-user-gender-classification
"""
import zipfile


def unzip(zip_filepath, dest_path):
    """
        解压zip文件
    """
    with zipfile.ZipFile(zip_filepath) as zf:
        zf.extractall(path=dest_path)


def get_dataset_filename(zip_filepath):
    """
        获取数据库文件名
    """
    with zipfile.ZipFile(zip_filepath) as zf:
        return zf.namelist()[0]


def cal_acc(true_labels, pred_labels):
    """
        计算准确率
    """
    n_total = len(true_labels)
    correct_list = [true_labels[i] == pred_labels[i] for i in range(n_total)]

    acc = sum(correct_list) / n_total
    return acc
