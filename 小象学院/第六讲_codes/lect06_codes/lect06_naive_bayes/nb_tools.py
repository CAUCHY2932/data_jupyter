# -*- coding: utf-8 -*-

"""
    作者:     梁斌
    版本:     1.0
    日期:     2017/02/13
    项目名称： 朴素贝叶斯 模型的手工实现
    参考：    http://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
"""
import numpy as np
import scipy.stats


class NaiveBayes(object):
    """
        Naive Bayes 类
    """
    def __init__(self, n_feat):
        self.n_feat = n_feat    # 样本的特征维度

    def get_cls_stats(self, X_train, y_train):
        """
            返回训练集中每个类的统计参数
        """
        # 获取类别
        unique_cls_list = list(set(y_train.tolist()))

        # 构造字典
        dataset_stats = {}
        for cls in unique_cls_list:
            # 获取属于该类的样本
            samples_in_cls = X_train[y_train == cls]

            # 获取该类样本中每个特征的统计参数
            samples_in_cls_stats = self.get_samples_stats(samples_in_cls)
            dataset_stats[cls] = samples_in_cls_stats

        return dataset_stats

    def get_samples_stats(self, samples):
        """
            返回一组样本中每个特征（属性）的统计参数
        """
        # 每个特征维度上计算统计参数，即均值和标准差
        samples_stats = [(np.mean(samples[:, i]), np.std(samples[:, i]))
                         for i in range(self.n_feat)]
        return samples_stats

    def predict(self, tr_cls_stats, X_test):
        """
            根据训练样本统计参数预测整个测试样本集
        """
        y_pred = []
        n_sample = X_test.shape[0]  # 测试样本的个数
        for i in range(n_sample):
            # 遍历每个测试样本
            sample = X_test[i, :]
            pred = self.predict_sample(tr_cls_stats, sample)
            y_pred.append(pred)

        return y_pred

    def predict_sample(self, tr_cls_stats, sample):
        """
            根据训练样本统计参数预测单一样本
        """
        cls_probs = self.cal_cls_probs(tr_cls_stats, sample)

        # 初始化
        best_label = None
        best_prob = -1

        for cls, cls_prob in cls_probs.items():
            if best_label is None or cls_prob > best_prob:
                best_prob = cls_prob
                best_label = cls

        return best_label

    def cal_cls_probs(self, tr_cls_stats, sample):
        """
            根据高斯分布及训练集的统计参数返回样本分类的概率
        """
        probs = {}
        for cls, cls_stats in tr_cls_stats.items():
            # 初始化属于cls类的概率
            probs[cls] = 1
            for i in range(len(cls_stats)):
                # 遍历cls类中每个特征维度上的统计参数
                mean, std = cls_stats[i]
                single_feat_vec = sample[i]

                # 根据高斯分布的概率密度函数及每个特征维度上的统计参数求联合该概率
                probs[cls] *= scipy.stats.norm.pdf(single_feat_vec, mean, std)

        return probs


def cal_acc(true_labels, pred_labels):
    """
        计算准确率
    """
    n_total = len(true_labels)
    correct_list = [true_labels[i] == pred_labels[i] for i in range(n_total)]

    acc = sum(correct_list) / n_total
    return acc
