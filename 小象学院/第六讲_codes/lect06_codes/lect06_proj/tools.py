# -*- coding: utf-8 -*-

"""
    作者:     梁斌
    版本:     1.0
    日期:     2017/02/13
    项目名称： 微博情感分析
"""
import re
import jieba.posseg as pseg
import pandas as pd
import math
import numpy as np

# 加载常用停用词
stopwords1 = [line.rstrip() for line in open('./中文停用词库.txt', 'r', encoding='utf-8')]
# stopwords2 = [line.rstrip() for line in open('./哈工大停用词表.txt', 'r', encoding='utf-8')]
# stopwords3 = [line.rstrip() for line in open('./四川大学机器智能实验室停用词库.txt', 'r', encoding='utf-8')]
# stopwords = stopwords1 + stopwords2 + stopwords3
stopwords = stopwords1


def proc_text(raw_line):
    """
        处理每行的文本数据
        返回分词结果
    """
    # 1. 使用正则表达式去除非中文字符
    filter_pattern = re.compile('[^\u4E00-\u9FD5]+')
    chinese_only = filter_pattern.sub('', raw_line)

    # 2. 结巴分词+词性标注
    words_lst = pseg.cut(chinese_only)

    # 3. 去除停用词
    meaninful_words = []
    for word, flag in words_lst:
        # if (word not in stopwords) and (flag == 'v'):
            # 也可根据词性去除非动词等
        if word not in stopwords:
            meaninful_words.append(word)

    return ' '.join(meaninful_words)


def split_train_test(text_df, size=0.8):
    """
        分割训练集和测试集
    """
    # 为保证每个类中的数据能在训练集中和测试集中的比例相同，所以需要依次对每个类进行处理
    train_text_df = pd.DataFrame()
    test_text_df = pd.DataFrame()

    labels = [0, 1, 2, 3]
    for label in labels:
        # 找出label的记录
        text_df_w_label = text_df[text_df['label'] == label]
        # 重新设置索引，保证每个类的记录是从0开始索引，方便之后的拆分
        text_df_w_label = text_df_w_label.reset_index()

        # 默认按80%训练集，20%测试集分割
        # 这里为了简化操作，取前80%放到训练集中，后20%放到测试集中
        # 当然也可以随机拆分80%，20%（尝试实现下DataFrame中的随机拆分）

        # 该类数据的行数
        n_lines = text_df_w_label.shape[0]
        split_line_no = math.floor(n_lines * size)
        text_df_w_label_train = text_df_w_label.iloc[:split_line_no, :]
        text_df_w_label_test = text_df_w_label.iloc[split_line_no:, :]

        # 放入整体训练集，测试集中
        train_text_df = train_text_df.append(text_df_w_label_train)
        test_text_df = test_text_df.append(text_df_w_label_test)

    train_text_df = train_text_df.reset_index()
    test_text_df = test_text_df.reset_index()
    return train_text_df, test_text_df


def get_word_list_from_data(text_df):
    """
        将数据集中的单词放入到一个列表中
    """
    word_list = []
    for _, r_data in text_df.iterrows():
        word_list += r_data['text'].split(' ')
    return word_list


def extract_feat_from_data(text_df, text_collection, common_words_freqs):
    """
        特征提取
    """
    # 这里只选择TF-IDF特征作为例子
    # 可考虑使用词频或其他文本特征作为额外的特征

    n_sample = text_df.shape[0]
    n_feat = len(common_words_freqs)
    common_words = [word for word, _ in common_words_freqs]

    # 初始化
    X = np.zeros([n_sample, n_feat])
    y = np.zeros(n_sample)

    print('提取特征...')
    for i, r_data in text_df.iterrows():
        if (i + 1) % 5000 == 0:
            print('已完成{}个样本的特征提取'.format(i + 1))

        text = r_data['text']

        feat_vec = []
        for word in common_words:
            if word in text:
                # 如果在高频词中，计算TF-IDF值
                tf_idf_val = text_collection.tf_idf(word, text)
            else:
                tf_idf_val = 0

            feat_vec.append(tf_idf_val)

        # 赋值
        X[i, :] = np.array(feat_vec)
        y[i] = int(r_data['label'])

    return X, y


def cal_acc(true_labels, pred_labels):
    """
        计算准确率
    """
    n_total = len(true_labels)
    correct_list = [true_labels[i] == pred_labels[i] for i in range(n_total)]

    acc = sum(correct_list) / n_total
    return acc

