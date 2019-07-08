# -*- coding: utf-8 -*-

"""
    作者:     梁斌
    版本:     1.0
    日期:     2017/02/13
    项目名称： 微博情感分析
"""
import os
import pandas as pd
import nltk
from tools import proc_text, split_train_test, get_word_list_from_data, \
    extract_feat_from_data, cal_acc
from nltk.text import TextCollection
from sklearn.naive_bayes import GaussianNB

dataset_path = './dataset'
text_filenames = ['0_simplifyweibo.txt', '1_simplifyweibo.txt',
                  '2_simplifyweibo.txt', '3_simplifyweibo.txt']

# 原始数据的csv文件
output_text_filename = 'raw_weibo_text.csv'

# 清洗好的文本数据文件
output_cln_text_filename = 'clean_weibo_text.csv'

# 处理和清洗文本数据的时间较长，通过设置is_first_run进行配置
# 如果是第一次运行需要对原始文本数据进行处理和清洗，需要设为True
# 如果之前已经处理了文本数据，并已经保存了清洗好的文本数据，设为False即可
is_first_run = True


def read_and_save_to_csv():
    """
        读取原始文本数据，将标签和文本数据保存成csv
    """

    text_w_label_df_lst = []
    for text_filename in text_filenames:
        text_file = os.path.join(dataset_path, text_filename)

        # 获取标签，即0, 1, 2, 3
        label = int(text_filename[0])

        # 读取文本文件
        with open(text_file, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()

        labels = [label] * len(lines)

        text_series = pd.Series(lines)
        label_series = pd.Series(labels)

        # 构造dataframe
        text_w_label_df = pd.concat([label_series, text_series], axis=1)
        text_w_label_df_lst.append(text_w_label_df)

    result_df = pd.concat(text_w_label_df_lst, axis=0)

    # 保存成csv文件
    result_df.columns = ['label', 'text']
    result_df.to_csv(os.path.join(dataset_path, output_text_filename),
                     index=None, encoding='utf-8')


def run_main():
    """
        主函数
    """
    # 1. 数据读取，处理，清洗，准备
    if is_first_run:
        print('处理清洗文本数据中...', end=' ')
        # 如果是第一次运行需要对原始文本数据进行处理和清洗

        # 读取原始文本数据，将标签和文本数据保存成csv
        read_and_save_to_csv()

        # 读取处理好的csv文件，构造数据集
        text_df = pd.read_csv(os.path.join(dataset_path, output_text_filename),
                              encoding='utf-8')

        # 处理文本数据
        text_df['text'] = text_df['text'].apply(proc_text)

        # 过滤空字符串
        text_df = text_df[text_df['text'] != '']

        # 保存处理好的文本数据
        text_df.to_csv(os.path.join(dataset_path, output_cln_text_filename),
                       index=None, encoding='utf-8')
        print('完成，并保存结果。')

    # 2. 分割训练集、测试集
    print('加载处理好的文本数据')
    clean_text_df = pd.read_csv(os.path.join(dataset_path, output_cln_text_filename),
                                encoding='utf-8')
    # 分割训练集和测试集
    train_text_df, test_text_df = split_train_test(clean_text_df)
    # 查看训练集测试集基本信息
    print('训练集中各类的数据个数：', train_text_df.groupby('label').size())
    print('测试集中各类的数据个数：', test_text_df.groupby('label').size())

    # 3. 特征提取
    # 计算词频
    n_common_words = 200

    # 将训练集中的单词拿出来统计词频
    print('统计词频...')
    all_words_in_train = get_word_list_from_data(train_text_df)
    fdisk = nltk.FreqDist(all_words_in_train)
    common_words_freqs = fdisk.most_common(n_common_words)
    print('出现最多的{}个词是：'.format(n_common_words))
    for word, count in common_words_freqs:
        print('{}: {}次'.format(word, count))
    print()

    # 在训练集上提取特征
    text_collection = TextCollection(train_text_df['text'].values.tolist())
    print('训练样本提取特征...', end=' ')
    train_X, train_y = extract_feat_from_data(train_text_df, text_collection, common_words_freqs)
    print('完成')
    print()

    print('测试样本提取特征...', end=' ')
    test_X, test_y = extract_feat_from_data(test_text_df, text_collection, common_words_freqs)
    print('完成')

    # 4. 训练模型Naive Bayes
    print('训练模型...', end=' ')
    gnb = GaussianNB()
    gnb.fit(train_X, train_y)
    print('完成')
    print()

    # 5. 预测
    print('测试模型...', end=' ')
    test_pred = gnb.predict(test_X)
    print('完成')

    # 输出准确率
    print('准确率：', cal_acc(test_y, test_pred))

if __name__ == '__main__':
    run_main()
