# -*- coding: utf-8 -*-

"""
    作者:     梁斌
    版本:     1.0
    日期:     2017/03/01
    项目名称：识别Twitter用户性别 (Twitter User Gender Classification)
    Kaggle地址：https://www.kaggle.com/crowdflower/twitter-user-gender-classification
"""
from skimage import io
import os
import re
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import pandas as pd
import math
import numpy as np
from skimage import exposure, img_as_float


# 头像图片保存路径
profile_image_path = './pro_img/'


def inspect_dataset(df_data):
    """pytoho
        查看加载的数据基本信息
    """
    print('数据集基本信息：')
    print(df_data.info())
    print('数据集有{}行，{}列'.format(df_data.shape[0], df_data.shape[1]))
    print('数据预览:')
    print(df_data.head())


def check_profile_image(img_link):
    """
        判断头像图片链接是否有效
        如果有效，下载到本地，并且返回保存路径
    """
    save_image_path = ''
    # 有效的图片扩展名
    valid_img_ext_lst = ['.jpeg', '.png', '.jpg']

    try:
        img_data = io.imread(img_link)
        image_name = img_link.rsplit('/')[-1]
        if any(valid_img_ext in image_name.lower() for valid_img_ext in valid_img_ext_lst):
            # 确保图片文件包含有效的扩展名
            save_image_path = os.path.join(profile_image_path, image_name)
            io.imsave(save_image_path, img_data)
    except:
        print('头像链接 {} 无效'.format(img_link))

    return save_image_path


def clean_text(text):
    """
        清洗文本数据
    """
    # just in case
    text = text.lower()

    # 去除特殊字符
    text = re.sub('\s\W', ' ', text)
    text = re.sub('\W\s', ' ', text)
    text = re.sub('\s+', ' ', text)

    return text


def split_train_test(df_data, size=0.8):
    """
        分割训练集和测试集
    """
    # 为保证每个类中的数据能在训练集中和测试集中的比例相同，所以需要依次对每个类进行处理
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()

    labels = [0, 1]
    for label in labels:
        # 找出gender的记录
        text_df_w_label = df_data[df_data['label'] == label]
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
        df_train = df_train.append(text_df_w_label_train)
        df_test = df_test.append(text_df_w_label_test)

    df_train = df_train.reset_index()
    df_test = df_test.reset_index()
    return df_train, df_test


def get_word_list_from_data(text_s):
    """
        将数据集中的单词放入到一个列表中
    """
    word_list = []
    for _, text in text_s.iteritems():
        word_list += text.split(' ')
    return word_list


def proc_text(text):
    """
        分词+去除停用词
    """
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(text)
    filtered_words = [word for word in words if word not in stopwords.words('english')]
    return " ".join(filtered_words)


def extract_tf_idf(text_s, text_collection, common_words_freqs):
    """
        提取tf-idf特征
    """
    # 这里只选择TF-IDF特征作为例子
    # 可考虑使用词频或其他文本特征作为额外的特征

    n_sample = text_s.shape[0]
    n_feat = len(common_words_freqs)

    common_words = [word for word, _ in common_words_freqs]

    # 初始化
    X = np.zeros([n_sample, n_feat])

    print('提取tf-idf特征...')
    for i, text in text_s.iteritems():
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

    return X


def hex_to_rgb(value):
    """
        十六进制颜色码转换为RGB值
    """
    rgb_list = list(int(value[i:i + 2], 16) for i in range(0, 6, 2))
    return rgb_list


def extract_rgb_feat(hex_color_s):
    """
         从十六进制颜色码中提取RGB值作为特征
    """
    n_sample = hex_color_s.shape[0]
    n_feat = 3

    # 初始化
    X = np.zeros([n_sample, n_feat])

    print('提取RGB特征...')
    for i, hex_val in hex_color_s.iteritems():
        feat_vec = hex_to_rgb(hex_val)

        # 赋值
        X[i, :] = np.array(feat_vec)

    return X


def extract_rgb_hist_feat(img_path_s):
    """
        从图像中提取RGB直方图特征
    """
    n_sample = img_path_s.shape[0]
    n_bins = 100    # 每个通道bin的个数
    n_feat = n_bins * 3

    # 初始化
    X = np.zeros([n_sample, n_feat])

    print('提取RGB直方图特征...')
    for i, img_path in img_path_s.iteritems():
        # 加载图像
        img_data = io.imread(img_path)
        img_data = img_as_float(img_data)

        if img_data.ndim == 3:
            # 3个通道
            hist_r, _ = exposure.histogram(img_data[:, :, 0], nbins=n_bins)
            hist_g, _ = exposure.histogram(img_data[:, :, 1], nbins=n_bins)
            hist_b, _ = exposure.histogram(img_data[:, :, 2], nbins=n_bins)
        else:
            # 2个通道
            hist, _ = exposure.histogram(img_data, nbins=n_bins)
            hist_r = hist.copy()
            hist_g = hist.copy()
            hist_b = hist.copy()

        feat_vec = np.concatenate((hist_r, hist_b, hist_g))

        # 赋值
        X[i, :] = np.array(feat_vec)

    return X
