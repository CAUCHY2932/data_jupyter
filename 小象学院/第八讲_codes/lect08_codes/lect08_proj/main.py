# -*- coding: utf-8 -*-

"""
    作者:     梁斌
    版本:     1.0
    日期:     2017/03/01
    项目名称：识别Twitter用户性别 (Twitter User Gender Classification)
    Kaggle地址：https://www.kaggle.com/crowdflower/twitter-user-gender-classification
"""
import os
import pandas as pd
from common_tools import get_dataset_filename, unzip, cal_acc
from pd_tools import inspect_dataset, check_profile_image, \
    split_train_test, clean_text, proc_text, get_word_list_from_data, \
    extract_tf_idf, extract_rgb_feat, extract_rgb_hist_feat
import nltk
from nltk.text import TextCollection
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA


# 声明数据集路径
dataset_path = './dataset'  # 数据集路径
zip_filename = 'twitter-user-gender-classification.zip'  # zip文件名
zip_filepath = os.path.join(dataset_path, zip_filename)  # zip文件路径
cln_datapath = './cln_data.csv'     # 清洗好的数据路径

# 是否第一次运行
is_first_run = False


def run_main():
    """
        主函数
    """
    # 声明变量
    dataset_filename = get_dataset_filename(zip_filepath)  # 数据集文件名（在zip中）
    dataset_filepath = os.path.join(dataset_path, dataset_filename)  # 数据集文件路径

    if is_first_run:

        print('解压zip...', end='')
        unzip(zip_filepath, dataset_path)
        print('完成.')

        # 读取数据
        data = pd.read_csv(dataset_filepath, encoding='latin1',
                           usecols=['gender', 'description', 'link_color',
                                    'profileimage', 'sidebar_color', 'text'])
        # 1. 查看加载的数据集
        inspect_dataset(data)

        # 2. 数据清洗
        # 2.1. 根据 'gender' 列过滤数据
        filtered_data = data[(data['gender'] == 'male') | (data['gender'] == 'female')]

        # 2.2 过滤掉 'description' 列为空的数据
        filtered_data = filtered_data.dropna(subset=['description'])

        # 2.3 过滤掉 'link_color' 列和 'sidebar_color' 列非法的16进制数据
        filtered_data = filtered_data[filtered_data['link_color'].str.len() == 6]
        filtered_data = filtered_data[filtered_data['sidebar_color'].str.len() == 6]

        # 2.4 清洗文本数据
        print('清洗文本数据...')
        cln_desc = filtered_data['description'].apply(clean_text)
        cln_text = filtered_data['text'].apply(clean_text)
        filtered_data['cln_desc'] = cln_desc
        filtered_data['cln_text'] = cln_text

        # 2.5 根据profileimage的链接判断头像图片是否有效，
        # 并生成新的列代表头像图片保存的路径
        print('下载头像数据...')
        saved_img_s = filtered_data['profileimage'].apply(check_profile_image)
        filtered_data['saved_image'] = saved_img_s
        # 过滤掉无效的头像数据
        filtered_data = filtered_data[filtered_data['saved_image'] != '']

        # 保存处理好的数据
        filtered_data.to_csv(cln_datapath, index=False)

    # 读取处理好的数据
    clean_data = pd.read_csv(cln_datapath, encoding='latin1',
                             usecols=['gender', 'cln_desc', 'cln_text',
                                      'link_color', 'sidebar_color', 'saved_image'])

    # 查看label的分布
    print(clean_data.groupby('gender').size())

    # 替换male->0, female->1
    clean_data.loc[clean_data['gender'] == 'male', 'label'] = 0
    clean_data.loc[clean_data['gender'] == 'female', 'label'] = 1

    # 3. 分割数据集
    # 分词 去除停用词
    proc_desc_s = clean_data['cln_desc'].apply(proc_text)
    clean_data['desc_words'] = proc_desc_s

    proc_text_s = clean_data['cln_text'].apply(proc_text)
    clean_data['text_words'] = proc_text_s

    df_train, df_test = split_train_test(clean_data)
    # 查看训练集测试集基本信息
    print('训练集中各类的数据个数：', df_train.groupby('label').size())
    print('测试集中各类的数据个数：', df_test.groupby('label').size())

    # 4. 特征工程
    # 4.1 训练数据特征提取
    print('训练样本特征提取：')
    # 4.1.1 文本数据
    # description数据
    print('统计description词频...')
    n_desc_common_words = 50
    desc_words_in_train = get_word_list_from_data(df_train['desc_words'])
    fdisk = nltk.FreqDist(desc_words_in_train)
    desc_common_words_freqs = fdisk.most_common(n_desc_common_words)
    print('descriptino中出现最多的{}个词是：'.format(n_desc_common_words))
    for word, count in desc_common_words_freqs:
        print('{}: {}次'.format(word, count))
    print()

    # 提取desc文本的TF-IDF特征
    print('提取desc文本特征...', end=' ')
    desc_collection = TextCollection(df_train['desc_words'].values.tolist())
    tr_desc_feat = extract_tf_idf(df_train['desc_words'], desc_collection, desc_common_words_freqs)
    print('完成')
    print()

    # text数据
    print('统计text词频...')
    n_text_common_words = 50
    text_words_in_train = get_word_list_from_data(df_train['text_words'])
    fdisk = nltk.FreqDist(text_words_in_train)
    text_common_words_freqs = fdisk.most_common(n_text_common_words)
    print('text中出现最多的{}个词是：'.format(n_text_common_words))
    for word, count in text_common_words_freqs:
        print('{}: {}次'.format(word, count))
    print()

    # 提取text文本TF-IDF特征
    text_collection = TextCollection(df_train['text_words'].values.tolist())
    print('提取text文本特征...', end=' ')
    tr_text_feat = extract_tf_idf(df_train['text_words'], text_collection, text_common_words_freqs)
    print('完成')
    print()

    # 4.1.2 图像数据
    # link color的RGB特征
    tr_link_color_feat_ = extract_rgb_feat(df_train['link_color'])
    tr_sidebar_color_feat = extract_rgb_feat(df_train['sidebar_color'])

    # 头像的RGB直方图特征
    tr_profile_img_hist_feat = extract_rgb_hist_feat(df_train['saved_image'])

    # 组合文本特征和图像特征
    tr_feat = np.hstack((tr_desc_feat, tr_text_feat, tr_link_color_feat_,
                         tr_sidebar_color_feat, tr_profile_img_hist_feat))

    # 特征范围归一化
    scaler = StandardScaler()
    tr_feat_scaled = scaler.fit_transform(tr_feat)

    # 获取训练集标签
    tr_labels = df_train['label'].values

    # 4.2 测试数据特征提取
    print('测试样本特征提取：')
    # 4.2.1 文本数据
    # description数据
    # 提取desc文本的TF-IDF特征
    print('提取desc文本特征...', end=' ')
    te_desc_feat = extract_tf_idf(df_test['desc_words'], desc_collection, desc_common_words_freqs)
    print('完成')
    print()

    # text数据
    # 提取text文本TF-IDF特征
    print('提取text文本特征...', end=' ')
    te_text_feat = extract_tf_idf(df_test['text_words'], text_collection, text_common_words_freqs)
    print('完成')
    print()

    # 4.2.2 图像数据
    # link color的RGB特征
    te_link_color_feat_ = extract_rgb_feat(df_test['link_color'])
    te_sidebar_color_feat = extract_rgb_feat(df_test['sidebar_color'])

    # 头像的RGB直方图特征
    te_profile_img_hist_feat = extract_rgb_hist_feat(df_test['saved_image'])

    # 组合文本特征和图像特征
    te_feat = np.hstack((te_desc_feat, te_text_feat, te_link_color_feat_,
                         te_sidebar_color_feat, te_profile_img_hist_feat))

    # 特征范围归一化
    te_feat_scaled = scaler.transform(te_feat)

    # 获取训练集标签
    te_labels = df_test['label'].values

    # 4.3 PCA降维操作
    pca = PCA(n_components=0.95)  # 保留95%累计贡献率的特征向量
    tr_feat_scaled_pca = pca.fit_transform(tr_feat_scaled)
    te_feat_scaled_pca = pca.transform(te_feat_scaled)

    # 5. 模型建立训练，对比PCA操作前后的效果
    # 使用未进行PCA操作的特征
    lr_model = LogisticRegression()
    lr_model.fit(tr_feat_scaled, tr_labels)

    # 使用PCA操作后的特征
    lr_pca_model = LogisticRegression()
    lr_pca_model.fit(tr_feat_scaled_pca, tr_labels)

    # 6. 模型测试
    pred_labels = lr_model.predict(te_feat_scaled)
    pred_pca_labels = lr_pca_model.predict(te_feat_scaled_pca)
    # 准确率
    print('未进行PCA操作:')
    print('样本维度：', tr_feat_scaled.shape[1])
    print('准确率：{}'.format(cal_acc(te_labels, pred_labels)))

    print()
    print('进行PCA操作后:')
    print('样本维度：', tr_feat_scaled_pca.shape[1])
    print('准确率：{}'.format(cal_acc(te_labels, pred_pca_labels)))

    # 7. 删除解压数据，清理空间
    if os.path.exists(dataset_filepath):
        os.remove(dataset_filepath)


if __name__ == '__main__':
    run_main()
