# -*- coding: utf-8 -*-

"""
    作者:     梁斌
    版本:     1.0
    日期:     2017/02/13
    项目名称：电影口碑与海报图像的相关性分析
"""
import pandas as pd
from tools import get_n_face, get_color_mean
import matplotlib.pyplot as plt

# 数据集路径
dataset_path = './dataset/movie_metadata.csv'

# 是否第一次运行
is_first_run = False


def run_main():
    """
        主函数
    """
    if is_first_run:
        # 第一次运行程序
        data_df = pd.read_csv(dataset_path, nrows=100)

        # 记录海报中人脸个数
        print('海报人脸检测：')
        data_df['n_face'] = data_df['movie_imdb_link'].apply(get_n_face)

        # 保存结果
        data_df.to_csv('./imdb_face.csv', index=False, encoding='utf-8')

        # 记录海报的平均像素值（这里代表图像的主要颜色，可通过颜色的表达方式替代）
        print('海报像素均值计算：')
        data_df['color_mean'] = data_df['movie_imdb_link'].apply(get_color_mean)

        # 保存结果
        data_df.to_csv('./imdb_face_color.csv', index=False, encoding='utf-8')

    data_df = pd.read_csv('./imdb_face_color.csv')
    # 分析结果
    # 人脸个数和评分的关系
    # 去除无效人脸
    data_df = data_df[data_df['n_face'] != -1]
    face_score = data_df['imdb_score'].groupby(data_df['n_face']).mean()
    face_score.name = 'Score'
    face_score.index.name = 'Number of Face'
    face_score.plot(kind='bar')
    plt.show()

    # 像素均值和评分的关系
    color_score = data_df['imdb_score'].groupby(data_df['color_mean']).mean()
    color_score.name = 'Score'
    color_score.index.name = 'Mean of color'
    color_score.plot(kind='bar')
    plt.show()


if __name__ == '__main__':
    run_main()
