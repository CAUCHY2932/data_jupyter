# -*- coding: utf-8 -*-

"""
    作者:     梁斌
    版本:     1.0
    日期:     2017/02/13
    项目名称：电影口碑与海报图像的相关性分析
"""
from bs4 import BeautifulSoup
import urllib.request
import cognitive_face as CF
from skimage import io
import numpy as np


def get_img_link(movie_link):
    """
        通过电影的链接爬取海报的链接
    """
    movie_html = urllib.request.urlopen(movie_link)
    movie_html_obj = BeautifulSoup(movie_html, 'html.parser', from_encoding='utf-8')
    # 获取海报小图的链接
    small_poster_img_link = movie_html_obj.find('div', class_='poster').find('img')['src']

    # 获取海报大图的链接
    big_poster_img_link = small_poster_img_link[:small_poster_img_link.find('._V1_') + 4] + '.jpg'

    return big_poster_img_link


def get_n_face(movie_link):
    """
        通过图像链接获取包含的人脸个数
    """
    print('正在处理链接：', movie_link)
    img_link = get_img_link(movie_link)
    Key = 'xxxxxxxxxx '  # 这里请替换成自己申请的key
    CF.Key.set(Key)
    n_face = -1
    try:
        face_list = CF.face.detect(img_link)
        n_face = len(face_list)
        print('人脸个数：', n_face)
    except CF.util.CognitiveFaceException:
        print('无效图片')
    return n_face


def round_to_int(x, base=10):
    """
        将数字转换到最近的整数
    """
    return int(base * round(float(x)/base))


def get_color_mean(movie_link):
    """
        通过图像链接获取其平均像素值
    """
    print('正在处理链接：', movie_link)
    img_link = get_img_link(movie_link)
    image = io.imread(img_link)
    int_mean_color = round_to_int(np.mean(image))
    print('像素均值：', int_mean_color)
    return int_mean_color
