import argparse
import sys

import numpy as np
import cv2
import matplotlib.pyplot as plt
import openslide
import os


from matplotlib.colors import ListedColormap,LinearSegmentedColormap

##把slice转化为png图片


level=6 ##要根据mask的倍率进行调节

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

parser = argparse.ArgumentParser(description='Get tissue mask of WSI and save'
                                 ' it in npy format')

parser.add_argument('--level', default=6, type=int, help='at which WSI level'
                    ' to obtain the mask, default 6')

#获取切片的缩略图
def breast_thumbnail(slide_path,slide_name,args):
    # args.level
    slide = openslide.OpenSlide(slide_path)
    print(slide.level_dimensions[args.level])  # 输出为（w,h），但是获取的npy文件是（h, w），所以需要先对数据做转置运算
    #获取level为5 的宽和高
    slide_shape = slide.level_dimensions[args.level]
    slide_w = slide_shape[0]
    slide_h = slide_shape[1]
    # slide_region = slide.read_region((0, 0), 5, (slide_w - 1, slide_h - 1))
    # slide_npy = np.array(slide_region)

    thumbnail = slide.get_thumbnail((slide_w, slide_h))  # 获取切片的缩略图
    #将thumbnail文件转换为numpy文件
    thumbnail_npy = np.array(thumbnail)

    #去除白边并且获取指定分辨率的文件
    plt.figure(figsize=(slide_w / 100, slide_h / 100), dpi=100)
    plt.imshow(thumbnail_npy)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    # plt.savefig(r"D:\zhengtingting222\2021-3-15code\CD30\data\heatimage\heat-imags\yuan_tu/{}.png".format(slide_name))

    plt.savefig(r"D:\zhengtingting222\2021-3-15code\CD30\data\heatimage\shiyan\quan_png\/{}.png".format(slide_name))
    print("3.成功保存名为{}的缩略图".format(slide_name))
    plt.show()
    return slide_w, slide_h


slide_path_root =r"D:\zhengtingting222\2021-3-15code\CD30\data\heatimage\wsi"


args = parser.parse_args()
for slide in os.listdir(slide_path_root):
    slide_path = os.path.join(slide_path_root, slide)
    (slide_name, extension) = os.path.splitext(slide)
    slide_w, slide_h = breast_thumbnail(slide_path, slide_name,args)




