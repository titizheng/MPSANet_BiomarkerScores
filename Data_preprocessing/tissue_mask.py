import sys
import os
import argparse
import logging

import numpy as np
import openslide
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu


''''
这个提取mask是有问题的，因为此免疫组化的片子，背景和组织差距不大，
倒置用这个提取的mask都是有病的黄色的点点。
用这个提取的mask都是用病的区域，
所以预测的结果也都是有病的，这也可以间接的说明，
模型是准确的，能准确识别有病的区域

这个可以用来自动提取病变区域
'''



sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

parser = argparse.ArgumentParser(description='Extract the slide non-background tissue area'
                                 ' it in npy format')

parser.add_argument('--level', default=2, type=int, help='at which WSI level'
                    ' to obtain the mask, default 6')
parser.add_argument('--RGB_min', default=50, type=int, help='min value for RGB'
                    ' channel, default 50')


def run(args, wsi_path, npy_path):
    logging.basicConfig(level=logging.INFO)

    slide = openslide.OpenSlide(wsi_path)

    # note the shape of img_RGB is the transpose of slide.level_dimensions
    img_RGB = np.transpose(np.array(slide.read_region((0, 0),
                           args.level,
                           slide.level_dimensions[args.level]).convert('RGB')),
                           axes=[1, 0, 2])

    img_HSV = rgb2hsv(img_RGB) ##把RGB转化为HSV

    background_R = img_RGB[:, :, 0] > threshold_otsu(img_RGB[:, :, 0])
    background_G = img_RGB[:, :, 1] > threshold_otsu(img_RGB[:, :, 1])
    background_B = img_RGB[:, :, 2] > threshold_otsu(img_RGB[:, :, 2])
    tissue_RGB = np.logical_not(background_R & background_G & background_B)
    tissue_S = img_HSV[:, :, 1] > threshold_otsu(img_HSV[:, :, 1])
    min_R = img_RGB[:, :, 0] > args.RGB_min
    min_G = img_RGB[:, :, 1] > args.RGB_min
    min_B = img_RGB[:, :, 2] > args.RGB_min

    tissue_mask = tissue_S & tissue_RGB & min_R & min_G & min_B

    np.save(npy_path, tissue_mask)
    print("2.成功保存文件{}".format(npy_path))


def main(wsi_path, npy_path):
    args = parser.parse_args()
    run(args, wsi_path, npy_path)


if __name__ == '__main__':
    wsi_path_root = r"D:\zhengtingting222\2021-3-15code\CD30\data\heatimage\wsi"
    npy_path_root = r"D:\zhengtingting222\2021-3-15code\CD30\data\heatimage\maks"
    for wsi_name in os.listdir(wsi_path_root):
        wsi_path = os.path.join(wsi_path_root, wsi_name)
        print("1.成功读取wsi文件{}".format(wsi_path))
        (file_name, extension) = os.path.splitext(wsi_name)
        npy_path = os.path.join(npy_path_root, file_name + ".npy")
        main(wsi_path, npy_path)


