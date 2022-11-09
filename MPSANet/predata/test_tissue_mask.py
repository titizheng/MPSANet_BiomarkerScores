import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import argparse
import logging

import numpy as np
import openslide
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu


sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

parser = argparse.ArgumentParser(description='Get tissue mask of WSI and save'
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
    print("2.save {}".format(npy_path))


def main(wsi_path, npy_path):
    args = parser.parse_args()
    run(args, wsi_path, npy_path)


if __name__ == '__main__':
    wsi_path_root = "/home/omnisky/sata_16tb1/zttdata/IHC/KI-67/slides"
    npy_path_root = "/home/omnisky/sata_16tb1/zttdata/IHC/KI-67/mask/level3" #
    for wsi_name in os.listdir(wsi_path_root):
        wsi_path = os.path.join(wsi_path_root, wsi_name)
        print("1.slide path{}".format(wsi_path))
        (file_name, extension) = os.path.splitext(wsi_name)
        npy_path = os.path.join(npy_path_root, file_name + ".npy")
        main(wsi_path, npy_path)


