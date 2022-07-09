import os
import sys
import logging
import argparse

import numpy as np
import openslide
import cv2
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

parser = argparse.ArgumentParser(description='According to the annotation file, the  tissue region was extracted '
                                             'save it in npy format')

parser.add_argument('--level', default=0, type=int, help='at which WSI level'
                    ' to obtain the mask, default 6')


def run(args, slide_path, json_path, output_tumor_path):

    # get the level * dimensions e.g. tumor0.tif level 6 shape (1589, 7514)
    slide = openslide.OpenSlide(slide_path) #
    w, h = slide.level_dimensions[args.level]
    mask_tumor = np.zeros((h, w)) # the init mask, and all the value is 0

    # get the factor of level * e.g. level 6 is 2^6
    factor = slide.level_downsamples[args.level]


    with open(json_path) as f:
        dicts = json.load(f)

    tumor_polygons = dicts['positive'] ##negative

    for tumor_polygon in tumor_polygons:
        # plot a polygon
        name = tumor_polygon["name"]

        vertices = np.array(tumor_polygon["vertices"]) / factor
        vertices = vertices.astype(np.int32)

        cv2.fillPoly(mask_tumor, [vertices], (255))

    mask_tumor = mask_tumor[:] > 127
    mask_tumor = np.transpose(mask_tumor)

    np.save(output_tumor_path, mask_tumor)

def main(slide_path, json_path, output_tumor_path):
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    run(args, slide_path, json_path, output_tumor_path)

if __name__ == "__main__":
    # print(1)
    slide_path_root =r"D:\zhengtingting222\2021-3-15code\CD30\data\IHCdata\xin_6_27\WSI\test\1\wsi"
    json_path_root =r"D:\zhengtingting222\2021-3-15code\CD30\data\IHCdata\xin_6_27\WSI\test\1\json"  #由xml文件生成的json  npy文件
    tumor_mask_path_root =r"D:\zhengtingting222\2021-3-15code\CD30\data\IHCdata\xin_6_27\WSI\test\1\mask\normal\level0" ##存储根据json文件生成的mask文件


    for slide_name in os.listdir(slide_path_root):
        slide_path = os.path.join(slide_path_root, slide_name)
        print("1.slide路径: {}".format(slide_path))
        (file_name, extension) = os.path.splitext(slide_name)
        json_path = os.path.join(json_path_root, file_name + ".json")
        print("2.json文件路径: {}".format(json_path))
        output_tumor_path = os.path.join(tumor_mask_path_root, file_name + ".npy")
        main(slide_path, json_path, output_tumor_path)
        print("3.完成tumor_mask文件保存，保存路径为：{}".format(output_tumor_path))
