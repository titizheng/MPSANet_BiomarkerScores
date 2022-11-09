import argparse
import sys

import numpy as np
import cv2
import matplotlib.pyplot as plt
import openslide
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from matplotlib.colors import ListedColormap,LinearSegmentedColormap




level=6 ##

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

parser = argparse.ArgumentParser(description='Get tissue mask of WSI and save'
                                 ' it in npy format')

parser.add_argument('--level', default=2, type=int, help='at which WSI level'
                    ' to obtain the mask, default 6')


def breast_thumbnail(slide_path,slide_name,args):
    # args.level
    slide = openslide.OpenSlide(slide_path)
    print(slide.level_dimensions[args.level])  #
    #获取level为5 的宽和高
    slide_shape = slide.level_dimensions[args.level]
    slide_w = slide_shape[0]
    slide_h = slide_shape[1]
    # slide_region = slide.read_region((0, 0), 5, (slide_w - 1, slide_h - 1))
    # slide_npy = np.array(slide_region)

    thumbnail = slide.get_thumbnail((slide_w, slide_h))  #

    thumbnail_npy = np.array(thumbnail)

    plt.figure(figsize=(slide_w / 100, slide_h / 100), dpi=100)
    plt.imshow(thumbnail_npy)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)


    # plt.savefig(r"./{}.png".format(slide_name))
    print("3.Successfully saved thumbnail named {}".format(slide_name))
    plt.show()
    return slide_w, slide_h

def get_prob_map(npy_path, slide_name, slide_w, slide_h):
    image = np.load(npy_path) ##本来是HXW
    image = np.transpose(image, [1, 0]) #要转化为WXH
    print(image.shape)



    '''
    '#FFFFFF',白色'#F0F8FF', '#ADD8E6', '#98FB98', '#90EE90', '#ADFF2F',
                                                         '#ADFF2F',"#FFFF00",'#FFA500','#FF4500',"#DC143C"
    '''
    def colormap():
        return LinearSegmentedColormap.from_list('cmap',
                                                 ['#FFFFFF', '#000080', '#0000CD', '#0000FF', '#6495ED',
                                                  '#00CED1', '#00FFFF',
                                                  '#00FF7F', "#98FB98", '#FFFF00', '#FFD700', "#FFA500",
                                                  "#FF8C00", "#FF7F50", "#FF0000",
                                                  "#DC143C", '#B22222', '#802A2A'
                                                  ], 256)


    plt.figure(figsize=(slide_w / 100, slide_h / 100), dpi=100)
    plt.imshow(image, cmap=colormap(),interpolation='bicubic')#colormap()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    # plt.colorbar()
    plt.savefig(r"/home/omnisky/hdd_15T_sdc/zttdata/TCGA/TCGAHEHER2/{}.png".format(slide_name))
    # plt.savefig(r"/{}.png".format(slide_name))




    print("4.成功保存文件名为{}的概率图".format(slide_name))
    plt.show()


slide_path_root =r"/home/omnisky/hdd_15T_sdc/zttdata/TCGA/TCGAHEHER2/1slide/Her2Pos"
npy_path_root = r"/home/omnisky/hdd_15T_sdc/zttdata/TCGA/TCGAHEHER2/4masknpy/level0/Her2Pos"

args = parser.parse_args()
for slide in os.listdir(slide_path_root):
    slide_path = os.path.join(slide_path_root, slide)
    (slide_name, extension) = os.path.splitext(slide)
    # npy_path = os.path.join(npy_path_root, slide_name +'_'+str(args.level)+ ".npy")

    npy_path = os.path.join(npy_path_root, slide_name + ".npy")
    print("1.Obtained slice named {} successfully".format(slide_name))
    print("2.slide_path:",  slide_path, "npy_path:", npy_path)
    slide_w, slide_h = breast_thumbnail(slide_path, slide_name,args)
    get_prob_map(npy_path, slide_name, slide_w, slide_h)


