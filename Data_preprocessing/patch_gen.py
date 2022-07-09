
import sys
import os
import argparse
import logging
import time
from shutil import copyfile
from multiprocessing import Pool, Value, Lock

import openslide

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

parser = argparse.ArgumentParser(description='Generate patches from a given '
                                 'list of coordinates')
# parser.add_argument('wsi_path', default=None, metavar='WSI_PATH', type=str,
#                     help='Path to the input directory of WSI files')
# parser.add_argument('coords_path', default=None, metavar='COORDS_PATH',
#                     type=str, help='Path to the input list of coordinates')
# parser.add_argument('patch_path', default=None, metavar='PATCH_PATH', type=str,
#                     help='Path to the output directory of patch images')
parser.add_argument("--patch_number", default=500, metavar="PATCH_NUMB", type=int,
                    help="The number of patches extracted from WSI")
parser.add_argument('--patch_size', default=224, type=int, help='patch size, '
                    'default 768')
parser.add_argument('--level', default=5, type=int, help='level for WSI, to '
                    'generate patches, default 0')
parser.add_argument('--num_process', default=5, type=int,
                    help='number of mutli-process, default 5')

count = Value('i', 0)
lock = Lock()


def process(opts):
    i, pid, x_center, y_center, args, slide_path, file_name, patch_path_root = opts
    x = int(int(x_center) - args.patch_size / 2)
    y = int(int(y_center) - args.patch_size / 2)

    ##-----------------------------------------------------------------
    # print('输出样本的中心点',x_center,y_center)
    # print('输出处理之后的样本点',x,y)
    # print('输出args.patch_size的值',args.patch_size)
    # x,y，是0级下的坐标，leavl是扩大倍率


    slide = openslide.OpenSlide(slide_path)
    # print("slide_path", slide_path)
    img = slide.read_region(
        (x, y), args.level,
        (args.patch_size, args.patch_size)).convert('RGB')

    # img.save(os.path.join(patch_path_root, file_name + "-" + str(i) + '.png'))
    img.save(
        os.path.join(patch_path_root, file_name + "-" + str(x_center) + "-" + str(y_center) + "-" + str(i) + '.png'))

    global lock
    global count

    with lock:
        count.value += 1
        if (count.value) % 100 == 0:
            logging.info('{}, {} patches generated...'
                         .format(time.strftime("%Y-%m-%d %H:%M:%S"),
                                 count.value))


def run(args, slide_path, coor_path, file_name, patch_path_root):
    logging.basicConfig(level=logging.INFO)

    if not os.path.exists(patch_path_root):
        os.mkdir(patch_path_root)

    copyfile(coor_path, os.path.join(patch_path_root, 'list.txt'))

    opts_list = []
    infile = open(coor_path)
    for i, line in enumerate(infile):
        pid, x_center, y_center = line.strip('\n').split(',')
        # print('输出样本的中心点', x_center, y_center)
        opts_list.append((i, pid, x_center, y_center, args, slide_path, file_name, patch_path_root)) ##把一个中心样本点的文件中的所有点，取出来放在了opts_list
    infile.close()

    pool = Pool(processes=args.num_process)
    pool.map(process, opts_list)


def main(slide_path, coor_path, file_name, patch_path_root):
    args = parser.parse_args()
    run(args, slide_path, coor_path, file_name, patch_path_root)


if __name__ == '__main__':
    slide_path_root =r"D:\zhengtingting222\data\Askin\debug\WSI\training\melanoma"  ##原本病理图片的名字
    coor_path_root =r"D:\zhengtingting222\data\Askin\debug\WSI_sample_spot\training\level5" ##样本点的路径
    patch_path_root =r"D:\zhengtingting222\data\Askin\debug\WSI_sample_patch\training\level5"  ##c存放patch的路径

    # slide_path_root = r"D:\zhengtingting222\2021-3-15code\CD30\data\IHCdata\xin_6_27\WSI\train\wsi"  ##原本病理图片的名字
    # coor_path_root =r"D:\zhengtingting222\2021-3-15code\CD30\data\IHCdata\xin_6_27\WSI\train\sample\normal\level0"  ##样本点的路径
    # patch_path_root =r"D:\zhengtingting222\2021-3-15code\CD30\data\IHCdata\xin_6_27\WSI\train\patch\normal\level0"  ##c存放patch的路径

    slide_name = '62403.ndpi'
    slide_path = os.path.join(slide_path_root, slide_name)  ##具体到病理图片的路径
    (file_name, extension) = os.path.splitext(slide_name)  ##获取病理图片的名字
    coor_path = os.path.join(coor_path_root, file_name + ".txt")  ##txt文件的名字
    main(slide_path, coor_path, file_name, patch_path_root)

    # for slide_name in os.listdir(slide_path_root):
    #     slide_path = os.path.join(slide_path_root, slide_name) ##具体到病理图片的路径
    #     (file_name, extension) = os.path.splitext(slide_name)  ##获取病理图片的名字
    #     coor_path = os.path.join(coor_path_root, file_name + ".txt")  ##txt文件的名字
    #     main(slide_path, coor_path, file_name, patch_path_root)













# import sys
# import os
# import argparse
# import logging
# import time
# from shutil import copyfile
# from multiprocessing import Pool, Value, Lock
#
# import openslide
#
# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
# #WSI地址（输入病理切片），病理切片的坐标，输出的patch的地址
# parser = argparse.ArgumentParser(description='Generate patches from a given '
#                                  'list of coordinates')
# parser.add_argument('--wsi_path', default='/home/omnisky/qh/pathology/NCRF-master/HE_dataset/data/wsi/val/duizhao/', metavar='WSI_PATH', type=str,
#                     help='Path to the input directory of WSI files')
# parser.add_argument('--coords_path', default='/home/omnisky/qh/pathology/NCRF-master/HE_dataset/data/sample_coords/val/duizhao/list.txt', metavar='COORDS_PATH',
#                     type=str, help='Path to the input list of coordinates')
# parser.add_argument('--patch_path', default='/home/omnisky/qh/pathology/NCRF-master/HE_dataset/data/patch/val/duizhao/', metavar='PATCH_PATH', type=str,
#                     help='Path to the output directory of patch images')
# parser.add_argument('--patch_size', default=768, type=int, help='patch size, '
#                     'default 768')
# parser.add_argument('--level', default=0, type=int, help='level for WSI, to '
#                     'generate patches, default 0')
# parser.add_argument('--num_process', default=5, type=int,
#                     help='number of mutli-process, default 5')
#
# count = Value('i', 0)
# lock = Lock()
#
#
# def process(opts):
#     i, pid, x_center, y_center, args = opts
#     x = int(int(x_center) - args.patch_size / 2)
#     y = int(int(y_center) - args.patch_size / 2)#pid为坐标文档中的每一行的名称
#     wsi_path = os.path.join(args.wsi_path, pid + '.ndpi')
#     slide = openslide.OpenSlide(wsi_path)
#     # print(args.level)
#     img = slide.read_region(
#         (x, y), args.level,
#         (args.patch_size, args.patch_size)).convert('RGB')#将图片转换成RGB模式
#
#     img.save(os.path.join(args.patch_path, str(i) + '.png'))
#
#     global lock
#     global count
#
#     with lock:
#         count.value += 1
#         if (count.value) % 100 == 0:
#             logging.info('{}, {} patches generated...'
#                          .format(time.strftime("%Y-%m-%d %H:%M:%S"),
#                                  count.value))
#
#
# def run(args):
#     logging.basicConfig(level=logging.INFO)
#
#     if not os.path.exists(args.patch_path):
#         os.mkdir(args.patch_path)
#
#    # copyfile(args.coords_path, os.path.join(args.patch_path, 'list.txt'))#数据复制，coords为原目录，后面为目标目录
#
#     opts_list = []
#     infile = open(args.coords_path)
#     #enumerate是输出元素及其元素的序号,pid是肿瘤切片的名字
#     for i, line in enumerate(infile):
#         pid, x_center, y_center = line.strip('\n').split(',')
#         opts_list.append((i, pid, x_center, y_center, args))
#     infile.close()
#
#     pool = Pool(processes=args.num_process)
#     pool.map(process, opts_list)
#
#
# def main():
#     args = parser.parse_args()
#
#     run(args)
#
#
# if __name__ == '__main__':
#     main()
