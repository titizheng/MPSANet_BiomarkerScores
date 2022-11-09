
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import argparse
import logging
import time
from shutil import copyfile
from multiprocessing import Pool, Value, Lock

import openslide

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

parser = argparse.ArgumentParser(description='Generate patches from a given '
                                 'list of coordinates')
parser.add_argument('--patch_size', default=768, type=int, help='patch size, '
                    'default 768')
parser.add_argument('--level', default=0, type=int, help='level for WSI, to '
                    'generate patches, default 0')
parser.add_argument('--num_process', default=5, type=int,
                    help='number of mutli-process, default 5')

count = Value('i', 0)
lock = Lock()


def process(opts):
    i, pid, x_center, y_center, args, slide_path, file_name, patch_path_root = opts
    x = int(int(x_center) - args.patch_size / 2)
    y = int(int(y_center) - args.patch_size / 2)
    # x = int(x_center)
    # y = int(y_center)



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
        pid, x_center, y_center= line.strip('\n').split(',')
        # print('输出样本的中心点', x_center, y_center)
        idpng = i
        opts_list.append((i, pid, x_center, y_center, args, slide_path, file_name, patch_path_root))
    infile.close()

    pool = Pool(processes=args.num_process)
    pool.map(process, opts_list)


def main(slide_path, coor_path, file_name, patch_path_root):
    args = parser.parse_args()
    run(args, slide_path, coor_path, file_name, patch_path_root)


if __name__ == '__main__':
    slide_path_root = "/home/omnisky/hdd_15T_sdc/zttdata/TCGA/TCGAHEHER2/1slide/Test/Her2Pos"  ##原本病理图片的名字  Pos
    coor_path_root = "/home/omnisky/hdd_15T_sdc/zttdata/TCGA/TCGAHEHER2/5samplespot/Test/Her2Pos"  ##样本点的路径txt
    patch_path_root = "/home/omnisky/hdd_15T_sdc/zttdata/TCGA/TCGAHEHER2/6patchimage/Test/Her2Pos"  ##c存放patch的路径



    for slide_name in os.listdir(slide_path_root):
        slide_path = os.path.join(slide_path_root, slide_name) ##具体到病理图片的路径
        (file_name, extension) = os.path.splitext(slide_name)  ##获取病理图片的名字
        coor_path = os.path.join(coor_path_root, file_name + ".txt")  ##txt文件的名字
        main(slide_path, coor_path, file_name, patch_path_root)
















