import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import argparse
import logging
import json
import time


import numpy as np
import torch
from torch.utils.data import DataLoader

from model import MODELS_resnet

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from data.wsi_producer import GridWSIPatchDataset  # noqa
from model import MODELS  # noqa


parser = argparse.ArgumentParser(description='Get the probability map of tumor'
                                 ' patch predictions given a WSI')

parser.add_argument('--ckpt_path', default=r'/home/omnisky/hdd_15T_sdc/zttdata/TCGA/breastHer2/HER2_status/kf_save_model/train_train (1).ckpt', metavar='', type=str,
                    help='Path to the saved ckpt file of a pytorch model')

###-------------------------------------验证用的配置文件-------------------------
parser.add_argument('--cfg_path', default=r'/home/omnisky/hdd_15T_sdc/zttcode/cd-30NCRF-KF2/cd-30NCRF-KF/configs/parameters_data_test.json', metavar='CFG_PATH', type=str,
                    help='Path to the config file in json format related to'
                    ' the ckpt file')

parser.add_argument('--GPU', default='1', type=str, help='which GPU to use'
                    ', default 0')
parser.add_argument('--num_workers', default=0, type=int, help='number of '
                    'workers to use to make batch, default 5')
parser.add_argument('--eight_avg', default=0, type=int, help='if using average'
                    ' of the 8 direction predictions for each patch,'
                    ' default 0, which means disabled')
parser.add_argument('--level', default=0, type=int, help='at which WSI level'
                    ' to obtain the mask, default 6')

##这个是生成概率图
def get_probs_map(model, dataloader,cfg):
    probs_map = np.zeros(dataloader.dataset._mask.shape)  ##这个图的大小和mask的npy文件大小一样
    num_batch = len(dataloader) ##
    # only use the prediction of the center patch within the grid
    idx_center = dataloader.dataset._grid_size // 2

    count = 0
    P_count=0
    F_count=0
    probs_gl=[] ##单纯的存概率值
    time_now = time.time()
    for (data, x_mask, y_mask) in dataloader:
        with torch.no_grad():
            data = torch.autograd.Variable(data.cuda()) #, volatile=True,async=True,
        output = model(data)

        if len(output.shape) == 1:
            probs = output[idx_center].sigmoid().cpu().data.numpy().flatten()
        else:
            probs = output[:,
                           idx_center].sigmoid().cpu().data.numpy().flatten()

        probs_map[x_mask, y_mask] = probs
        probs_gl.append(probs)

        for i in probs:
            if i>0.5: ##这个应该改成i>=0.5
                P_count+=1
            else:
                F_count+=1
        count += 1

        time_spent = time.time() - time_now
        time_now = time.time()
        logging.info(
            '{}, flip : {}, rotate : {}, batch : {}/{}, Run Time : {:.2f}'
            .format(
                time.strftime("%Y-%m-%d %H:%M:%S"), dataloader.dataset._flip,
                dataloader.dataset._rotate, count, num_batch, time_spent))

    PP=P_count/(count*int(cfg['batch_size']))
    FF=F_count/(count*int(cfg['batch_size'])) ##
    print('输出P_count',P_count)
    print('输出F_count', F_count)
    print('输出阳性的占比：',PP)
    print('输出阴性的占比：', FF)

    PP = P_count / count

    FF = F_count / count  ##

    print('不除batch输出阳性的占比：', PP)
    print('不除batch输出阴性的占比：', FF)

    return probs_map,count,probs_gl

##加载数据，为了训练模型
def make_dataloader(args, wsi_path, mask_path, cfg, flip='NONE', rotate='NONE'):
    batch_size = cfg['batch_size']
    num_workers = args.num_workers
    ##-------------------------------------------这个里面并没有真实的标签
    dataloader = DataLoader(
        GridWSIPatchDataset(wsi_path, mask_path,
                            image_size=cfg['image_size'],
                            patch_size=cfg['patch_size'],
                            crop_size=cfg['crop_size'], normalize=True,
                            flip=flip, rotate=rotate),
        batch_size=batch_size, num_workers=num_workers, drop_last=False) ##这个batch_size是设置，一次输入进去模型的有几位

    return dataloader


def run(args, wsi_path, mask_path, prob_map_root):

    global probs_gl, probs_map, count

    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    logging.basicConfig(level=logging.INFO)

    with open(args.cfg_path) as f:
        cfg = json.load(f)

    if cfg['image_size'] % cfg['patch_size'] != 0:
            raise Exception('Image size / patch size != 0 : {} / {}'.
                            format(cfg['image_size'], cfg['patch_size']))

    patch_per_side = cfg['image_size'] // cfg['patch_size']
    grid_size = patch_per_side * patch_per_side

    mask = np.load(mask_path)

    ##---------------------------------------------模型加载----------------------------
    ckpt = torch.load(args.ckpt_path,map_location='cpu')
    model = MODELS_resnet[cfg['model']](num_nodes=grid_size, use_crf=cfg['use_crf'])
    model.load_state_dict(ckpt['state_dict'])
    model = model.cuda().eval()
    ##---------------------------------------------数据处理----------------------------
    if not args.eight_avg:
        dataloader = make_dataloader(args, wsi_path, mask_path, cfg, flip='NONE', rotate='NONE') ##改变数据输入格式，4个进行的
        probs_map,count,probs_gl = get_probs_map(model, dataloader,cfg)
        print('禁用下面的操作')


    prob_map = os.path.join(prob_map_root, file_name+".npy") #'_'+str(args.level)+
    np.save(prob_map, probs_map)


    print("3.成功输出概率文件{}".format(prob_map))



def main(wsi_path, mask_path, prob_map_root):
    args = parser.parse_args()
    # run(args, wsi_path, mask_path, prob_map)
    run(args, wsi_path, mask_path, prob_map_root)


if __name__ == '__main__':

    wsi_path_root =r"/home/omnisky/hdd_15T_sdc/zttdata/TCGA/breastHer2/HER2_status/retugen/slide_test/2"
    mask_path_root =r'/home/omnisky/hdd_15T_sdc/zttdata/TCGA/breastHer2/HER2_status/retugen/masknpy/2'
    prob_map_root =r'/home/omnisky/hdd_15T_sdc/zttdata/TCGA/breastHer2/HER2_status/retugen/retunpy/2-1'
    args = parser.parse_args()
    for wsi_name in os.listdir(wsi_path_root):
        wsi_path = os.path.join(wsi_path_root, wsi_name)
        print("1.成功读取wsi路径，路径为{}".format(wsi_path))
        (file_name, extension) = os.path.splitext(wsi_name)
        mask_path = os.path.join(mask_path_root, file_name +  ".npy")#+'_'+str(args.level)
        print("2.成功读取mask的路径，路径为{}".format(mask_path))

        main(wsi_path, mask_path, prob_map_root)


