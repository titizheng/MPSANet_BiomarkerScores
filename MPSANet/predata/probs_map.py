from __init__ import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import sys
import argparse
import logging
import json
import time


import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import  pandas as pd

from KFwsi.model import MODELS_resnet

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from KFwsi.data.wsi_producer import GridWSIPatchDataset  # noqa
from model import MODELS  # noqa


parser = argparse.ArgumentParser(description='Get the probability map of tumor'
                                 ' patch predictions given a WSI')
# parser.add_argument('wsi_path', default=None, metavar='WSI_PATH', type=str,
#                     help='Path to the input WSI file')
parser.add_argument('--ckpt_path', default='./train_attention_CRF_5.ckpt', metavar='', type=str,
                    help='Path to the saved ckpt file of a pytorch model')


parser.add_argument('--cfg_path', default='./pre_model_crf.json', metavar='CFG_PATH', type=str,
                    help='Path to the config file in json format related to'
                    ' the ckpt file')

parser.add_argument('--GPU', default='3', type=str, help='which GPU to use'
                    ', default 0')
parser.add_argument('--num_workers', default=2, type=int, help='number of '
                    'workers to use to make batch, default 5')
parser.add_argument('--eight_avg', default=0, type=int, help='if using average'
                    ' of the 8 direction predictions for each patch,'
                    ' default 0, which means disabled')
parser.add_argument('--level', default=2, type=int, help='at which WSI level'
                    ' to obtain the mask, default 6')

##这个是生成概率图
def get_probs_map(model, dataloader,cfg):
    probs_map = np.zeros(dataloader.dataset._mask.shape)  ##这
    num_batch = len(dataloader) ##
    # only use the prediction of the center patch within the grid
    idx_center = dataloader.dataset._grid_size // 2

    count = 0
    P_count=0
    F_count=0
    probs_gl=[] ##
    time_now = time.time()
    for (data, x_mask, y_mask) in dataloader:
        with torch.no_grad():
            data = torch.autograd.Variable(data.cuda()) #, volatile=True,async=True,
        output = model(data)
        # because of torch.squeeze at the end of forward in resnet.py, if the
        # len of dim_0 (batch_size) of predata is 1, then output removes this dim.
        # should be fixed in resnet.py by specifying torch.squeeze(dim=2) later
        if len(output.shape) == 1:
            probs = output[idx_center].sigmoid().cpu().data.numpy().flatten()
        else:
            probs = output[:,
                           idx_center].sigmoid().cpu().data.numpy().flatten()

        probs_map[x_mask, y_mask] = probs
        probs_gl.append(probs)

        for i in probs:
            if i>0.5:
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


    return probs_map,count,probs_gl


def make_dataloader(args, wsi_path, mask_path, cfg, flip='NONE', rotate='NONE'):
    batch_size = cfg['batch_size']
    num_workers = args.num_workers

    dataloader = DataLoader(
        GridWSIPatchDataset(wsi_path, mask_path,
                            image_size=cfg['image_size'],
                            patch_size=cfg['patch_size'],
                            crop_size=cfg['crop_size'], normalize=True,
                            flip=flip, rotate=rotate),
        batch_size=batch_size, num_workers=num_workers, drop_last=False)

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

    ##-------------------------------------------------------------------------
    ckpt = torch.load(args.ckpt_path,map_location='cpu')
    model = MODELS_resnet[cfg['model']](num_nodes=grid_size, use_crf=cfg['use_crf'])
    model.load_state_dict(ckpt['state_dict'])
    model = model.cuda().eval()
    ##-------------------------------------------------------------------------
    if not args.eight_avg:
        dataloader = make_dataloader(args, wsi_path, mask_path, cfg, flip='NONE', rotate='NONE')
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


    wsi_path_root ="/home/omnisky/sata_16tb1/zttdata/IHC/KI-67/slides"  #
    mask_path_root ='/home/omnisky/sata_16tb1/zttdata/IHC/KI-67/mask/level2' #
    prob_map_root ='/home/omnisky/sata_16tb1/zttdata/IHC/KI-67/promap/level2' #
    args = parser.parse_args()
    for wsi_name in os.listdir(wsi_path_root):
        wsi_path = os.path.join(wsi_path_root, wsi_name)
        print("1.sldie path{}".format(wsi_path))
        (file_name, extension) = os.path.splitext(wsi_name)
        mask_path = os.path.join(mask_path_root, file_name +  ".npy")#+'_'+str(args.level)
        print("2.mask path{}".format(mask_path))

        main(wsi_path, mask_path, prob_map_root)


