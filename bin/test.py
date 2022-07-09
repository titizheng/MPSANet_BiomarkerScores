import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import argparse
import logging
import json
import time

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from tensorboardX import SummaryWriter

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from data.image_producer import GridImageDataset  # noqa
from wsi.model import MODELS, MODELS_resnet ,MODELS_vgg # noqa
import seaborn as sn
import pandas as pd
import  matplotlib.pylab as plt
from sklearn import metrics
import numpy as np



#测试的时候是没有标签的，生成patch，让模型自己预测，有多少个正，多少个负，根据正负patch的个数，输出最后的标签

parser = argparse.ArgumentParser(description='Train model')

parser.add_argument('--ckpt_path', default=r'/home/omnisky/hdd_15T_sdc/zttdata/TCGA/TCGAHEHER2/testresult/KF-result/train_train.ckpt', metavar='', type=str,
                    help='Path to the saved ckpt file of a pytorch model')

parser.add_argument('--cfg_path', default=r'/home/omnisky/hdd_15T_sdc/zttcode/cd-30NCRF-KF2/cd-30NCRF-KF/configs/resnet18_crf_test3.json', metavar='CFG_PATH', type=str,
                    help='Path to the config file in json format related to'
                    ' the ckpt file')



parser.add_argument('--save_path', default=r"/home/omnisky/hdd_15T_sdc/zttdata/TCGA/TCGAHEHER2/testresult/KF-result/Test", metavar='SAVE_PATH', type=str,
                    help='Path to the saved models')


parser.add_argument('--num_workers', default=0, type=int, help='number of'
                    ' workers for each data loader, default 2.') ##单线程工作，这样的缺点就是很慢
parser.add_argument('--device_ids', default='1', type=str, help='comma'
                    ' separated indices of GPU to use, e.g. 0,1 for using GPU_0'
                    ' and GPU_1, default 0.')


def valid_epoch(summary, cfg, model,
                dataloader_tumor, dataloader_normal):
    model.eval()

    steps = len(dataloader_tumor)
    batch_size = dataloader_tumor.batch_size
    grid_size = dataloader_tumor.dataset._grid_size
    dataiter_tumor = iter(dataloader_tumor)
    dataiter_normal = iter(dataloader_normal)


    acc_sum = 0
    y_true=[] ##模型的真实标签所有patch的
    probs_gl=[]##存放模型输出结果的所有概率值
    predicts_lable=[] ##模型预测的所有patch的标签值
    for step in range(steps):
        with torch.no_grad():
            data_tumor, target_tumor = next(dataiter_tumor)
        #####################----------------------------------删除了原本的版本-------------------------
            data_tumor = Variable(data_tumor.cuda(async=True))
            target_tumor = Variable(target_tumor.cuda(async=True))

            data_normal, target_normal= next(dataiter_normal)
            data_normal = Variable(data_normal.cuda(async=True))
            target_normal = Variable(target_normal.cuda(async=True))

            # print('输出肿瘤的标签',target_tumor)
            # print('输出正常的标签', target_normal)
        data = torch.cat([data_tumor, data_normal])
        target = torch.cat([target_tumor, target_normal])
        output = model(data)

        probs = output.sigmoid()
        ##------------------------------真实的标签------------
        # target1 =target.cpu().numpy()
        # y_true.append(target1)
        # print('输出真实的标签', target1)

        ##------------------------------模型输出的概率值------------
        probs1=probs.detach().cpu().numpy()
        # probs_gl.append(probs1)
        # print('输出模型预测的概率值',probs1)

        ##------------------------------模型输出的标签------------
        # predicts = (max(probs) >= 0.5).type(torch.cuda.FloatTensor) ##把大于0.5的都赋值为1，小于0.5的赋值为0
        # predicts1 = predicts.cpu().numpy()
        # predicts_lable.append(predicts1)

        predicts_list = []  #这个是预测的标签值
        for presi in probs1:
            val =max(presi)
            if val > 0.8:
                predicts_list.append(1.0000)
            else:
                predicts_list.append(0.0000)
            probs_gl.append(val)

        target_chag = []
        for tari in target:
            if 1.0000 in tari:
                target_chag.append(1.0000)
            else:
                target_chag.append(0.0000)

        count = 0
        for i,j in zip(target_chag,predicts_list):
            if i == j:
                count+=1
        acc_data = count *1.0 / (batch_size  * 2)

        # print('输出真实的标签',target_chag)
        # print('输出模型预测的标签',predicts_list)

        y_true.append(target_chag)
        predicts_lable.append(predicts_list) #predicts_lable这个是所有patch的标签，predicts_list 这个是每次batch_size的标签

        acc_sum += acc_data

    summary['acc'] = acc_sum / steps

    print('输出模型的精度',summary['acc'])

    # -------------------------------------混淆矩阵-----------------

    kind =['Normal','Tumour']  # 类别名称
    conf_numpy = metrics.confusion_matrix(np.array(y_true).flatten(), np.array(predicts_lable).flatten())
    print('输出混淆矩阵的值',conf_numpy)
    conf_df = pd.DataFrame(conf_numpy, index=kind, columns=kind)  # 将矩阵转化为 DataFrame
    conf_fig = sn.heatmap(conf_df, annot=True,center=None, fmt="d")  # 绘制 heatmap
    plt.savefig(r"/home/omnisky/hdd_15T_sdc/zttdata/TCGA/TCGAHEHER2/testresult/KF-result/Test/hunxiaojuzhen.png")
    plt.show()


  ##------------------------------

    tn = conf_numpy[0][0]
    fp = conf_numpy[0][1]
    fn = conf_numpy[1][0]
    tp = conf_numpy[1][1]



    # summary['acc'] = acc_sum / steps
    summary['Recall']=tp*1.0/(tp+fn)
    summary['Precision']=tp*1.0/(tp+fp)
    summary['Specificity'] = tn * 1.0 / (tn + fp)
    summary['F1']=(2* summary['Recall']*summary['Precision'])/(summary['Recall']+summary['Precision'])



    #-------------------------------------ROC-----------------
    print('输出所有的真实标签',np.array(y_true).flatten())
    fpr, tpr, thresholds_keras = metrics.roc_curve(np.array(y_true).flatten(),np.array(probs_gl).flatten())
    auc = metrics.auc(fpr, tpr)
    print("AUC : ", auc)
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='AUC (area = {:.3f})'.format(auc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC')
    plt.legend(loc='best')
    plt.savefig(r"/home/omnisky/hdd_15T_sdc/zttdata/TCGA/TCGAHEHER2/testresult/KF-result/Test/ROC_2.png")
    plt.show()



    return summary


def run(args):
    with open(args.cfg_path) as f:
        cfg = json.load(f) ##加载json文件

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)  #创建保存模型的文件

    with open(os.path.join(args.save_path, 'cfg.json'), 'w') as f:
        json.dump(cfg, f, indent=1)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_ids
    num_GPU = len(args.device_ids.split(','))
    # batch_size_train = cfg['batch_size'] * num_GPU
    # batch_size_valid = cfg['batch_size'] * num_GPU

    batch_size_valid = cfg['batch_size']
    num_workers = args.num_workers * num_GPU

    if cfg['image_size'] % cfg['patch_size'] != 0:
            raise Exception('Image size / patch size != 0 : {} / {}'.
                            format(cfg['image_size'], cfg['patch_size']))

    patch_per_side = cfg['image_size'] // cfg['patch_size'] ##最大的图片要是小的图片的几倍
    grid_size = patch_per_side * patch_per_side##生成网格的形状

    turmor_path = np.array(os.listdir(cfg['data_path_tumor_valid']))  ##只有数组的形式，下面的用turmor_path[train_index]这种形式取才不会出错
    normal_path = np.array(os.listdir(cfg['data_path_normal_valid']))

#------------------------------------------------------------测试集数据处理部分-----------------------------------
##测试的时候不进行数据增强


    dataset_tumor_valid = GridImageDataset(data_path=cfg['data_path_tumor_valid'],
                                           image_name_list=turmor_path,
                                           json_path=cfg['json_path_valid'],
                                           img_size=cfg['image_size'],
                                           patch_size=cfg['patch_size'],
                                           crop_size=cfg['crop_size'])
    dataset_normal_valid = GridImageDataset(data_path=cfg['data_path_normal_valid'],
                                            image_name_list=normal_path,
                                            json_path=cfg['json_path_valid'],
                                            img_size=cfg['image_size'],
                                            patch_size=cfg['patch_size'],
                                            crop_size=cfg['crop_size'])

    dataloader_tumor_valid = DataLoader(dataset_tumor_valid,
                                        batch_size=batch_size_valid,
                                        num_workers=num_workers,
                                        drop_last=True
                                        )
    dataloader_normal_valid = DataLoader(dataset_normal_valid,
                                         batch_size=batch_size_valid,
                                         num_workers=num_workers,
                                         drop_last=True
                                         )


    summary_valid = {'loss': float('inf'), 'acc': 0}
    summary_writer = SummaryWriter(args.save_path) ##这个是保存模型的路径，此时保存的是验证集的损失和精度

#####-------------------------------------加载模型----------------------------
    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    model = MODELS_resnet[cfg['model']](num_nodes=grid_size, use_crf=cfg['use_crf'])
    # model = MODELS_vgg[cfg['model']](num_nodes=grid_size, use_crf=cfg['use_crf'])

    model.load_state_dict(ckpt['state_dict'])
    model = model.cuda().eval()

    time_now = time.time()

    summary_valid = valid_epoch(summary_valid, cfg, model,
                                    dataloader_tumor_valid,
                                    dataloader_normal_valid)
    time_spent = time.time() - time_now

    logging.info('{}, , Test Loss : {:.5f}, '
            'Test Acc : {:.3f}, Run Time : {:.2f}'
            .format(
                time.strftime("%Y-%m-%d %H:%M:%S"),  summary_valid['loss'],summary_valid['acc'], time_spent))

    logging.info(
        '{},  Validation Loss : {:.5f}, '
        'Validation Acc : {:.3f},'
        'Validation Recall : {:.3f},'
        'Validation Precision : {:.3f},'
        'Validation Specificity : {:.3f},'
        'Validation F1 : {:.3f},'
        ' Run Time : {:.2f}'
            .format(
            time.strftime("%Y-%m-%d %H:%M:%S"),  summary_valid['loss'],
            summary_valid['acc'], summary_valid['Recall'],
            summary_valid['Precision'],
            summary_valid['Specificity'],
            summary_valid['F1'],
            time_spent))


    summary_writer.add_scalar(
            'valid/loss', summary_valid['loss'])
    summary_writer.add_scalar(
            'valid/acc', summary_valid['acc'])
    summary_writer.close()


def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
