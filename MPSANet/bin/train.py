import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import argparse
import logging
import json
import time

import torch

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss, DataParallel
from torch.optim import SGD
from torch.optim import Adam

from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from tensorboardX import SummaryWriter
from sklearn.metrics import auc, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
from KFwsi.data.image_producer import GridImageDataset
# from KFwsi.predata.image_producer import GridImageDataset  # noqa
from KFwsi.model import MODELS, MODELS_vgg, MODELS_resnet  # ,MODELS_Efficient # noqa
from sklearn.model_selection import KFold
import numpy as np

parser = argparse.ArgumentParser(description='Train model')


parser.add_argument('--cfg_path', default=r".json",  metavar='CFG_PATH', type=str,    help='Path to the config file in json format')
parser.add_argument('--save_path',   default=r"",  metavar='SAVE_PATH', type=str, help='Path to the saved models')

parser.add_argument('--num_workers', default=0, type=int, help='number of' ' workers for each predata loader, default 2.')
parser.add_argument('--device_ids', default='2', type=str, help='comma' ' separated indices of GPU to use, e.g. 0,1 for using GPU_0' ' and GPU_1, default 0.')



def train_epoch(summary, summary_writer, cfg, model, loss_fn, optimizer,
                dataloader_tumor, dataloader_normal):
    model.train()

    steps = len(dataloader_tumor)
    batch_size = dataloader_tumor.batch_size
    grid_size = dataloader_tumor.dataset._grid_size
    dataiter_tumor = iter(dataloader_tumor)  ##
    dataiter_normal = iter(dataloader_normal)

    time_now = time.time()
    for step in range(steps):
        ##获取阳性样本和标签
        data_tumor, target_tumor = next(dataiter_tumor)
        data_tumor = Variable(data_tumor.cuda(async=True))
        target_tumor = Variable(target_tumor.cuda(async=True))
        # 获取阴性样本和标签
        data_normal, target_normal = next(dataiter_normal)
        data_normal = Variable(data_normal.cuda(async=True))
        target_normal = Variable(target_normal.cuda(async=True))


        idx_rand = Variable(
            torch.randperm(batch_size * 2).cuda(async=True))


        data = torch.cat([data_tumor, data_normal])[idx_rand]
        target = torch.cat([target_tumor, target_normal])[idx_rand]
        output = model(data)



        # target=target.view()
        loss = loss_fn(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        probs = output.sigmoid()
        predicts = (probs >= 0.5).type(torch.cuda.FloatTensor)
        acc_data = (predicts == target).type(
            torch.cuda.FloatTensor).sum().item() * 1.0 / (
                           batch_size * grid_size * 2)
        loss_data = loss.item()

        time_spent = time.time() - time_now
        time_now = time.time()
        if step % 50 == 0:
            logging.info(
                '{}, Epoch : {}, Step : {}, Training Loss : {:.5f}, '
                'Training Acc : {:.3f}, Run Time : {:.2f}'
                    .format(
                    time.strftime("%Y-%m-%d %H:%M:%S"), summary['epoch'] + 1,
                                                        summary['step'] + 1, loss_data, acc_data, time_spent))

        summary['step'] += 1

        if summary['step'] % cfg['log_every'] == 0:
            summary_writer.add_scalar('train/loss', loss_data, summary['step'])
            summary_writer.add_scalar('train/acc', acc_data, summary['step'])

    summary['epoch'] += 1

    return summary




def valid_epoch(summary, cfg, model, loss_fn,
                dataloader_tumor, dataloader_normal):
    model.eval()

    steps = len(dataloader_tumor)
    batch_size = dataloader_tumor.batch_size
    grid_size = dataloader_tumor.dataset._grid_size
    dataiter_tumor = iter(dataloader_tumor)
    dataiter_normal = iter(dataloader_normal)

    loss_sum = 0
    acc_sum = 0
    tn_sum = 0
    fp_sum = 0
    fn_sum = 0
    tp_sum = 0

    y_true = []
    probs_gl = []
    predicts_lable = []

    for step in range(steps):
        with torch.no_grad():
            data_tumor, target_tumor = next(dataiter_tumor)
            data_tumor = Variable(data_tumor.cuda(async=True))
            target_tumor = Variable(target_tumor.cuda(async=True))

            data_normal, target_normal = next(dataiter_normal)
            data_normal = Variable(data_normal.cuda(async=True))  # volatile=True
            target_normal = Variable(target_normal.cuda(async=True))

        data = torch.cat([data_tumor, data_normal])
        target = torch.cat([target_tumor, target_normal])
        output = model(data)
        loss = loss_fn(output, target)

        probs = output.sigmoid()
        predicts = (probs >= 0.5).type(torch.cuda.FloatTensor)
        acc_data = (predicts == target).type(
            torch.cuda.FloatTensor).sum().item() * 1.0 / (
                           batch_size * grid_size * 2)
        loss_data = loss.item()

        loss_sum += loss_data
        acc_sum += acc_data


        cm = confusion_matrix(np.array(target.cpu()).flatten(), np.array(predicts.cpu()).flatten())

        tn = cm[0][0]
        fp = cm[0][1]
        fn = cm[1][0]
        tp = cm[1][1]

        tn_sum += tn
        fp_sum += fp
        fn_sum += fn
        tp_sum += tp

        target1 = target.cpu().numpy()
        y_true.append(target1)

        probs1 = probs.detach().cpu().numpy()
        probs_gl.append(probs1)


    summary['loss'] = loss_sum / steps
    summary['acc'] = acc_sum / steps
    summary['Recall'] = tp_sum * 1.0 / (tp_sum + fn_sum)
    summary['Precision'] = tp_sum * 1.0 / (tp_sum + fp_sum)
    summary['Specificity'] = tn_sum * 1.0 / (tn_sum + fp_sum)
    summary['F1'] = (2 * summary['Recall'] * summary['Precision']) / (summary['Recall'] + summary['Precision'])

    return (summary, y_true, probs_gl)


def run(args):
    global fpr, tpr, roc_auc
    with open(args.cfg_path) as f:
        cfg = json.load(f)  ##

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)  #
    with open(os.path.join(args.save_path, 'cfg.json'), 'w') as f:

        json.dump(cfg, f, indent=1)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_ids
    num_GPU = len(args.device_ids.split(','))
    batch_size_train = cfg['batch_size'] * num_GPU


    batch_size_valid = cfg['batch_size']

    # num_workers = args.num_workers * num_GPU
    num_workers = args.num_workers

    if cfg['image_size'] % cfg['patch_size'] != 0:
        raise Exception('Image size / patch size != 0 : {} / {}'.
                        format(cfg['image_size'], cfg['patch_size']))

    patch_per_side = cfg['image_size'] // cfg['patch_size']  ##
    grid_size = patch_per_side * patch_per_side  ##

    # model = MODELS_vgg[cfg['model']](num_nodes=grid_size, use_crf=cfg['use_crf'])
    model = MODELS_resnet[cfg['model']](num_nodes=grid_size, use_crf=cfg['use_crf'])

    # model = MODELS_Efficient[cfg['model']]

    # model = MODELS[cfg['model']](num_nodes=grid_size, use_crf=cfg['use_crf'])


    # model = DataParallel(model, device_ids=None)  ##
    model = model.cuda()
    loss_fn = BCEWithLogitsLoss().cuda()

    optimizer = SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'])

    # optimizer = Adam(model.parameters(), lr=cfg['lr'])


    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    turmor_path = np.array(os.listdir(cfg['data_path_tumor_train']))  #
    normal_path = np.array(os.listdir(cfg['data_path_normal_train']))  ##
    kf = KFold(n_splits=5, shuffle=True)
    kf.get_n_splits(turmor_path, normal_path)  #
    i = 1

    fig, ax = plt.subplots(1, 1)
    x1 = []
    y1 = []

    for train_index, val_index in kf.split(normal_path):

        print('*******************************The {} th cross validation, training set data length {}, verification set data length {}'
              '************************'.format(i,
                                                                                                           len(train_index),
                                                                                                           len(val_index)))
        tumor_train, normal_train = turmor_path[train_index], normal_path[train_index]  ##
        tumor_val, normal_val = turmor_path[val_index], normal_path[val_index]


        dataset_tumor_train = GridImageDataset(data_path=cfg['data_path_tumor_train'],
                                               image_name_list=tumor_train,
                                               json_path=cfg['json_path_train'],
                                               img_size=cfg['image_size'],
                                               patch_size=cfg['patch_size'],
                                               crop_size=cfg['crop_size'])
        dataset_normal_train = GridImageDataset(data_path=cfg['data_path_normal_train'],
                                                image_name_list=normal_train,
                                                json_path=cfg['json_path_train'],
                                                img_size=cfg['image_size'],
                                                patch_size=cfg['patch_size'],
                                                crop_size=cfg['crop_size'])

        dataset_tumor_valid = GridImageDataset(data_path=cfg['data_path_tumor_train'],
                                               image_name_list=tumor_val,
                                               json_path=cfg['json_path_train'],
                                               img_size=cfg['image_size'],
                                               patch_size=cfg['patch_size'],
                                               crop_size=cfg['crop_size'])
        dataset_normal_valid = GridImageDataset(data_path=cfg['data_path_normal_train'],
                                                image_name_list=normal_val,
                                                json_path=cfg['json_path_train'],
                                                img_size=cfg['image_size'],
                                                patch_size=cfg['patch_size'],
                                                crop_size=cfg['crop_size'])

        dataloader_tumor_train = DataLoader(dataset_tumor_train,
                                            batch_size=batch_size_train,
                                            num_workers=num_workers,
                                            drop_last=True,
                                            shuffle=True
                                            )
        dataloader_normal_train = DataLoader(dataset_normal_train,
                                             batch_size=batch_size_train,
                                             num_workers=num_workers,
                                             drop_last=True,
                                             shuffle=True
                                             )
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

        summary_train = {'epoch': 0, 'step': 0}
        # summary_valid = {'loss': float('inf'), 'acc': 0}
        summary_valid = {'loss': float('inf'), 'acc': 0, 'Recall': 0, 'Precision': 0, 'Specificity': 0, 'F1': 0}
        summary_writer = SummaryWriter(args.save_path)
        loss_valid_best = float('inf')
        for epoch in range(cfg['epoch']):

            summary_train = train_epoch(summary_train, summary_writer, cfg, model,
                                        loss_fn, optimizer,
                                        dataloader_tumor_train,
                                        dataloader_normal_train)
            torch.save({'epoch': summary_train['epoch'],
                        'step': summary_train['step'],
                        'state_dict': model.state_dict()},  # model.module.state_dict()
                       os.path.join(args.save_path, 'train_train.ckpt'))

        time_now = time.time()

        # if len(dataloader_normal_valid) is not 0:
        summary_valid, y_true, probs_gl = valid_epoch(summary_valid, cfg, model, loss_fn,
                                                      dataloader_tumor_valid,
                                                      dataloader_normal_valid)
        time_spent = time.time() - time_now

        logging.info(
            '{}, Epoch : {}, Step : {}, Validation Loss : {:.5f}, '
            'Validation Acc : {:.3f},'
            'Validation Recall : {:.3f},'
            'Validation Precision : {:.3f},'
            'Validation Specificity : {:.3f},'
            'Validation F1 : {:.3f},'
            ' Run Time : {:.2f}'
                .format(
                time.strftime("%Y-%m-%d %H:%M:%S"), summary_train['epoch'],
                summary_train['step'], summary_valid['loss'],
                summary_valid['acc'], summary_valid['Recall'],
                summary_valid['Precision'],
                summary_valid['Specificity'],
                summary_valid['F1'],
                time_spent))

        summary_writer.add_scalar(
            'valid/loss', summary_valid['loss'], summary_train['step'])
        summary_writer.add_scalar(
            'valid/acc', summary_valid['acc'], summary_train['step'])

        if summary_valid['loss'] < loss_valid_best:
            loss_valid_best = summary_valid['loss']

            torch.save({'epoch': summary_train['epoch'],
                        'step': summary_train['step'],
                        'state_dict': model.state_dict()},  # model.module.state_dict()
                       os.path.join(args.save_path, 'val_best.ckpt'))

        summary_writer.close()


        fpr, tpr, thresholds_keras = roc_curve(np.array(y_true).flatten(), np.array(probs_gl).flatten(), 1)

        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        # 计算auc
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)



        ax.plot(fpr, tpr, lw=1, alpha=0.3, label='fold %d(area=%0.2f)' % (i, roc_auc))
        i += 1
        x1.append(fpr)
        y1.append(tpr)

        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')

    # 画对角线
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'AUC = %0.2f $\pm$ %0.2f' % (mean_auc, std_auc),
            lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_tpr, tprs_lower, tprs_upper, color='gray', alpha=.2)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])

    ax.set_title('ROC')
    ax.legend(loc='lower right')  # lower right


    axins = inset_axes(ax, width="30%", height="30%", loc='lower left',
                       bbox_to_anchor=(0.2, 0.1, 1, 1),
                       bbox_transform=ax.transAxes)  #


    for x, y in zip(x1, y1):
        axins.plot(x, y, lw=1, alpha=0.3)


    xlim0 = 0
    xlim1 = 0.3
    # Y
    ylim0 = 0.7
    ylim1 = 1

    #
    axins.set_xlim(xlim0, xlim1)
    axins.set_ylim(ylim0, ylim1)

    #
    axins.set_xlim(xlim0, xlim1)
    axins.set_ylim(ylim0, ylim1)
    mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)



    plt.savefig(r'/model_ROC.png')
    plt.show()


def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()

