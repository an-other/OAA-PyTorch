import sys
import os
sys.path.append(os.getcwd())

import torch
import argparse
import time
import shutil
import json
import my_optim
import torch.optim as optim
from models import vgg
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
from utils import AverageMeter
from utils.LoadData import train_data_loader
from tqdm import trange, tqdm


def get_arguments():
    parser = argparse.ArgumentParser(description='The Pytorch code of OAA')
    parser.add_argument("--img_dir", type=str, default='/content/OAA-PyTorch/VOCdevkit/VOC2012/JPEGImages/', help='Directory of training images')
    parser.add_argument("--train_list", type=str, default='/content/OAA-PyTorch/data/voc12/train_cls.txt')
    parser.add_argument("--test_list", type=str, default='/content/OAA-PyTorch/data/voc12/val_cls.txt')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--iter_size", type=int, default=5)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--decay_points", type=str, default='10')
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--disp_interval", type=int, default=100)
    parser.add_argument("--snapshot_dir", type=str, default='exp')
    parser.add_argument("--resume", type=str, default='False')
    parser.add_argument("--global_counter", type=int, default=0)
    parser.add_argument("--current_epoch", type=int, default=0)
    parser.add_argument("--att_dir", type=str, default='/content/OAA-PyTorch/content/OAA-PyTorch/runs/exp1/')
    parser.add_argument("--restore_from", type=str, default='/content/drive/MyDrive/oaa/imagenet_epoch_19.pth')

    return parser.parse_args()

def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):
    savepath = os.path.join(args.snapshot_dir, filename)
    torch.save(state, savepath)
    if is_best:
        shutil.copyfile(savepath, os.path.join(args.snapshot_dir, 'model_best.pth.tar'))

def get_model(args):
    model = vgg.vgg16(pretrained=True, num_classes=args.num_classes, att_dir=args.att_dir, training_epoch=args.epoch)
    model = torch.nn.DataParallel(model).cuda()
    param_groups = model.module.get_parameter_groups()
    optimizer = optim.SGD([
        {'params': param_groups[0], 'lr': args.lr},
        {'params': param_groups[1], 'lr': 2*args.lr},
        {'params': param_groups[2], 'lr': 10*args.lr},
        {'params': param_groups[3], 'lr': 20*args.lr}], momentum=0.9, weight_decay=args.weight_decay, nesterov=True)

    return  model, optimizer

def load_checkpoint(args):
    model = vgg.vgg16(num_classes=args.num_classes,att_dir=args.att_dir, training_epoch=args.epoch)
    model = torch.nn.DataParallel(model).cuda()

    pretrained_dict = torch.load(args.restore_from)['state_dict']
    optimizer_dict = torch.load(args.restore_from)['optimizer']
    model_dict = model.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    print("Weights cannot be loaded:")
    print([k for k in model_dict.keys() if k not in pretrained_dict.keys()])

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('load model successfully')
    param_groups = model.module.get_parameter_groups()
    optimizer = optim.SGD([
        {'params': param_groups[0], 'lr': args.lr},
        {'params': param_groups[1], 'lr': 2*args.lr},
        {'params': param_groups[2], 'lr': 10*args.lr},
        {'params': param_groups[3], 'lr': 20*args.lr}], momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    optimizer.load_state_dict(optimizer_dict)
    print('load optimizer successfully')
    return model,optimizer

def validate(model, val_loader):
    
    print('\nvalidating ... ', flush=True, end='')
    val_loss = AverageMeter()
    model.eval()
    
    with torch.no_grad():
        for idx, dat in tqdm(enumerate(val_loader)):
            img_name, img, label = dat
            label = label.cuda(non_blocking=True)
            logits = model(img)
            if len(logits.shape) == 1:
                logits = logits.reshape(label.shape)
            loss_val = F.multilabel_soft_margin_loss(logits, label)   
            val_loss.update(loss_val.data.item(), img.size()[0])

    print('validating loss:', val_loss.avg)

def train(args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    
    total_epoch = args.epoch
    #global_counter = args.global_counter
    global_counter = torch.load(args.restore_from)['global_counter']
    #current_epoch = args.current_epoch
    current_epoch = torch.load(args.restore_from)['epoch']

    train_loader, val_loader = train_data_loader(args)
    max_step = total_epoch*len(train_loader)
    args.max_step = max_step 
    print('Max step:', max_step)
    
    #model, optimizer = get_model(args)
    model,optimizer=load_checkpoint(args)
    #print(model)
    model.train()
    end = time.time()

    while current_epoch < total_epoch:
        model.train()
        losses.reset()
        batch_time.reset()
        res = my_optim.reduce_lr(args, optimizer, current_epoch)
        steps_per_epoch = len(train_loader)
        
        #validate(model, val_loader)
        index = 0  
        flag = 0
        for idx, dat in enumerate(train_loader):
            img_name, img, label = dat
            label = label.cuda(non_blocking=True)
            
            logits = model(img, current_epoch, label, index)
            index += args.batch_size

            if len(logits.shape) == 1:
                logits = logits.reshape(label.shape)
            loss_val = F.multilabel_soft_margin_loss(logits, label) / args.iter_size
            loss_val.backward()

            flag += 1
            if flag == args.iter_size:
                optimizer.step()
                optimizer.zero_grad()
                flag = 0

            losses.update(loss_val.data.item(), img.size()[0])
            batch_time.update(time.time() - end)
            end = time.time()
            
            global_counter += 1
            if global_counter % 1000 == 0:
                losses.reset()

            if global_counter % args.disp_interval == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'LR: {:.5f}\t' 
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        current_epoch, global_counter%len(train_loader), len(train_loader), 
                        optimizer.param_groups[0]['lr'], loss=losses))

        if (current_epoch == args.epoch-1) | (current_epoch%5==0):
            save_checkpoint(args,
                            {
                                'epoch': current_epoch,
                                'global_counter': global_counter,
                                'state_dict':model.state_dict(),
                                'optimizer':optimizer.state_dict()
                            }, is_best=False,
                            filename='%s_epoch_%d.pth' %(args.dataset, current_epoch))
        current_epoch += 1

if __name__ == '__main__':
    args = get_arguments()
    print('Running parameters:\n', args)
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    train(args)
