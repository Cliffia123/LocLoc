# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
from operator import mod, truediv
from pickle import TRUE
import sys
from typing import Iterable, List, Optional
import pdb
import os
import torch.nn.functional as F

import torch
import torch.nn as nn
from torch.nn.modules import conv
from torch.nn.modules.activation import Threshold
import numpy as np
from numpy.lib.function_base import delete

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from utils import utils
from utils.vis_tools import *
from modules.loss import JointsMSELoss

import matplotlib.pyplot as plt

IMAGE_MEAN_VALUE = [0.485, 0.456, 0.406]
IMAGE_STD_VALUE = [0.229, 0.224, 0.225]
T_vis_path = os.path.join(os.getcwd(), 'vis_test')
from tqdm import tqdm

def train_one_epoch(
    model_pre: torch.nn.Module, # type: ignore
    model: torch.nn.Module, # type: ignore
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer, # type: ignore
    device: torch.device,
    epoch: int,
    epochs:int,
    loss_scaler,
    max_norm: float = 0,
    model_ema: Optional[ModelEma] = None,
    mixup_fn: Optional[Mixup] = None,
    set_training_mode=True,
):
    model_pre.eval()
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    data_t = tqdm(data_loader)
    for i, (input, labels, paths) in enumerate(data_t):
        input = input.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size=len(paths)

        with torch.cuda.amp.autocast(): # type: ignore
            interme, _ = model_pre.forward_features(input)
            outputs = model(interme)
            loss = nn.CrossEntropyLoss()(outputs, labels)
    
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = (
            hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        )
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=is_second_order,
        )

        torch.cuda.synchronize() # type: ignore
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
        metric_logger.meters['top1_cls'].update(acc1.item(), n=batch_size)
        metric_logger.meters['top5_cls'].update(acc5.item(), n=batch_size)

        description = "[T:{0:3d}/{1:3d}] Top1-cls: {2:6.2f}, Top5-cls: {3:6.2f}, Loss: {4:7.4f}, ". \
            format(epoch, epochs, metric_logger.top1_cls.global_avg, metric_logger.top5_cls.global_avg, metric_logger.loss.global_avg)
        data_t.set_description(desc=description)
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_localizer(model, generator, train_loader, optimizer, device, epoch, epochs):
    
    losses = utils.AverageMeter('Loss', ':.4e')
    image_mean = torch.reshape(torch.tensor(IMAGE_MEAN_VALUE), (1, 3, 1, 1))
    image_std = torch.reshape(torch.tensor(IMAGE_STD_VALUE), (1, 3, 1, 1))
    
    model.eval()
    generator.train()
    data_t = tqdm(train_loader)
    criterion = JointsMSELoss(use_target_weight=False).cuda()

    for i, (input, target, label, name) in enumerate(data_t):
        bs = input.shape[0]
        target = target.to(device)
        label = label.to(device)
        input = input.to(device)

        _, patch_embed = model.forward_features(input) #Output[96, 196, 384]
        hw = int((patch_embed.shape[1])**0.5) #14
        patch_embed = patch_embed.permute([0, 2, 1]).contiguous()
        patch_embed = patch_embed.reshape(bs,384,hw,hw)
        
        outputs = generator(torch.from_numpy(patch_embed.cpu().numpy()).cuda()) #Output[bs, 1, 56,56]
        target = target.cuda(non_blocking=True)

        if isinstance(outputs, list):
            loss = criterion(outputs[0], target, label)
            for output in outputs[1:]:
                loss += criterion(output, target, label)
        else:
            output = outputs
            loss = criterion(output, target, label)
    
        batch_size=len(target)
        losses.update(loss.item(), batch_size)
        description = "[T:{0:3d}/{1:3d}], Loss: {2:7.4f}, ". \
            format(epoch, epochs, losses.avg)
        data_t.set_description(desc=description)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  
        
    return losses.avg


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


@torch.no_grad()
def evaluate(data_loader, model_pre,  model_classifier, model_generator, device, args=None, threshold_loc=0.5, epoch=0):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    image_mean = torch.reshape(torch.tensor(IMAGE_MEAN_VALUE), (1, 3, 1, 1))
    image_std = torch.reshape(torch.tensor(IMAGE_STD_VALUE), (1, 3, 1, 1))
    # switch to evaluation mode
    model_pre.eval()
    model_classifier.eval()
    model_generator.eval()

    LocSet = []
    IoUSet = []
    IoUSetTop5 = []
    data_t = tqdm(data_loader)
    cnt, cnt_top1, cnt_top5, cnt_loc, thresh = 0, 0, 0, 0, 0.8
    top1top5 = np.load('log/top1top5_cub.npy', allow_pickle=True).item()

    for i,(images, label, gt_bboxes, names) in enumerate(data_t):
        images = images.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        batch = images.shape[0]

        with torch.cuda.amp.autocast():
            prepare_tokens, patch_embed = model_pre.forward_features(images) #Output[96, 196, 384]
            output = model_classifier(prepare_tokens) # Classify Results

            hw = int((patch_embed.shape[1])**0.5)
            patch_embed = patch_embed.permute([0, 2, 1]).contiguous()
            patch_embed = patch_embed.reshape(batch,384, hw, hw)

            loc_maps = model_generator(torch.from_numpy(patch_embed.cpu().numpy()).cuda()) #Output [bs, 1, 56,56]
            _, pre_logits = output.topk(5, 1, True, True)

            image_ = images.clone().detach().cpu() * image_mean + image_std
            image_ = image_.numpy().transpose(0, 2, 3, 1)
            image_ = image_[:, :, :, ::-1] * 255
            
            loc_maps = loc_maps.clone().detach().cpu().numpy().transpose(0, 2, 3, 1)  

            for _b in range(batch):
                # top1_, top5_ = top1top5[names[_b]]
                name = names[_b].split("/")[1]
                predict_box, cam_b = return_box_cam(loc_maps[_b].astype('float32'), image_[_b], gt_bboxes[_b].cpu(), th=threshold_loc)
                # * compute loc acc
                max_iou = -1
                iou = utils.IoU(gt_bboxes[_b].cpu(), predict_box)
                if iou > max_iou:
                    max_iou = iou
                LocSet.append(max_iou)
                temp_loc_iou = max_iou
                if pre_logits[_b][0] != label[_b]:
                    max_iou = -1
                IoUSet.append(max_iou)
                max_iou = -1
                for i in range(5):
                    if pre_logits[_b][i] == label[_b]:
                        max_iou = temp_loc_iou
                        break

                IoUSetTop5.append(max_iou)
                
                # saving_folder = './cam_box'
                # if not os.path.isdir(saving_folder):
                #     os.makedirs(saving_folder)
                # if iou>0.8:
                #     cv2.imwrite(os.path.join(saving_folder, name), cam_b)
                # if iou>=0.5:
                #     cnt_loc  += 1
                #     cnt_top1 += 1 if top1_ else 0
                #     cnt_top5 += 1 if top5_ else 0
                # cnt += 1

        acc1, acc5 = accuracy(output, label, topk=(1, 5))
        batch_size = images.shape[0]

        metric_logger.meters['top1_cls'].update(acc1.item(), n=batch_size)
        metric_logger.meters['top5_cls'].update(acc5.item(), n=batch_size)

        description = "[V:{0:3d}/{1:3d}] Top1-cls: {2:6.2f}, Top5-cls: {3:6.2f} ". \
            format(epoch, args.epochs, metric_logger.top1_cls.global_avg, metric_logger.top5_cls.global_avg)
        data_t.set_description(desc=description)

        # * compute cls loc acc
        loc_acc_top1 = np.sum(np.array(IoUSet) >= 0.5) / len(IoUSet)
        loc_acc_top5 = np.sum(np.array(IoUSetTop5) >= 0.5) / len(IoUSetTop5)
        loc_acc_gt = np.sum(np.array(LocSet) >= 0.5) / len(LocSet)

        # loc_acc_top1 = np.sum(cnt_top1/cnt)
        # loc_acc_top5 = np.sum(cnt_top5/cnt)
        # loc_acc_gt = np.sum(cnt_loc/cnt)
        
        
    metric_logger.update(top1_loc=loc_acc_top1*100)
    metric_logger.update(top5_loc=loc_acc_top5*100)
    metric_logger.update(gt_loc=loc_acc_gt*100)

    del images
    del label
    del names
    del gt_bboxes
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluate_cls(model_pre, model, data_loader, device, args=None, threshold_loc=0.5, epoch=0):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode
    model.eval()

    LocSet = []
    IoUSet = []
    IoUSetTop5 = []
    data_t = tqdm(data_loader)
    model_pre.eval()
    model.eval()
    for i,(images, labels, paths) in enumerate(data_t):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        batch = images.shape[0]

        with torch.cuda.amp.autocast():
            interme,_ = model_pre.forward_features(images)
            outputs = model(interme)
            loss = nn.CrossEntropyLoss()(outputs, labels)

        
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['top1_cls'].update(acc1.item(), n=batch_size)
        metric_logger.meters['top5_cls'].update(acc5.item(), n=batch_size)

        description = "[V:{0:3d}/{1:3d}] Top1-cls: {2:6.2f}, Top5-cls: {3:6.2f}, Loss: {4:7.4f}, ". \
            format(epoch, args.epochs, metric_logger.top1_cls.global_avg, metric_logger.top5_cls.global_avg, metric_logger.loss.global_avg)
        data_t.set_description(desc=description)


    del images
    del labels
    del paths
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def validate_loc(model, generator, val_loader, device, args, epoch):
    losses = utils.AverageMeter('Loss', ':.4e')
    acc_pd = utils.AverageMeter('Acc', ':.4e')
    model.eval()
    generator.eval()

    data_t = tqdm(val_loader)
    prfq = len(val_loader) // 6
    idx = 0
    criterion = JointsMSELoss(use_target_weight=False).cuda()

    with torch.no_grad():
        for i, (input, label, gt_bbox, names) in enumerate(data_t):
            bs = input.shape[0]
            input = input.to(device)

            #get loc
            _, patch_embed = model.forward_features(input) #Output [bs, 384, 14, 14]
            hw = int((patch_embed.shape[1])**0.5) #14
            patch_embed = patch_embed.permute([0, 2, 1]).contiguous()
            patch_embed = patch_embed.reshape(bs,384,hw,hw)

            outputs = generator(torch.from_numpy(patch_embed.cpu().numpy()).cuda()) #Output[bs, 1, 64,64]
            batch_size=len(input)            
            pd_acc, cnt = accuracy_loc(input.detach().cpu(), outputs.detach().cpu().numpy(), gt_bbox, False, 0.5, size=input.size(-1))

            pd_loc = len(pd_acc[pd_acc>=0.5])/len(pd_acc)
            acc_pd.update(pd_loc,cnt)
            idx += bs
            description = "[V:{0:3d}/{1:3d}], GT-K: {2:7.4f}, ".format(epoch, args.epochs, acc_pd.avg)
            data_t.set_description(desc=description)
    return acc_pd.avg

