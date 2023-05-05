# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
from os import makedirs
from pyexpat import model
from symbol import parameters
from sys import prefix
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import pdb
import os
import time
from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma
from ptflops import get_model_complexity_info

from utils.datasets import build_dataset
from engine import train_one_epoch, evaluate
from utils.samplers import RASampler
from modules.models import VisionTransformer, UnetNet, DeconvNet
from utils import utils
from modules.ops import *
from utils.get_args import get_args_parser
from utils.cub_feat import CUBFEATDataset, ImageNetDataset

import modules.models as vits
from functools import partial


def main(args):
    utils.init_distributed_mode(args)
    
    print(args)

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    if args.data_set == "CUB":
        data_loader_val = torch.utils.data.DataLoader(
            CUBFEATDataset(root=args.data_path, is_train=False),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        
    elif args.data_set == "IMNET":
        data_loader_val = torch.utils.data.DataLoader(
            ImageNetDataset(root=args.data_path, is_train=False),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    args.nb_classes=200
    print(f"Creating model: {args.model}")
    model_pre = create_model(
        'deit_pre_model',
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )
    checkpoint = torch.load(args.resume, map_location='cpu')
    print('load pretrained model from {}'.format(args.resume))
    model_dict = model_pre.state_dict()
    pretrained_dict = checkpoint['model']
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model_pre.load_state_dict(model_dict, strict=False)
    model_pre.to(device)
    model_pre.eval()
    

    # * Loading Classifier
    print("====================Loading Classifier====================")
    if not os.path.exists(args.output_dir):
        os.path.makedirs(args.output_dir)

    model_classifier = create_model(
        'deit_mini_transformer',
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )

    model_classifier.to(device)
    model_classifier.eval()


    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_classifier)
    lr_scheduler, _ = create_scheduler(args, optimizer)
    print('load pretrained model from {}'.format(args.classifier))
    cls_dict = torch.load(args.classifier, map_location='cpu')
    cls_dict = cls_dict['state_dict']
    cls_dict = {k.replace("module.", ""): v for k, v in cls_dict.items()}
    msg = model_classifier.load_state_dict(cls_dict)
    print('Loaded Classifier Encoder with msg: {}'.format(msg))

    # * Loading Generator
    print("====================Loading Generator====================")
    model_generator = UnetNet(num_classes=200)
    # model_generator = DeconvNet()
    model_generator = model_generator.cuda()
    model_generator.eval()

    print('load pretrained model from {}'.format(args.locator))
    rg_dict = torch.load(args.locator, map_location='cpu')
    rg_dict = rg_dict['state_dict']
    msg = model_generator.load_state_dict(rg_dict)
    print('Loaded Localization Encoder with msg: {}'.format(msg))

    if args.data_set == 'IMNET':
        threshold_loc_list = [0.3,0.34,0.38,0.4,0.42,0.45]
    else:
        threshold_loc_list = [0.45, 0.48, 0.5, 0.51, 0.52, 0.53]

    for _th in threshold_loc_list:
        test_stats = evaluate(data_loader_val, model_pre, model_classifier, model_generator, device, args, threshold_loc=_th, epoch=0)
        print("Evaluation Result:\n"
                "Loc GT:{0:.3f}\n"
                "Loc Top1:{1:.3f} Loc Top5:{2:.3f}\n"
                "Cls Top1: {3:.3f} Cls Top5:{4:.3f}".
                format(test_stats["gt_loc"], 
                test_stats["top1_loc"], test_stats["top5_loc"], 
                test_stats["top1_cls"], test_stats["top5_cls"]))
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'DeiT training and evaluation script', parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)