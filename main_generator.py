# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
from os import makedirs
from sys import prefix
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import pdb
import os
import time
from tqdm import tqdm

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma
from ptflops import get_model_complexity_info

from utils.cub_feat import build_dataset
from utils.samplers import RASampler
from utils import utils
from utils.get_args import get_args_parser

from modules.models import VisionTransformer, MiniTransformer, UnetNet, DeconvNet
from modules import models
from modules.ops import *
from engine import train_one_epoch,train_localizer, evaluate, evaluate_cls, validate_loc

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

    dataset_train, dataset_val, args.nb_classes = build_dataset(args=args)

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print(
                    'Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.'
                )
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
            )
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

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

    for p in model_pre.parameters():
            p.requires_grad = False
    model_pre.eval()
    
    #* Training Generator
    # generator = UnetNet(num_classes=200)
    generator = DeconvNet()
    generator = generator.cuda()

    if args.distributed:
        generator = torch.nn.parallel.DistributedDataParallel(
            generator, device_ids=[args.gpu], find_unused_parameters=True
        )
        generator = generator.module

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, generator)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, [111,1112], 0.1,
            last_epoch=-1) 
    # loss_scaler = NativeScaler()

    # lr_scheduler, _ = create_scheduler(args, optimizer)


    # output_dir = Path(args.output_dir)
	
    # * Begin Training
    print("====================Training Generator====================")
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_perf = 0.0
    best_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        # if args.distributed:
        #     data_loader_train.sampler.set_epoch(epoch)

        losses = train_localizer(model_pre, generator, data_loader_train, optimizer, device, epoch,  args.epochs)
        perf_indicator = validate_loc(model_pre, generator, data_loader_val, device, args, epoch)
        
        if perf_indicator >= best_perf:
            best_perf = perf_indicator
            best_epoch = epoch
            path = os.path.join('log',args.data_set+'_loc_best.pth')
            print("Evaluation Result:\n""loss: {0:.3f},predit_acc:{1:.3f}".format(losses, best_perf*100))
            utils.save_on_master({'state_dict': generator.state_dict(),},path,)
        print("Until %d epochs, Best Loc Epoch: %d" % (epoch, best_epoch ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'DeiT training and evaluation script', parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
