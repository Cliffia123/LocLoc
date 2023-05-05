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

from utils.datasets import build_dataset
from engine import train_one_epoch, evaluate, evaluate_cls
from utils.samplers import RASampler
from modules import models

from utils import utils

from modules.ops import *
from utils.get_args import get_args_parser

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
            sampler_train = torch.utils.data.DistributedSampler( # type: ignore
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print(
                    'Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.'
                )
            sampler_val = torch.utils.data.DistributedSampler( # type: ignore
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
            )
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val) # type: ignore
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader( # type: ignore
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader( # type: ignore
        dataset_val,
        sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.nb_classes,
        )

    print(f"Creating model: {args.model}")
    pre_model = create_model(
        'deit_pre_model',
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )
    checkpoint = torch.load(args.resume, map_location='cpu')
    print('load pretrained model from {}'.format(args.resume))
    model_dict = pre_model.state_dict()
    pretrained_dict = checkpoint['model']
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    pre_model.load_state_dict(model_dict, strict=False)
    pre_model.to(device)

    for p in pre_model.parameters():
            p.requires_grad = False
    pre_model.eval()
    
    
    model = create_model(
        'deit_mini_transformer',
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='',
        )

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool='avg',
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url( # type: ignore
                args.teacher_path, map_location='cpu', check_hash=True
            )
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()

    output_dir = Path(args.output_dir)
    
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url( # type: ignore
                args.resume, map_location='cpu', check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        print('load pretrained model from {}'.format(args.resume))
        model_dict = model_without_ddp.state_dict()
        pretrained_dict = checkpoint['model']
        pretrained_dict_new = {}
        for k, v in pretrained_dict.items():
            k_list = k.split('.')
            k_list[0] = k_list[0]+'_mini'
            k = '.'.join(k_list)
            pretrained_dict_new[k] = v
#         for k in model_dict.keys():
#             if k not in pretrained_dict:
#                 print('Key {} is new added for Our Net'.format(k))
#         pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict_new)
        model_without_ddp.load_state_dict(model_dict, strict=False)

        if (
            not args.eval
            and 'optimizer' in checkpoint
            and 'lr_scheduler' in checkpoint
            and 'epoch' in checkpoint
        ):
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    best_cls1 = 0
    path = os.path.join('log',args.data_set+'_cls_best.pth')

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(pre_model, model, data_loader_train, optimizer, device, epoch, args.epochs,loss_scaler,
        args.clip_grad, model_ema, mixup_fn, set_training_mode=True,)

        lr_scheduler.step(epoch)
        test_stats = evaluate_cls(pre_model, model, data_loader_val, device, args=args, threshold_loc=0.5, epoch=epoch)

        is_best = test_stats["top1_cls"] > best_cls1
        best_cls1 = max(test_stats["top1_cls"], best_cls1)
        
        if is_best:
            best_cls5 =  test_stats["top5_cls"]
            
            print("Evaluation Result:\n"
                            "Loc Top:{0:.3f} Loc Top5:{1:.3f}\n".
                            format( best_cls1, best_cls5))
            utils.save_on_master({
                            'state_dict': model.state_dict()
                        },path,)
        
        if epoch+1 == args.epochs:
            
            test_stats = evaluate(data_loader_val, model, device, args, threshold_loc=0.1, epoch=epoch)
            print(
                f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
            )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'DeiT training and evaluation script', parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
