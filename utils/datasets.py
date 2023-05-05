from curses import mousemask
import os
from re import I
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import numpy as np
import random
import pdb
import cv2
import logging
import h5py
import albumentations as A
from albumentations.pytorch import ToTensorV2
from timm.data import create_transform

logger = logging.getLogger(__name__)
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def get_transforms():
    
    train_transform2 = create_transform(
        input_size=224,
        is_training=True,
        color_jitter=0.4,#0.4,
        auto_augment='rand-m9-mstd0.5-inc1',#'rand-m9-mstd0.5-inc1',
        interpolation='bicubic',
        re_prob=0,#,0.25,
        re_mode='pixel',
        re_count=1,
    )

    loc_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])
    
    cls_transform = transforms.Compose([
        transforms.Resize((256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])
    
    return train_transform2, loc_transform, cls_transform


class CUBFEATDataset(Dataset):
    def __init__(self, root, is_train):
        self.root = root
        self.is_train = is_train
        self.resize_size =  256
        self.loc_size = 256
        self.flip = True
        self.color_rgb = False

        self.heatmap_size = np.array([112, 112])

        self.image_list = self.remove_1st_column(open(
            os.path.join('datasets/CUB', 'images.txt'), 'r').readlines())
        self.label_list = self.remove_1st_column(open(
            os.path.join('datasets/CUB', 'image_class_labels.txt'), 'r').readlines())
        self.split_list = self.remove_1st_column(open(
            os.path.join('datasets/CUB', 'train_test_split.txt'), 'r').readlines())
       
        if self.is_train:
            self.index_list = self.get_index(self.split_list, '1')
        else:
            self.index_list = self.get_index(self.split_list, '0')
        
        if self.is_train:
            with h5py.File(os.path.join('datasets/DinoCAM/cub_grabcut_train.h5'),'r') as f:
                data = f['dataset']
                self.garbcut_numpy = data[:]

        else:
            with h5py.File(os.path.join('datasets/DinoCAM/cub_grabcut_val.h5'),'r') as f:
                data = f['dataset']
                self.target_numpy = data[:]
        self.train_transform,  self.loc_transform , self.cls_transform= get_transforms()

    def __getitem__(self, idx):
        name = self.image_list[self.index_list[idx]]
        image_path = os.path.join(self.root, name)
        label = int(self.label_list[self.index_list[idx]])-1
        target = self.generate_target(idx)

        image = Image.open(image_path).convert('RGB')
        if self.is_train: 
            
            image = image.resize((256, 256))

            lam = np.random.beta(1.0, 1.0)
            bbx1, bby1, bbx2, bby2 = rand_bbox((256,256), lam)
            target[bbx1:bbx2, bby1:bby2] = 1

            image = image * target[:,:,np.newaxis]
            image_ = image.astype(np.uint8)
            image_ = Image.fromarray(image_)
            input = self.train_transform(image_)
            return input, label, name

        else:
            cls_image = self.cls_transform(image)
            return cls_image, label, name

    def get_index(self, list, value):
        index = []
        for i in range(len(list)):
            if list[i] == value:
                index.append(i)
        return index
    
    def remove_1st_column(self, input_list):
        output_list = []
        for i in range(len(input_list)):
            if len(input_list[i][:-1].split(' '))==2:
                output_list.append(input_list[i][:-1].split(' ')[1])
            else:
                output_list.append(input_list[i][:-1].split(' ')[1:])
        return output_list

    def __len__(self):
        return len(self.index_list)

    def generate_target(self,idx):
        if self.is_train:
            cam_ori = torch.from_numpy(self.garbcut_numpy[idx])#torch.load(os.path.join(self.target_dir, name[:-4] +'.pth'))
            cam_ori = cam_ori.squeeze(0)
        else:
            cam_ori = torch.from_numpy(self.target_numpy[idx])#torch.load(os.path.join(self.target_dir, name[:-4] +'.pth'))

        cam_np = cv2.resize(cam_ori.numpy() , (256, 256))
        cam_np = cam_np.astype('float32')

        cam_min, cam_max = cam_np.min(), cam_np.max()
        target = (cam_np - cam_min) / (cam_max - cam_min)
        
        #try 0-1 distribution
        target[target>=0.05] = 1
        target[target<0.05] = 0

        return target
    
class ImageNetDataset(Dataset):
    def __init__(self, root, is_train=False):
        self.root = root
        self.is_train = is_train
      
        if self.is_train:
            datalist = os.path.join('datasets/IMNET', 'train.txt')
            self.image_dir = os.path.join(self.root, 'train')
        else:
            datalist = os.path.join('datasets/IMNET', 'val_folder_new.txt')
            self.image_dir = os.path.join(self.root, 'val_folder/val')

        if self.is_train:
            with h5py.File(os.path.join('datasets/DinoIMNET/imagenet_grabcut_train.h5'),'r') as f:
                data = f['dataset']
                self.garbcut_numpy = data[:]
        names = []
        labels = []
        bboxes = []
        with open(datalist) as f:
            for line in f:
                info = line.strip().split()
                names.append(info[0])
                labels.append(int(info[1]))
                if self.is_train is False:
                    bboxes.append(np.array(list(map(float, info[2:]))).reshape(-1,4))

        self.names = names
        self.labels = labels
        if self.is_train is False:
            self.bboxes = bboxes

        self.train_transform,  self.loc_transform , self.cls_transform= get_transforms()
        
    # For multiple bbox
    def __getitem__(self, idx):
        name = self.names[idx]
        label = self.labels[idx]

        image = Image.open(os.path.join(self.image_dir, name)).convert('RGB')

        if self.is_train:
            image = image.resize((256, 256))
            target = self.generate_target(idx)

            lam = np.random.beta(1.0, 1.0)
            bbx1, bby1, bbx2, bby2 = rand_bbox((256,256), lam)
            target[bbx1:bbx2, bby1:bby2] = 1

            image = image * target[:,:,np.newaxis]
            image_ = image.astype(np.uint8)
            image_ = Image.fromarray(image_)
            input = self.train_transform(image_)
            return input, label, name

        else:
            cls_image = self.cls_transform(image)
            return cls_image, label, name

    def __len__(self):
        return len(self.names)

    def generate_target(self,idx):
        target = self.garbcut_numpy[idx]
        target = cv2.resize(target, (256, 256))
        return target
    
def build_dataset(args):
    if args.data_set == 'CUB':
        dataset_train = CUBFEATDataset(root=args.data_path, is_train=True)
        dataset_val = CUBFEATDataset(root=args.data_path, is_train=False)
        nb_classes = 200

    elif args.data_set == 'IMNET':
        dataset_train = ImageNetDataset(root=args.data_path, is_train=True)
        dataset_val = ImageNetDataset(root=args.data_path, is_train=False)
        nb_classes = 1000
    return dataset_train, dataset_val, nb_classes

def rand_bbox(size, lam):
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2