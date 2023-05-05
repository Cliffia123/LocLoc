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
#     train_transform2 = A.Compose([
#             A.Normalize(),
#             A.Resize(256, 256),
#             A.CenterCrop(224, 224),
#             A.HorizontalFlip(p=0.5),
#             ToTensorV2()
#         ])
    train_transform2 = A.Compose([
            A.Normalize(),
            A.Resize(256, 256),
            A.RandomCrop(height=224, width=224),
            A.HorizontalFlip(p=0.5),
            ToTensorV2()
        ])
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
        self.bbox_list = self.remove_1st_column(open(
            os.path.join('datasets/CUB', 'bounding_boxes.txt'), 'r').readlines())

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

    # For multiple bbox
    def __getitem__(self, idx):
        name = self.image_list[self.index_list[idx]]
        image_path = os.path.join(self.root, name)
        label = int(self.label_list[self.index_list[idx]])-1
        target = self.generate_target(idx)

        image = Image.open(image_path).convert('RGB')

        if self.is_train: 
            image            = cv2.imread(image_path)
            image            = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pair             = self.train_transform(image=image, mask=target)
            input, target      = pair['image'], pair['mask']
            target = cv2.resize(target.numpy(), (112, 112))
            return input, target, label, name

        else:
            image = Image.open(image_path).convert('RGB')
            image_width, image_height = image.size

            input = self.loc_transform(image)
            cls_image = self.cls_transform(image)

            bbox = self.bbox_list[self.index_list[idx]]
            bbox = [int(float(value)) for value in bbox]
            [x, y, bbox_width, bbox_height] = bbox

            resize_size = 224
            shift_size = 0
            left_bottom_x = int(max(x / image_width * resize_size - shift_size, 0))
            left_bottom_y = int(max(y / image_height * resize_size - shift_size, 0))
            right_top_x = int(min((x + bbox_width) / image_width * resize_size - shift_size, resize_size - 1))
            right_top_y = int(min((y + bbox_height) / image_height * resize_size - shift_size, resize_size - 1))

            gt_bbox = np.array([left_bottom_x, left_bottom_y, right_top_x, right_top_y]).reshape(-1)
            gt_bbox = " ".join(list(map(str, gt_bbox)))

            target = cv2.resize(target, (self.heatmap_size[0], self.heatmap_size[1]))
            target = torch.from_numpy(target).unsqueeze(0)
            
            return input, cls_image, target, label, gt_bbox, name

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
#             target = cam_ori.numpy()
        else:
            cam_ori = torch.from_numpy(self.target_numpy[idx])#torch.load(os.path.join(self.target_dir, name[:-4] +'.pth'))
#             target = cam_ori.numpy()
#         cam_ori = torch.mean(cam_ori[:3,:], dim=0, keepdim=False)
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

        else:
            with h5py.File(os.path.join('datasets/DinoIMNET/imagenet_grabcut_val.h5'),'r') as f:
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
        target = self.generate_target(idx)

        image = Image.open(os.path.join(self.image_dir, name)).convert('RGB')
        image_size = list(image.size)

        if self.is_train:
            loc_image = self.loc_transform(image)
            return loc_image, target, label, name

        else:
            loc_image = self.loc_transform(image)
            cls_image = self.cls_transform(image)
            bbox = self.bboxes[idx]
            [x1, y1, x2, y2] = np.split(bbox, 4, 1)

            resize_size = 224
            loc_size = 224
            shift_size = 0
            [image_width, image_height] = image_size
            left_bottom_x = np.maximum(x1 / image_width * resize_size - shift_size, 0).astype(int)
            left_bottom_y = np.maximum(y1 / image_height * resize_size - shift_size, 0).astype(int)
            right_top_x = np.minimum(x2 / image_width * resize_size - shift_size, loc_size - 1).astype(int)
            right_top_y = np.minimum(y2 / image_height * resize_size - shift_size, loc_size - 1).astype(int)

            gt_bbox = np.concatenate((left_bottom_x, left_bottom_y, right_top_x, right_top_y),axis=1).reshape(-1)
            gt_bbox = " ".join(list(map(str, gt_bbox)))
            target = torch.from_numpy(target).unsqueeze(0)

            return loc_image, cls_image, target, label, gt_bbox, name

    def __len__(self):
        return len(self.names)

    def generate_target(self,idx):
        cam_ori = torch.from_numpy(self.garbcut_numpy[idx])

        if not self.is_train:
            cam_np = cv2.resize(cam_ori.numpy() , (56, 56))

        return cam_ori

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