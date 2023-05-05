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
    train_transform = A.Compose([
            A.Normalize(),
            A.Resize(256, 256),
            A.RandomCrop(height=224, width=224),
            A.HorizontalFlip(p=0.5),
            ToTensorV2()
        ])
    loc_transformer = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])
    
    return train_transform, loc_transformer, val_transform


class CUBFEATDataset(Dataset):
    def __init__(self, root, is_train):
        print(root)
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
        self.train_transform, self.loc_transform, self.val_transform = get_transforms()

    # For multiple bbox
    def __getitem__(self, idx):
        name = self.image_list[self.index_list[idx]]
        image_path = os.path.join(self.root,name)
        label = int(self.label_list[self.index_list[idx]])-1
        target = self.generate_target(idx)

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
            width, height = image.size

            image = self.val_transform(image)

            bbox = self.bbox_list[self.index_list[idx]]
            bbox = [int(float(value)) for value in bbox]
            # [x, y, bbox_width, bbox_height] = bbox
            # resize_size = 224
            # crop_size = 224
            # shift_size = 0
            # # shift_size = (resize_size - crop_size) // 2
            
            # left_bottom_x = int(max(x / image_width * resize_size - shift_size, 0))
            # left_bottom_y = int(max(y / image_height * resize_size - shift_size, 0))

            # right_top_x = int(min((x + bbox_width) / image_width * resize_size - shift_size, crop_size - 1))
            # right_top_y = int(min((y + bbox_height) / image_height * resize_size - shift_size, crop_size - 1))
            
            # gt_bbox = np.array([left_bottom_x, left_bottom_y, right_top_x, right_top_y]).reshape(-1)
            # gt_bbox = " ".join(list(map(str, gt_bbox)))

            resize_min = 256.0
            if width > height:
                n_width = width * resize_min / height
                n_height = resize_min
            else:
                n_width = resize_min
                n_height = height * resize_min / width
            bbox[0] = bbox[0] / width * n_width
            bbox[1] = bbox[1] / height * n_height
            bbox[2] = bbox[2] / width * n_width
            bbox[3] = bbox[3] / height * n_height
            crop_wh = 224
            temp_crop_x = int(round((n_width - crop_wh + 1) / 2.0))
            temp_crop_y = int(round((n_height - crop_wh + 1) / 2.0))
            bbox[0] -= temp_crop_x
            bbox[1] -= temp_crop_y
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            gt_bbox = np.clip(np.array(bbox), 0, crop_wh)

            # target = cv2.resize(target, (self.heatmap_size[0], self.heatmap_size[1]))
            # target = torch.from_numpy(target).unsqueeze(0)
            return image, label, gt_bbox, name

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
            with h5py.File(os.path.join('datasets/DinoIMNET/imagenet_train.h5'),'r') as f:
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

        self.top1top5_list = []
        self.train_transform, self.loc_transform, self.val_transform = get_transforms()
        
    # For multiple bbox
    def __getitem__(self, idx):
        name = self.names[idx]
        label = self.labels[idx]
        image_path = os.path.join(self.image_dir, name)

        if self.is_train:
            target = self.generate_target(idx)
            image            = cv2.imread(image_path)
            image            = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pair             = self.train_transform(image=image, mask=target)
            input, target      = pair['image'], pair['mask']
            target = cv2.resize(target.numpy(), (56, 56))        
            return input, target, label, name

        else:
            image = Image.open(image_path).convert('RGB')
            image_width, image_height = image.size
            image = self.val_transform(image)

            bbox = self.bboxes[idx]
            # x_min, y_min, x_max, y_max = bbox[0]
            # resize_size = 224
            # crop_size = 224
            # shift_size = 0

            # left_bottom_x = int(max(x_min / image_width * resize_size - shift_size, 0))
            # left_bottom_y = int(max(y_min / image_height * resize_size - shift_size, 0))
            # right_top_x = int(min(x_max / image_width * resize_size - shift_size, crop_size-1))
            # right_top_y = int(min(y_max / image_height * resize_size - shift_size, crop_size-1))
            # gt_bbox = np.clip(np.array([left_bottom_x, left_bottom_y, right_top_x, right_top_y]), 0, crop_size)
            
            #* For evaluation
            bbox = bbox[0]
            bbox[2] = bbox[2] - bbox[0]
            bbox[3] = bbox[3] - bbox[1]            
            resize_min = 256.0
            if image_width > image_height:
                n_width = image_width * resize_min / image_height
                n_height = resize_min
            else:
                n_width = resize_min
                n_height = image_height * resize_min / image_width
            bbox[0] = bbox[0] / image_width * n_width
            bbox[1] = bbox[1] / image_height * n_height
            bbox[2] = bbox[2] / image_width * n_width
            bbox[3] = bbox[3] / image_height * n_height
            crop_wh = 224
            temp_crop_x = int(round((n_width - crop_wh + 1) / 2.0))
            temp_crop_y = int(round((n_height - crop_wh + 1) / 2.0))
            bbox[0] -= temp_crop_x
            bbox[1] -= temp_crop_y
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            gt_bbox = np.clip(np.array(bbox), 0, crop_wh)
           
            return image, label, gt_bbox, name

    def __len__(self):
        return len(self.names)

    def generate_target(self,idx):
        # cam_ori = torch.from_numpy(self.garbcut_numpy[idx])
        # cam_ori = self.garbcut_numpy[idx]
        cam_ori = torch.from_numpy(self.garbcut_numpy[idx])#torch.load(os.path.join(self.target_dir, name[:-4] +'.pth'))
        cam = torch.mean(cam_ori[:3,:], dim=0, keepdim=False)
        #print(cam.max())
        cam_np = cv2.resize(cam.numpy() , (256, 256))
        cam_min, cam_max = cam_np.min(), cam_np.max()
        target = (cam_np - cam_min) / (cam_max - cam_min)
        #try 0-1 distribution
        target[target>=0.05] = 1
        target[target<0.05] = 0
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