#!/usr/bin/env python
'''
===============================================================================
Interactive Image Segmentation using GrabCut algorithm.

This sample shows interactive image segmentation using grabcut algorithm.

USAGE:
    python grabcut.py <filename>

README FIRST:
    Two windows will show up, one for input and one for output.

    At first, in input window, draw a rectangle around the object using
mouse right button. Then press 'n' to segment the object (once or a few times)
For any finer touch-ups, you can press any of the keys below and draw lines on
the areas you want. Then again press 'n' for updating the output.

Key '0' - To select areas of sure background
Key '1' - To select areas of sure foreground
Key '2' - To select areas of probable background
Key '3' - To select areas of probable foreground

Key 'n' - To update the segmentation
Key 'r' - To reset the setup
Key 's' - To save the results
===============================================================================
'''

import sys
import numpy as np
import cv2 as cv
import igraph as ig
from utils.grabcut.GMM import GaussianMixture


RED = [0, 0, 255]         # PR BG
GREEN = [0, 255, 0]       # PR FG
BLACK = [0, 0, 0]         # sure BG
WHITE = [255, 255, 255]   # sure FG

DRAW_BG = {'color': BLACK, 'val': 0}
DRAW_FG = {'color': WHITE, 'val': 1}
DRAW_PR_FG = {'color': GREEN, 'val': 3}
DRAW_PR_BG = {'color': RED, 'val': 2}

# setting up flags

class GrabCut:
    def __init__(self, img, mask, rect=None, gmm_components=5):
        self.img = np.asarray(img, dtype=np.float64)
        self.rows, self.cols, _ = img.shape

        self.mask = mask # * mask初始值都为0
        if rect is not None:
            self.mask[rect[1]:rect[1] + rect[3],
                      rect[0]:rect[0] + rect[2]] = DRAW_PR_FG['val'] # * rect 框的都看作是可能的前景，所以这个框必须最大限度去掉背景
        # * 当前mask 有背景和可能的前景
        self.classify_pixels() #* 统计前景和背景的点数

        # Best number of GMM components K suggested in paper
        self.gmm_components = gmm_components
        self.gamma = 50  # Best gamma suggested in paper formula (5)
        self.beta = 0

        self.left_V = np.empty((self.rows, self.cols - 1)) #* (256, 255)
        self.upleft_V = np.empty((self.rows - 1, self.cols - 1))
        self.up_V = np.empty((self.rows - 1, self.cols))
        self.upright_V = np.empty((self.rows - 1, self.cols - 1))

        self.bgd_gmm = None
        self.fgd_gmm = None
        self.comp_idxs = np.empty((self.rows, self.cols), dtype=np.uint32) #* (256,256)

        self.gc_graph = None
        self.gc_graph_capacity = None           #* Edge capacities
        self.gc_source = self.cols * self.rows  #* "object" 256*256
        self.gc_sink = self.gc_source + 1       #* "background" 

        self.calc_beta_smoothness()
        self.init_GMMs()
        self.run()

    def calc_beta_smoothness(self):
        _left_diff = self.img[:, 1:] - self.img[:, :-1]
        _upleft_diff = self.img[1:, 1:] - self.img[:-1, :-1]
        _up_diff = self.img[1:, :] - self.img[:-1, :]
        _upright_diff = self.img[1:, :-1] - self.img[:-1, 1:]

        self.beta = np.sum(np.square(_left_diff)) + np.sum(np.square(_upleft_diff)) + \
            np.sum(np.square(_up_diff)) + \
            np.sum(np.square(_upright_diff))
        self.beta = 1 / (2 * self.beta / (
            # Each pixel has 4 neighbors (left, upleft, up, upright)
            4 * self.cols * self.rows
            # The 1st column doesn't have left, upleft and the last column doesn't have upright
            - 3 * self.cols
            - 3 * self.rows  # The first row doesn't have upleft, up and upright
            + 2))  # The first and last pixels in the 1st row are removed twice
        # print('Beta:', self.beta)

        # Smoothness term V described in formula (11)
        self.left_V = self.gamma * np.exp(-self.beta * np.sum(
            np.square(_left_diff), axis=2))
        self.upleft_V = self.gamma / np.sqrt(2) * np.exp(-self.beta * np.sum(
            np.square(_upleft_diff), axis=2))
        self.up_V = self.gamma * np.exp(-self.beta * np.sum(
            np.square(_up_diff), axis=2))
        self.upright_V = self.gamma / np.sqrt(2) * np.exp(-self.beta * np.sum(
            np.square(_upright_diff), axis=2))

    def classify_pixels(self):
        self.bgd_indexes = np.where(np.logical_or( #logical_or(x1,x2)有true，则true，np.where 返回index
            self.mask == DRAW_BG['val'], self.mask == DRAW_PR_BG['val']))
        
        self.fgd_indexes = np.where(np.logical_or(
            self.mask == DRAW_FG['val'], self.mask == DRAW_PR_FG['val']))

        assert self.bgd_indexes[0].size > 0
        assert self.fgd_indexes[0].size > 0

        # print('(pr_)bgd count: %d, (pr_)fgd count: %d' % (
        #     self.bgd_indexes[0].size, self.fgd_indexes[0].size))

    def init_GMMs(self):
        # print(self.img[self.bgd_indexes].shape)
        self.bgd_gmm = GaussianMixture(self.img[self.bgd_indexes])
        self.fgd_gmm = GaussianMixture(self.img[self.fgd_indexes])

    def assign_GMMs_components(self): #* 判定每个pixel属于哪个component,在comp_idxs上对应位置标明component的number
        """Step 1 in Figure 3: Assign GMM components to pixels"""
        self.comp_idxs[self.bgd_indexes] = self.bgd_gmm.which_component(
            self.img[self.bgd_indexes])
        
        self.comp_idxs[self.fgd_indexes] = self.fgd_gmm.which_component(
            self.img[self.fgd_indexes])

    def learn_GMMs(self): #* fit(X, labels)
        """Step 2 in Figure 3: Learn GMM parameters from data z"""
        self.bgd_gmm.fit(self.img[self.bgd_indexes], self.comp_idxs[self.bgd_indexes])

        self.fgd_gmm.fit(self.img[self.fgd_indexes], self.comp_idxs[self.fgd_indexes])

    def construct_gc_graph(self):
        bgd_indexes = np.where(self.mask.reshape(-1) == DRAW_BG['val'])
        fgd_indexes = np.where(self.mask.reshape(-1) == DRAW_FG['val'])
        
        #* pr_indexes 是指 uncertain pixel
        pr_indexes = np.where(np.logical_or(
            self.mask.reshape(-1) == DRAW_PR_BG['val'], self.mask.reshape(-1) == DRAW_PR_FG['val']))

        # print('bgd count: %d, fgd count: %d, uncertain count: %d' % (
        #     len(bgd_indexes[0]), len(fgd_indexes[0]), len(pr_indexes[0])))

        #* gc_source: 256*256
        #* pr_indexes: uncertain points (array([12830, 12831, 12832, ..., 46031, 46032, 46033]),)

        edges = []
        self.gc_graph_capacity = []

        # t-links
        edges.extend(list(zip([self.gc_source] * pr_indexes[0].size, pr_indexes[0])))
        # a = [self.gc_source] * pr_indexes[0].size
        
        # print("a:",len(a))
        _D = -np.log(self.bgd_gmm.calc_prob(self.img.reshape(-1, 3)[pr_indexes])) #* 计算 uncertain 的概率
        self.gc_graph_capacity.extend(_D.tolist()) 
        assert len(edges) == len(self.gc_graph_capacity)

        edges.extend(
            list(zip([self.gc_sink] * pr_indexes[0].size, pr_indexes[0])))
        _D = -np.log(self.fgd_gmm.calc_prob(self.img.reshape(-1, 3)[pr_indexes]))
        self.gc_graph_capacity.extend(_D.tolist())
        assert len(edges) == len(self.gc_graph_capacity)

        edges.extend(
            list(zip([self.gc_source] * bgd_indexes[0].size, bgd_indexes[0])))
        self.gc_graph_capacity.extend([0] * bgd_indexes[0].size)
        assert len(edges) == len(self.gc_graph_capacity)

        edges.extend(
            list(zip([self.gc_sink] * bgd_indexes[0].size, bgd_indexes[0])))
        self.gc_graph_capacity.extend([9 * self.gamma] * bgd_indexes[0].size)
        assert len(edges) == len(self.gc_graph_capacity)

        edges.extend(
            list(zip([self.gc_source] * fgd_indexes[0].size, fgd_indexes[0])))
        self.gc_graph_capacity.extend([9 * self.gamma] * fgd_indexes[0].size)
        assert len(edges) == len(self.gc_graph_capacity)

        edges.extend(
            list(zip([self.gc_sink] * fgd_indexes[0].size, fgd_indexes[0])))
        self.gc_graph_capacity.extend([0] * fgd_indexes[0].size)
        assert len(edges) == len(self.gc_graph_capacity)
        #* self.gc_graph_capacity -> 256*256*2

        # n-links
        #* 2.1
        img_indexes = np.arange(self.rows * self.cols, #* [[1,2,3...256*256]]
                                dtype=np.uint32).reshape(self.rows, self.cols)
        mask1 = img_indexes[:, 1:].reshape(-1) # (256, 255) -> (65280,)
        mask2 = img_indexes[:, :-1].reshape(-1) # (256,255) -> (62580,)
        edges.extend(list(zip(mask1, mask2)))
        self.gc_graph_capacity.extend(self.left_V.reshape(-1).tolist())#self.left_V = (256, 255)->(65280,)
        assert len(edges) == len(self.gc_graph_capacity)

        #* 2.2
        mask1 = img_indexes[1:, 1:].reshape(-1) #(255, 255) -> (65025,)
        mask2 = img_indexes[:-1, :-1].reshape(-1)#(255, 255) -> (65025,)
        edges.extend(list(zip(mask1, mask2)))
        self.gc_graph_capacity.extend(self.upleft_V.reshape(-1).tolist())
        assert len(edges) == len(self.gc_graph_capacity)

        #* 2.3
        mask1 = img_indexes[1:, :].reshape(-1) # (255, 256) -> (65280,)
        mask2 = img_indexes[:-1, :].reshape(-1) # (255, 256) -> (65280,)
        edges.extend(list(zip(mask1, mask2)))
        self.gc_graph_capacity.extend(self.up_V.reshape(-1).tolist())
        assert len(edges) == len(self.gc_graph_capacity)
        
        #* 2.4
        mask1 = img_indexes[1:, :-1].reshape(-1) #(255, 255) -> (65025,)
        mask2 = img_indexes[:-1, 1:].reshape(-1) #(255, 255) -> (65025,)
        edges.extend(list(zip(mask1, mask2)))

        self.gc_graph_capacity.extend(self.upright_V.reshape(-1).tolist()) #upright_V: (255,255)
        # print(len(edges), self.cols, self.rows)
        # print( 4 * self.cols * self.rows - 3 * (self.cols + self.rows) + 2 + 2 * self.cols * self.rows)
        
        assert len(edges) == len(self.gc_graph_capacity)
        assert len(edges) == 4 * self.cols * self.rows - 3 * (self.cols + self.rows) + 2 + \
            2 * self.cols * self.rows

        self.gc_graph = ig.Graph(self.cols * self.rows + 2)
        self.gc_graph.add_edges(edges)

    def estimate_segmentation(self):
        """Step 3 in Figure 3: Estimate segmentation"""
        mincut = self.gc_graph.st_mincut(
            self.gc_source, self.gc_sink, self.gc_graph_capacity)
        # print('foreground pixels: %d, background pixels: %d' % (len(mincut.partition[0]), len(mincut.partition[1])))
        pr_indexes = np.where(np.logical_or(
            self.mask == DRAW_PR_BG['val'], self.mask == DRAW_PR_FG['val']))
        img_indexes = np.arange(self.rows * self.cols,
                                dtype=np.uint32).reshape(self.rows, self.cols)
        self.mask[pr_indexes] = np.where(np.isin(img_indexes[pr_indexes], mincut.partition[0]),
                                         DRAW_PR_FG['val'], DRAW_PR_BG['val'])
        self.classify_pixels()

    def run(self, num_iters=1, skip_learn_GMMs=False):
        # print('skip learn GMMs:', skip_learn_GMMs)
        for _ in range(num_iters):
            if not skip_learn_GMMs:
                self.assign_GMMs_components()
                self.learn_GMMs()
            self.construct_gc_graph()
            self.estimate_segmentation()
            skip_learn_GMMs = False
            # print('data term: %f, smoothness term: %f, total energy: %f' % self.calc_energy())
import h5py
import os
import torch
def read_mask(idx=0):
    with h5py.File(os.path.join('files/cam_train.h5'),'r') as f:
                data = f['dataset']
                target_numpy = data[:]
    cam_ori = torch.from_numpy(target_numpy[idx])#torch.load(os.path.join(self.target_dir, name[:-4] +'.pth'))

    cam = torch.mean(cam_ori[:3,:], dim=0, keepdim=False)
    cam_np = cam
    cam_min, cam_max = cam_np.min(), cam_np.max()
    target = (cam_np - cam_min) / (cam_max - cam_min)
    return target

def read_txt(fine_name):
    input_list = open(os.path.join(fine_name), 'r').readlines()
    output_list = { }
    for i in range(len(input_list)):
        name = input_list[i][:-1].split(' ')[1]
        output_list[i] = name
    return output_list

if __name__ == '__main__':
    ppd_mask = 'ppd_mask'
    refined_mask = 'refined_mask'

    if not os.path.isdir(ppd_mask):
        os.makedirs(ppd_mask)

    if not os.path.isdir(refined_mask):
        os.makedirs(refined_mask)

    output_list = read_txt('files/train2.txt')
    root = '/GPUFS/nsccgz_ywang_zfd/caoxz/data/CUB_200_2011/images'
    for idx, name in output_list.items():
        # print(idx, name)
        mask = read_mask(idx=int(idx))
        mask = cv.resize(mask.numpy(), (256, 256))
        mask[mask>=0.1] = 1
        mask[mask<0.03] = 0
        # cv.imwrite(os.path.join(ppd_mask, name.split("/")[-1]),mask*255.0)
        
        pos = np.logical_and(mask >=0.03, mask <0.1)
        mask[pos.data] = 2
        img = cv.imread(os.path.join(root,name))
        img = cv.resize(img, (256, 256))

        gc = GrabCut(img, mask, None)
        mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
        cv.imwrite(os.path.join(refined_mask,name.split("/")[-1]), mask2*255.0)

    print("++++++=====done=====++++++")


