from operator import gt
import numpy as np
import torch
from numpy.lib.function_base import blackman
from skimage import measure

import os
import cv2
import random
import colorsys
import pdb
from skimage import measure
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def cam_gen_box(scoremap_image, threshold):
    scoremap_image = scoremap_image * 255.0
    _CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0
    scoremap_image = np.expand_dims((scoremap_image).astype(np.uint8), 2)
    _, thr_gray_heatmap = cv2.threshold(
        src=scoremap_image,
        thresh=int(threshold * np.max(scoremap_image)),
        maxval=255,
        type=cv2.THRESH_BINARY,
    )
    contours = cv2.findContours(
        image=thr_gray_heatmap, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
    )[_CONTOUR_INDEX]

    if len(contours) == 0:
        return [0, 0, 224, 224]

    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    return [x, y, w + x, h + y]


def resize_cam(cam, size=(224, 224), mask=False):
    cam = cv2.resize(cam, (size[0], size[1]))
    cam_min, cam_max = cam.min(), cam.max()
    cam = (cam - cam_min) / (cam_max - cam_min)
    return cam

def intensity_to_rgb(intensity, cmap='cubehelix', normalize=False):
    assert intensity.ndim == 2, intensity.shape
    intensity = intensity.astype("float")

    if normalize:
        intensity -= intensity.min()
        intensity /= intensity.max()
    cmap = 'jet'
    cmap = plt.get_cmap(cmap)
    intensity = cmap(intensity)[..., :3]
    return intensity.astype('float32') * 255.0

def return_box_cam(cam, image, gt_box, th=0.5):
    # print(gt_box, len(gt_box))
    # print(gt_box)
    cam = resize_cam(cam, size=(224, 224))
    heatmap = intensity_to_rgb(cam, normalize=True).astype('uint8')
    heatmap_BGR = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    blend = image * 0.4 + heatmap_BGR * 0.6 # type: ignore
   
    box = np.array(get_bboxes(cam, th))
    cv2.rectangle(blend, # Green box
                        (box[0], box[1]),
                        (box[2], box[3]),
                        (0, 255, 0), 2)
                        
    cv2.rectangle(blend, # Red box
                        (int(gt_box[0]), int(gt_box[1])),
                        (int(gt_box[2]), int(gt_box[3])),
                        (0, 0, 255), 2)

    return box, blend


def norm_atten_map(attn_map):
    min_val = np.min(attn_map)
    max_val = np.max(attn_map)
    attn_norm = (attn_map - min_val) / (max_val - min_val + 1e-15)
    if max_val - min_val == 0:
        return np.zeros_like(attn_map)
    return attn_norm


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = (
            image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
        )
    return image


def display_instances(
    image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5
):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask, (10, 10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print("{name} saved.")
    return


def re_pca_box(heatmap, size=224):
    boxes = []
    if heatmap.ndim == 3:
        for he in heatmap:
            highlight = np.zeros(he.shape)
            highlight[he > 0] = 1
            # max component
            all_labels = measure.label(highlight)
            highlight = np.zeros(highlight.shape)
            highlight[all_labels == count_max(all_labels.tolist())] = 1

            # visualize heatmap
            # show highlight in origin image
            highlight = np.round(highlight * 255)
            highlight_big = cv2.resize(
                highlight, (size, size), interpolation=cv2.INTER_NEAREST
            )
            props = measure.regionprops(highlight_big.astype(int))
            if len(props) == 0:
                bbox = [0, 0, size, size]
            else:
                temp = props[0]['bbox']
                bbox = [temp[1], temp[0], temp[3], temp[2]]
            temp_bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
            boxes.append(temp_bbox)
        return boxes  # x,y,w,h
    else:
        highlight = np.zeros(heatmap.shape)
        highlight[heatmap > 0] = 1
        # max component
        all_labels = measure.label(highlight)
        highlight = np.zeros(highlight.shape)
        highlight[all_labels == count_max(all_labels.tolist())] = 1

        # visualize heatmap
        # show highlight in origin image
        highlight = np.round(highlight * 255)
        highlight_big = cv2.resize(
            highlight, (size, size), interpolation=cv2.INTER_NEAREST
        )
        props = measure.regionprops(highlight_big.astype(int))
        if len(props) == 0:
            bbox = [0, 0, size, size]
        else:
            temp = props[0]['bbox']
            bbox = [temp[1], temp[0], temp[3], temp[2]]
        return bbox


def count_max(x):
    count_dict = {}
    for xlist in x:
        for item in xlist:
            if item == 0:
                continue
            if item not in count_dict.keys():
                count_dict[item] = 0
            count_dict[item] += 1
    if count_dict == {}:
        return -1
    count_dict = sorted(count_dict.items(), key=lambda d: d[1], reverse=True)
    return count_dict[0][0]

def cal_iou(box1, box2, method='iou'):
    """
    support:
    1. box1 and box2 are the same shape: [N, 4]
    2.
    :param box1:
    :param box2:
    :return:
    """
    box1 = np.asarray(box1, dtype=float)
    box2 = np.asarray(box2, dtype=float)
    if box1.ndim == 1:
        box1 = box1[np.newaxis, :]
    if box2.ndim == 1:
        box2 = box2[np.newaxis, :]

    iw = np.minimum(box1[:, 2], box2[:, 2]) - np.maximum(box1[:, 0], box2[:, 0]) + 1
    ih = np.minimum(box1[:, 3], box2[:, 3]) - np.maximum(box1[:, 1], box2[:, 1]) + 1

    i_area = np.maximum(iw, 0.0) * np.maximum(ih, 0.0)
    box1_area = (box1[:, 2] - box1[:, 0] + 1) * (box1[:, 3] - box1[:, 1] + 1)
    box2_area = (box2[:, 2] - box2[:, 0] + 1) * (box2[:, 3] - box2[:, 1] + 1)

    if method == 'iog':
        iou_val = i_area / (box2_area)
    elif method == 'iob':
        iou_val = i_area / (box1_area)
    else:
        iou_val = i_area / (box1_area + box2_area - i_area)
    return iou_val

def get_bboxes(cam, cam_thr=0.5):
    """
    cam: single image with shape (h, w, 1)
    thr_val: float value (0~1)
    return estimated bounding box
    """
    #pdb.set_trace()
    cam = (cam * 255.).astype(np.uint8)
    map_thr = cam_thr * np.max(cam)

    _, thr_gray_heatmap = cv2.threshold(cam,
                                        int(map_thr), 255,
                                        cv2.THRESH_TOZERO)

    contours, _ = cv2.findContours(thr_gray_heatmap,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        estimated_bbox = [x, y, x + w, y + h]
    else:
        estimated_bbox = [0, 0, 1, 1]

    return estimated_bbox  #, thr_gray_heatmap, len(contours)


def accuracy_loc( input, output, bbox_label, is_Train = True, thr=0.5,size=224):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    
    cnt = input.shape[0]
    acc = np.zeros(cnt)
    tar_acc = np.zeros(cnt)
    pd_bbox = []
    for b in range(cnt):
        out_b = output[b,0, :, :] 
        out_b = cv2.resize(out_b , (size, size))
        cam_min, cam_max = out_b.min(), out_b.max()
        out_b = (out_b - cam_min) / (cam_max - cam_min)
        out_bbox = np.array(get_bboxes(out_b, cam_thr=thr) )
        pd_bbox.append(out_bbox)

        if not is_Train:
            max_iou = 0
            gt_bbox = bbox_label[b].cpu()
            gt_bbox = list(map(float, gt_bbox))
            gt_box_cnt = len(gt_bbox) // 4
            for i in range(gt_box_cnt):
                gt_box = gt_bbox[i * 4:(i + 1) * 4]
                iou_i = cal_iou(out_bbox, gt_box)
                if iou_i > max_iou:
                    max_iou = iou_i
            acc[b] = max_iou
    return acc, cnt,  

