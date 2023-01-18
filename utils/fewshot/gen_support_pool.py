#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 15:30:24 2020

@author: fanq15
"""

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import os.path as osp
import glob
import pandas as pd

from tqdm import tqdm
from os.path import join
from pathlib import Path


IMG_FORMATS = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo"]


def vis_image(im, bboxs, im_name):
    dpi = 300
    fig, ax = plt.subplots()
    ax.imshow(im, aspect='equal')
    plt.axis('off')
    height, width, channels = im.shape
    fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    # Show box (off by default, box_alpha=0.0)
    for bbox in bboxs:
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1],
                          fill=False, edgecolor='r',
                          linewidth=0.5, alpha=1))
    os.path.basename(im_name)
    plt.savefig(im_name, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close('all')


def crop_support(img, bbox, img_size):
    image_shape = img.shape[:2]  # h, w
    data_height, data_width = image_shape

    img = img.transpose(2, 0, 1)

    x1 = int((bbox[0] - bbox[2] / 2) * image_shape[1])
    x2 = int((bbox[0] + bbox[2] / 2) * image_shape[1])
    y1 = int((bbox[1] - bbox[3] / 2) * image_shape[0])
    y2 = int((bbox[1] + bbox[3] / 2) * image_shape[0])

    width = x2 - x1
    height = y2 - y1
    context_pixel = 128  # int(16 * im_scale)

    new_x1 = 0
    new_y1 = 0
    new_x2 = width
    new_y2 = height
    if not isinstance(img_size, list):
        img_size = (img_size, img_size)

    if width >= height:
        crop_x1 = x1 - context_pixel
        crop_x2 = x2 + context_pixel

        # New_x1 and new_x2 will change when crop context or overflow
        new_x1 = new_x1 + context_pixel
        new_x2 = new_x1 + width
        if crop_x1 < 0:
            new_x1 = new_x1 + crop_x1
            new_x2 = new_x1 + width
            crop_x1 = 0
        if crop_x2 > data_width:
            crop_x2 = data_width

        short_size = height
        long_size = crop_x2 - crop_x1
        y_center = int((y2 + y1) / 2)  # math.ceil((y2 + y1) / 2)
        # int(y_center - math.ceil(long_size / 2))
        crop_y1 = int(y_center - (long_size / 2))
        # int(y_center + math.floor(long_size / 2))
        crop_y2 = int(y_center + (long_size / 2))

        # New_y1 and new_y2 will change when crop context or overflow
        new_y1 = new_y1 + math.ceil((long_size - short_size) / 2)
        new_y2 = new_y1 + height
        if crop_y1 < 0:
            new_y1 = new_y1 + crop_y1
            new_y2 = new_y1 + height
            crop_y1 = 0
        if crop_y2 > data_height:
            crop_y2 = data_height

        crop_short_size = crop_y2 - crop_y1
        crop_long_size = crop_x2 - crop_x1
        square = np.zeros((3, crop_long_size, crop_long_size), dtype=np.uint8)
        # int(math.ceil((crop_long_size - crop_short_size) / 2))
        delta = int((crop_long_size - crop_short_size) / 2)
        square_y1 = delta
        square_y2 = delta + crop_short_size

        new_y1 = new_y1 + delta
        new_y2 = new_y2 + delta

        crop_box = img[:, crop_y1:crop_y2, crop_x1:crop_x2]
        square[:, square_y1:square_y2, :] = crop_box
    else:
        crop_y1 = y1 - context_pixel
        crop_y2 = y2 + context_pixel

        # New_y1 and new_y2 will change when crop context or overflow
        new_y1 = new_y1 + context_pixel
        new_y2 = new_y1 + height
        if crop_y1 < 0:
            new_y1 = new_y1 + crop_y1
            new_y2 = new_y1 + height
            crop_y1 = 0
        if crop_y2 > data_height:
            crop_y2 = data_height

        short_size = width
        long_size = crop_y2 - crop_y1
        x_center = int((x2 + x1) / 2)  # math.ceil((x2 + x1) / 2)
        # int(x_center - math.ceil(long_size / 2))
        crop_x1 = int(x_center - (long_size / 2))
        # int(x_center + math.floor(long_size / 2))
        crop_x2 = int(x_center + (long_size / 2))

        # New_x1 and new_x2 will change when crop context or overflow
        new_x1 = new_x1 + math.ceil((long_size - short_size) / 2)
        new_x2 = new_x1 + width
        if crop_x1 < 0:
            new_x1 = new_x1 + crop_x1
            new_x2 = new_x1 + width
            crop_x1 = 0
        if crop_x2 > data_width:
            crop_x2 = data_width

        crop_short_size = crop_x2 - crop_x1
        crop_long_size = crop_y2 - crop_y1
        square = np.zeros((3, crop_long_size, crop_long_size), dtype=np.uint8)
        # int(math.ceil((crop_long_size - crop_short_size) / 2))
        delta = int((crop_long_size - crop_short_size) / 2)
        square_x1 = delta
        square_x2 = delta + crop_short_size

        new_x1 = new_x1 + delta
        new_x2 = new_x2 + delta
        crop_box = img[:, crop_y1:crop_y2, crop_x1:crop_x2]
        square[:, :, square_x1:square_x2] = crop_box

    square = square.astype(np.float32, copy=False)
    square_scale = float(img_size[0]) / long_size
    square = square.transpose(1, 2, 0)
    # None, None, fx=square_scale, fy=square_scale, interpolation=cv2.INTER_LINEAR)
    square = cv2.resize(square, img_size, interpolation=cv2.INTER_LINEAR)
    square = square.astype(np.uint8)

    new_x1 = int(new_x1 * square_scale)
    new_y1 = int(new_y1 * square_scale)
    new_x2 = int(new_x2 * square_scale)
    new_y2 = int(new_y2 * square_scale)

    new_x = (new_x1 + new_x2) / 640
    new_y = (new_y1 + new_y2) / 640
    new_width = (new_x2 - new_x1) / 320
    new_height = (new_y2 - new_y1) / 320

    support_data = square
    support_box = np.array([new_x, new_y, new_width, new_height]).astype(np.float32)
    return support_data, support_box


def get_imgs_labels(img_dir):
    img_paths = glob.glob(osp.join(img_dir, "*"), recursive=True)
    img_paths = sorted(
        p for p in img_paths if p.split(".")[-1].lower() in IMG_FORMATS
    )
    assert img_paths, f"No images found in {img_dir}."

    # check and load anns
    label_dir = osp.join(
        osp.dirname(osp.dirname(img_dir)), "labels", osp.basename(img_dir)
    )
    assert osp.exists(label_dir), f"{label_dir} is an invalid directory path!"

    label_paths = sorted(
        osp.join(label_dir, osp.splitext(osp.basename(p))[0] + ".txt")
        for p in img_paths
    )
    all_labels = []
    for label_path in label_paths:
        with open(label_path, "r") as f:
            labels = [
                x.split() for x in f.read().strip().splitlines() if len(x)
            ]
            labels = np.array(labels, dtype=np.float32)
            all_labels.append(labels.tolist())
    return img_paths, all_labels


def gen_support_pool(img_dir, img_size=320):
    support_img_dir = Path(img_dir).parent / 'support'
    support_df_cache = \
        Path(img_dir).parent.parent / 'labels' / 'support.cache'
    if support_img_dir.exists() and support_df_cache.exists():
        return pd.read_pickle(support_df_cache)

    if not support_img_dir.exists():
        support_img_dir.mkdir()

    support_dict = {}

    support_dict['support_box'] = []
    support_dict['category_id'] = []
    support_dict['image_name'] = []
    support_dict['file_path'] = []

    img_paths, labels = get_imgs_labels(img_dir)

    for img_path, label in tqdm(zip(img_paths, labels)):
        if len(label) == 0:
            continue

        img_path = Path(img_path)
        img_name = img_path.name
        im = cv2.imread(str(img_path))
        # im = cv2.resize(im, (640, 640), interpolation=cv2.INTER_LINEAR)
        for item in label:
            category_id = int(item[0])
            # x_cen, y_cen, x_len, y_len
            ann = item[1:]
            support_img, support_box = crop_support(im, ann, img_size)
            file_path = join(support_img_dir,
                             '{}_{:04d}.jpg'.format(
                                 img_name.replace(
                                     img_path.suffix, ''), category_id
                             ))
            cv2.imwrite(file_path, support_img)
            # cv2.imwrite(file_path, im)
            support_dict['support_box'].append(support_box)
            # support_dict['support_box'].append(ann)
            support_dict['category_id'].append(category_id)
            support_dict['image_name'].append(img_name)
            support_dict['file_path'].append(file_path)

    support_df = pd.DataFrame.from_dict(support_dict)

    support_df.to_pickle(support_df_cache)
    return support_df
