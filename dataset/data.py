import math
import os
import random
import time

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
import numpy as np
import cv2
from warnings import warn
import copy

# TODO: train_test_split should be random


def show_img_iter(images, size=20, gray=True):
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(nrows=1, ncols=images.shape[0], figsize=(size, size))
    for j in range(images.shape[0]):
        if gray:
            if len(images[j].shape) == 2:
                axes[j].imshow(images[j], cmap='gray')
            elif images[j].shape[0] <= 3:
                axes[j].imshow(images[j][0], cmap='gray')
        elif images[j].shape[0] == 3:
            to_pil = torchvision.transforms.ToPILImage()
            axes[j].imshow(to_pil(images[j]))


def get_empty(real, gray=True):
    path = r'data\r_empty.png' if real else r'data\empty.png'
    if gray:
        return read_image(path, mode=ImageReadMode.GRAY).squeeze()
    else:
        return read_image(path, mode=ImageReadMode.RGB)


def bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def get_square_cropped_img_iter(images, real: bool, pad=0, empty_img=None):
    img_iter_long = np.zeros(shape=(images.shape[0] + 1, images.shape[1], images.shape[2]))

    if empty_img is None:
        img_iter_long[0] = get_empty(real).numpy()
    else:
        img_iter_long[0] = empty_img
    img_iter_long[1:] = images

    # bg_sub_iter = np.zeros(img_iter_long.shape)
    cropped_images = []
    bg_sub = cv2.createBackgroundSubtractorMOG2(varThreshold=16)

    bg_sub.apply(img_iter_long[0])
    for i in range(images.shape[0]):
        mask = bg_sub.apply(img_iter_long[i + 1])
        rmin, rmax, cmin, cmax = bounding_box(mask)
        diff = (cmax - cmin) - (rmax - rmin)
        if diff > 0:
            rmax += math.ceil(float(abs(diff)) / 2)
            rmin -= math.floor(float(abs(diff)) / 2)
        elif diff < 0:
            cmax += math.ceil(float(abs(diff)) / 2)
            cmin -= math.floor(float(abs(diff)) / 2)
        assert cmax - cmin == rmax - rmin
        rmin = max(0, rmin - pad)
        rmax = min(mask.shape[0], rmax + pad)
        cmin = max(0, cmin - pad)
        cmax = min(mask.shape[1], cmax + pad)

        cropped_images.append(torch.from_numpy(images[i][rmin:rmax, cmin:cmax]))

    return cropped_images


class PartDataset(Dataset):
    def __init__(self, classes, img_dir, pics_per_iter, transform_load, transform_get=None, bg_subtract_renders=False, render_crop_pad=0,
                 bg_subtract_real=True, real_crop_pad=10, use_cache=False, gray=True):
        self.gray = gray
        self.real_crop_pad = real_crop_pad
        self.bg_subtract_real = bg_subtract_real
        self.render_crop_pad = render_crop_pad
        self.bg_subtract_renders = bg_subtract_renders
        self.img_dir = img_dir
        self.pics_per_iter = pics_per_iter
        self.transform_load = transform_load
        self.transform_get = transform_get
        self.use_cache = use_cache

        self.cache = {}
        self.files = []
        self.class_to_files = {}
        self.class_to_iter_amount = {}

        classes = list(dict.fromkeys(classes))
        files = os.listdir(img_dir)
        self.not_found_parts = set()

        for i, f in enumerate(files):
            fname, ext = os.path.splitext(f)
            real = fname[0:2] == 'r_'

            if len(fname.split('_')) == 4:
                if real:
                    part_name, iter_num, _ = fname.split('_')[1:]
                else:
                    Exception(f'Unexpected image name: {fname}')
            elif len(fname.split('_')) == 3:
                part_name, iter_num, _ = fname.split('_')
            else:
                raise Exception(f'Unexpected image name: {fname}')

            if part_name not in classes:
                self.not_found_parts.add(part_name)
                continue

            if part_name not in self.class_to_files:
                self.class_to_files[part_name] = set()
            self.class_to_files[part_name].add(i)

        for part_name in classes:
            pics = [(i, files[i]) for i in self.class_to_files[part_name]]
            iter_amount = max([self.parse_name(pic[1])[1] for pic in pics]) + 1

            for i in range(iter_amount):
                iter_pics = [p for p in pics if self.parse_name(p[1])[1] == i]
                if len(iter_pics) != self.pics_per_iter:
                    self.class_to_files[part_name] -= {p[0] for p in iter_pics}

            if len(self.class_to_files[part_name]) > 0:
                for pidx in self.class_to_files[part_name]:
                    self.files.append(files[pidx])
                self.class_to_files[part_name] = range(len(self.files) - len(self.class_to_files[part_name]), len(self.files))

        self.class_to_files = {key: self.class_to_files[key] for key in classes}

        files_in_dic = set().union(*self.class_to_files.values())
        assert min(files_in_dic) == 0
        assert max(files_in_dic) == len(self.files) - 1
        assert len(files_in_dic) == len(self.files)
        assert len(self.files) % self.pics_per_iter == 0

    def __len__(self):
        return int(len(self.files) / self.pics_per_iter)

    def __getitem__(self, idx: int):
        if self.use_cache and idx in self.cache.keys():
            if self.transform_get is not None:
                return self.transform_get(self.cache[idx][0]), self.cache[idx][1]
            return self.cache[idx]

        fname = self.files[idx * self.pics_per_iter]
        part_name, iter_num, iter_path, real, ext = self.parse_name(fname)

        def get_img(i):
            p = iter_path + str(i) + ext
            mode = ImageReadMode.GRAY if self.gray else ImageReadMode.RGB
            return read_image(p, mode=mode)

        unpr_images = [get_img(i) for i in range(self.pics_per_iter)]

        if self.bg_subtract_real and real:
            images_ls = get_square_cropped_img_iter(np.array(unpr_images), real=real, pad=self.real_crop_pad)
        elif self.bg_subtract_renders and not real:
            images_ls = get_square_cropped_img_iter(np.array(unpr_images), real=real, pad=self.render_crop_pad)
        else:
            images_ls = unpr_images

        images_ls = [self.transform_load(img) for img in images_ls]
        images = torch.stack(images_ls)

        if self.transform_get is not None:
            images = self.transform_get(images)

        label = torch.zeros(len(self.class_to_files.keys()))
        label[list(self.class_to_files.keys()).index(part_name)] = 1.0

        if self.use_cache:
            self.cache[idx] = images, label

        return images, label

    def get_files(self, part_name):
        return [self.files[i] for i in self.class_to_files[part_name]]

    def parse_name(self, fname):
        fname, ext = os.path.splitext(fname)
        real = fname[0:2] == 'r_'

        if real:
            part_name, iter_num, _ = fname.split('_')[1:]
        else:
            part_name, iter_num, _ = fname.split('_')

        rtxt = 'r_' if real else ''
        iter_path = self.img_dir + rtxt + part_name + '_' + iter_num + '_'

        return part_name, int(iter_num), iter_path, real, ext

    @staticmethod
    def from_cache(cache, **kwargs):
        dset = PartDataset(**kwargs)
        dset.cache = copy.deepcopy(cache)
        dset.use_cache = True

    @staticmethod
    def train_test_split(dset: 'PartDataset', train_split=0.8):
        if dset is None: return

        train = copy.deepcopy(dset)
        train.files.clear()

        test = copy.deepcopy(dset)
        test.files.clear()

        for key in dset.class_to_files.keys():
            train.class_to_files[key] = set()
            test.class_to_files[key] = set()

        for part_name, pic_idx in dset.class_to_files.items():
            iter_amount = max([dset.parse_name(dset.files[pic])[1] for pic in pic_idx]) + 1
            train_iter_amount = int(round(train_split * iter_amount, 0))

            for i, pic_i in enumerate(pic_idx):
                if i <= train_iter_amount * dset.pics_per_iter:
                    if part_name not in train.class_to_files:
                        train.class_to_files[part_name] = set()
                    train.class_to_files[part_name].add(pic_i)
                    train.files.append(dset.files[pic_i])
                else:
                    if part_name not in test.class_to_files:
                        test.class_to_files[part_name] = set()
                    test.class_to_files[part_name].add(pic_i)
                    test.files.append(dset.files[pic_i])

        assert len(test) + len(train) == len(dset)
        return train, test
