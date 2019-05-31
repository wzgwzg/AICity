from __future__ import absolute_import

import os.path as osp
import math
import random
import numpy as np

from PIL import Image
import cPickle

import torch


class Preprocessor(object):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        img = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, fname, pid, camid


def do_resize(img, height, width):
    w, h = img.size
    if h == height and w == width:
        return img
    return img.resize((width, height), Image.BILINEAR)


class PreprocessorWithMasks(object):
    def __init__(self, dataset, root=None, masks_root=None, 
            height=None, width=None, num_masks=None, 
            transform=None, is_training=True):
        super(PreprocessorWithMasks, self).__init__()
        self.dataset = dataset
        self.root = root
        self.masks_root = masks_root
        self.height = height
        self.width = width
        self.num_masks = num_masks
        self.transform = transform
        self.is_training = is_training

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname; masks_fpath = None
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        if self.masks_root is not None:
            masks_fpath = osp.join(self.masks_root, fname+'.pkl')
        is_hf = False
        if self.is_training:
            is_hf = random.random() < 0.5
        img = Image.open(fpath).convert('RGB')
        img = do_resize(img, self.height, self.width)
        if is_hf:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if masks_fpath is not None:
            with open(masks_fpath, 'r') as fp:
                masks = cPickle.load(fp)
                c, h, w = masks.shape
                if is_hf:
                    masks = masks.reshape((-1, w))
                    for i in range(c*h):
                        masks[i] = masks[i][::-1]
                    masks = masks.reshape(c, h, w)
        if self.transform is not None:
            img = self.transform(img)
            masks = torch.from_numpy(masks).type_as(img)
        return img, masks, fname, pid, camid
