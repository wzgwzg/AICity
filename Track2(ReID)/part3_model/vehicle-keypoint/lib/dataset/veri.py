# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import pickle
from collections import defaultdict
from collections import OrderedDict

import json_tricks as json
import numpy as np

from dataset.JointsDataset import JointsDataset
from nms.nms import oks_nms


logger = logging.getLogger(__name__)


class VERIDataset(JointsDataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super(VERIDataset, self).__init__(cfg, root, image_set, is_train, transform)
        self.image_width = cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]
        self.pixel_std = 200.0

        self.num_joints = 20
        self.db = self._get_db()

        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)

        logger.info('=> load {} samples'.format(len(self.db)))

    def _get_db(self):
        db = []

        with open(os.path.join(self.root, self.image_set), 'r') as fp:
            for line in fp:
                tokens = line.strip().split()
                image_path = os.path.join(self.root, tokens[0])
                anno_joints = tokens[1:41]
                joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
                joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float)
                for ipt in range(self.num_joints):
                    x = anno_joints[ipt * 2 + 0]
                    y = anno_joints[ipt * 2 + 1]
                    joints_3d[ipt, 0] = x
                    joints_3d[ipt, 1] = y
                    joints_3d[ipt, 2] = 0
                    t_vis = 0
                    if x >= 0 and y >= 0:
                       t_vis = 1
                    joints_3d_vis[ipt, 0] = t_vis
                    joints_3d_vis[ipt, 1] = t_vis
                    joints_3d_vis[ipt, 2] = 0

                center = np.array( \
                        [self.image_width * 0.5, self.image_height * 0.5], \
                        dtype=np.float32)
                scale = np.array( \
                        [self.image_width / self.pixel_std, self.image_height / self.pixel_std], \
                        dtype=np.float32)
                
                db.append({
                    'image': image_path,
                    'center': center,
                    'scale': scale,
                    'joints_3d': joints_3d,
                    'joints_3d_vis': joints_3d_vis,
                    'filename': '',
                    'imgnum': 0,
                })

            return db
