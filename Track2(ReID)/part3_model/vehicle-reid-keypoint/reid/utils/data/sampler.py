from __future__ import absolute_import
from collections import defaultdict

import os
import random
import numpy as np
import torch
from torch.utils.data.sampler import (
    Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler,
    WeightedRandomSampler)


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances=1):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(self.num_samples)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            if len(t) >= self.num_instances:
                t = np.random.choice(t, size=self.num_instances, replace=False)
            else:
                t = np.random.choice(t, size=self.num_instances, replace=True)
            ret.extend(t)
        return iter(ret)


class RandomIdentityAndCameraSampler(Sampler):
    def __init__(self, data_source, num_instances=1):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = {}
        for index, (_, pid, camid) in enumerate(data_source):
            if pid not in self.index_dic:
                self.index_dic[pid] = {}
            if camid not in self.index_dic[pid]:
                self.index_dic[pid][camid] = []
            self.index_dic[pid][camid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(self.num_samples)
        ret = []
        for i in indices:
            pid = self.pids[i]
            camids = self.index_dic[pid].keys()
            if len(camids) >= self.num_instances:
                camids = np.random.choice(camids, size=self.num_instances, replace=False)
            else:
                new_camids = []
                while (self.num_instances-len(new_camids) >= len(camids)):
                    new_camids.extend(camids)
                if self.num_instances-len(new_camids) != 0:
                    supl_camids = np.random.choice(camids, size=(self.num_instances-len(new_camids)), replace=False)
                    new_camids.extend(supl_camids)
                camids = new_camids
            t = []
            for camid in camids:
                index = np.random.choice(self.index_dic[pid][camid], size=1, replace=False)
                t.append(index[0])
            ret.extend(t)
        return iter(ret)
