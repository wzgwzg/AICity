from __future__ import print_function, absolute_import
import os.path as osp

from ..utils.data import Dataset


class Small_Vehicle(Dataset):
    def __init__(self, root, split_id=0, num_val=100, download=True):
        super(Small_Vehicle, self).__init__(root, split_id=split_id)

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. " +
                               "You can use download=True to download it.")

        self.load(num_val)
