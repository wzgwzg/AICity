from __future__ import print_function, absolute_import
import time
from collections import OrderedDict
from collections import defaultdict
from sklearn.metrics import average_precision_score
import numpy as np
import torch
import pickle
import os


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor

def extract_features(pickle_file):
    f = open(pickle_file, 'rb')
    features = pickle.load(f)
    f.close()
    return features



def vehicle_pairwise_distance(query_features, test_features,  query, gallery):
    x = torch.cat([torch.from_numpy(query_features[f]).unsqueeze(0) for f in query], 0)
    y = torch.cat([torch.from_numpy(test_features[f]).unsqueeze(0) for f in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist


if __name__ == '__main__':
    print('running...')
    root_path = os.getcwd() 
   
    
    distmat_pfile = open(root_path + '/reid/pickle_file/dist_file/may_9_gd1_twf2_rerank_type_direct.pkl', 'r')
    distmat = pickle.load(distmat_pfile)
    distmat_pfile.close()

    sort_distmat_index = np.argsort(distmat, axis=1)
    with open('track2.txt', 'w') as f:
        for item in sort_distmat_index:
            for i in range(99):
                f.write(str(item[i] + 1) + ' ')
            f.write(str(item[99] + 1) + '\n')

    print('Done')
