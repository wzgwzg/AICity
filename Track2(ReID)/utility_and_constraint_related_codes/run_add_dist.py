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
    x = x.cuda()
    y = y.cuda()
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist



if __name__ == '__main__':
    print('running...')
    root_path = os.getcwd()
    print(root_path)
    
    #query_features = extract_features(root_path + '/reid/pickle_file/pure_features/query_newbig7_cat_complete.pkl')
    #test_features = extract_features(root_path + '/reid/pickle_file/pure_features/test_newbig7_cat_complete.pkl')
    
    #q_f = open(root_path + '/reid/txt_file/query.txt', 'r')
    #t_f = open(root_path + '/reid/txt_file/test.txt', 'r')
    #query = []
    #gallery = []

    #for line in q_f.readlines():
    #    query.append(line.strip())

    #for line in t_f.readlines():
    #    gallery.append(line.strip())

    #q_f.close()
    #t_f.close()
    
    q_track_file = open(root_path + '/reid/pickle_file/track_info/query_track_info.pkl', 'rb')
    g_track_file = open(root_path + '/reid/pickle_file/track_info/gallery_track_info.pkl', 'rb')
    q_track_info = pickle.load(q_track_file)
    g_track_info = pickle.load(g_track_file)
    q_track_file.close()
    g_track_file.close()
    
    q_g_direct_sim_file = open(root_path + '/reid/pickle_file/direction_file/q_g_direct_sim.pkl', 'rb')
    q_g_direct_sim = pickle.load(q_g_direct_sim_file)
    q_g_direct_sim_file.close()
    
    q_q_direct_sim_file = open(root_path + '/reid/pickle_file/direction_file/q_q_direct_sim.pkl', 'rb')
    q_q_direct_sim = pickle.load(q_q_direct_sim_file)
    q_q_direct_sim_file.close()

    g_g_direct_sim_file = open(root_path + '/reid/pickle_file/direction_file/g_g_direct_sim.pkl', 'rb')
    g_g_direct_sim = pickle.load(g_g_direct_sim_file)
    g_g_direct_sim_file.close()

    #q_g_dist = vehicle_pairwise_distance(query_features, test_features, query, gallery)
    #q_g_dist = q_g_dist.cpu().numpy()
    
    distmat_pfile = open(root_path + '/reid/pickle_file/dist_file/distmat_newbig7_no_strategy_no_multi_no_rerank.pkl', 'rb')
    distmat_dict = pickle.load(distmat_pfile)
    distmat_pfile.close()
    q_g_dist = distmat_dict['q_g_dist']
    q_q_dist = distmat_dict['q_q_dist']
    g_g_dist = distmat_dict['g_g_dist']

    for i in range(1, 1053):
        q_name = '%06d' %i
        q_name = q_name + '.jpg'
        q_cam = q_track_info[q_name]['cam']
        q_tid = q_track_info[q_name]['id']
        for j in range(1, 18291):
            g_name = '%06d' %(j)
            g_name = g_name + '.jpg'
            g_cam = g_track_info[g_name]['cam']
            g_tid = g_track_info[g_name]['id']
            if g_cam == q_cam and g_tid != q_tid:
                if q_g_dist[i-1][j-1] <=2.5:
                    q_g_dist[i-1][j-1] = q_g_dist[i-1][j-1] + 0.5 
                else:
                    q_g_dist[i-1][j-1] = q_g_dist[i-1][j-1] + 1.5 
                    if q_cam == 'c035':
                        q_g_dist[i-1][j-1] = q_g_dist[i-1][j-1] + 4
            if (g_cam == 'c006' and q_cam == 'c009') or (g_cam == 'c009' and q_cam == 'c006') or (g_cam == 'c007' and q_cam == 'c008') or (g_cam == 'c008' and q_cam == 'c007'):
                if q_g_direct_sim[i-1][j-1] >= 0.03:
                    q_g_dist[i-1][j-1] = q_g_dist[i-1][j-1] + 2.5  
 
    	print('q_g_dist completed')
 
    #q_q_dist = vehicle_pairwise_distance(query_features, query_features, query, query)
    #q_q_dist = q_q_dist.cpu().numpy()
    
    for i in range(1, 1053):
        q_name = '%06d' %i
        q_name = q_name + '.jpg'
        q_cam = q_track_info[q_name]['cam']
        q_tid = q_track_info[q_name]['id']
        for j in range(1, 1053):
            q2_name = '%06d' %(j)
            q2_name = q2_name + '.jpg'
            q2_cam = q_track_info[q2_name]['cam']
            q2_tid = q_track_info[q2_name]['id']
            if q2_cam == q_cam and q2_tid != q_tid:
        	if q_q_dist[i-1][j-1] <=2.5:
                    q_q_dist[i-1][j-1] = q_q_dist[i-1][j-1] + 0.5 
                else:
                    q_q_dist[i-1][j-1] = q_q_dist[i-1][j-1] + 1.5 
                    if q_cam == 'c035':
                        q_q_dist[i-1][j-1] = q_q_dist[i-1][j-1] + 4 
            if (q2_cam == 'c006' and q_cam == 'c009') or (q2_cam == 'c009' and q_cam == 'c006') or (q2_cam == 'c007' and q_cam == 'c008') or (q2_cam == 'c008' and q_cam == 'c007'):
            	if q_q_direct_sim[i-1][j-1] >= 0.03:
                    q_q_dist[i-1][j-1] = q_q_dist[i-1][j-1] + 2.5 
    print('q_q_dist completed')

    #g_g_dist = vehicle_pairwise_distance(test_features, test_features, gallery, gallery)
    #g_g_dist = g_g_dist.cpu().numpy()
    
    
    for i in range(1, 18291):
        g_name = '%06d' %i
        g_name = g_name + '.jpg'
        g_cam = g_track_info[g_name]['cam']
        g_tid = g_track_info[g_name]['id']
        for j in range(1, 18291):
            g2_name = '%06d' %(j)
            g2_name = g2_name + '.jpg'
            g2_cam = g_track_info[g2_name]['cam']
            g2_tid = g_track_info[g2_name]['id']
            if g2_cam == g_cam and g2_tid != g_tid:
                if g_g_dist[i-1][j-1] <=2.5:
                    g_g_dist[i-1][j-1] = g_g_dist[i-1][j-1] + 0.5
                else:
                    g_g_dist[i-1][j-1] = g_g_dist[i-1][j-1] + 1.5
                    if g_cam == 'c035':
                        g_g_dist[i-1][j-1] = g_g_dist[i-1][j-1] + 4.0
            if (g2_cam == 'c006' and g_cam == 'c009') or (g2_cam == 'c009' and g_cam == 'c006') or (g2_cam == 'c007' and g_cam == 'c008') or (g2_cam == 'c008' and g_cam == 'c007'):
            	if g_g_direct_sim[i-1][j-1] >= 0.03:
                    g_g_dist[i-1][j-1] = g_g_dist[i-1][j-1] + 2.5 
    print('g_g_dist completed')
    
    distmat_dict = {'q_g_dist':q_g_dist, 'q_q_dist':q_q_dist, 'g_g_dist':g_g_dist}
    distmat_pfile = open(root_path + '/reid/pickle_file/distmat_strategy.pkl', 'wb')
    pickle.dump(distmat_dict, distmat_pfile)
    distmat_pfile.close()

    
    print('Done')
