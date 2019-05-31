import os 
import pdb
import shutil
import numpy as np
import pickle


with open('qg_dist.pkl','r') as fid:
    qg_dist = pickle.load(fid)

with open('group_gallery_info_by_ori_track_info.pkl','r') as fid:
    group_gallery_info_ori_track_info = pickle.load(fid)


group_dis = np.zeros((1052,18290))
for each_group in group_gallery_info_ori_track_info.keys():
    pics = group_gallery_info_ori_track_info[each_group]['pic']
    pic_index = [int(ii.split('.')[0])-1 for ii in pics]
    group_all_dis = qg_dist[:,pic_index]
    group_min_dis = np.min(group_all_dis,axis=1)
    for ii in pic_index:
        group_dis[:,ii] = group_min_dis
with open('group_dis.pkl','w') as fid:
    pickle.dump(group_dis,fid)

cur_dist = qg_dist + group_dis
with open('new_q_g_dist.pkl','w') as fid:
    pickle.dump(cur_dist,fid)
print("Done!")



 
