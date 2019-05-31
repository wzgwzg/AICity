import os
import pickle
import shutil
import pdb
import numpy as np
import copy

with open('test_track.txt','r') as fid:
    test_track_ori = fid.readlines()
print('load given gallery track info')


with open('gallery_track_info.pkl','r') as fid:
    gallery_track_info = pickle.load(fid)
print('load my gallery group res')



ori_gallery_group = {}
count = 0
groups_info_each_gallery_ori = {}

for line in test_track_ori:
    track_contain_id = line.strip()
    track_ids = track_contain_id.split(' ')
    ori_gallery_group[count] = track_ids
    
    for each_gallery in track_ids:
        groups_info_each_gallery_ori[each_gallery]=count
    count +=1
# pdb.set_trace()
gallery_combine_ori_track_info = {}
for i in range(18290):
    gallery_name = str(i+1).zfill(6)+'.jpg'
    print(gallery_name)

    gallery_cam_extend = set()
    gallery_cam  = gallery_track_info[gallery_name]['cam']
    gallery_id = gallery_track_info[gallery_name]['id']
    gallery_cid = gallery_cam +'_'+ str(gallery_id)
    gallery_cam_extend.add(gallery_cid)

    group_id = groups_info_each_gallery_ori[gallery_name]
    group_imgs = ori_gallery_group[group_id]
    for each_img in group_imgs:
        cur_img_cam = gallery_track_info[each_img]['cam']
        cur_img_id = gallery_track_info[each_img]['id']
        cur_cid = cur_img_cam +'_'+str(cur_img_id)
        gallery_cam_extend.add(cur_cid)
    gallery_combine_ori_track_info[gallery_name] = {}
    gallery_combine_ori_track_info[gallery_name]['cam'] = gallery_cam
    gallery_combine_ori_track_info[gallery_name]['id'] = gallery_id
    gallery_combine_ori_track_info[gallery_name]['cams'] = gallery_cam_extend
    gallery_combine_ori_track_info[gallery_name]['groupid'] = group_id
pdb.set_trace()
with open('gallery_cid_info.pkl','w') as fid:
    pickle.dump(gallery_combine_ori_track_info,fid)

group_gallery_info_ori_track_info = {}
for i in range(count):
    print(i)
    group_gallery_info_ori_track_info[i] = {}
    group_gallery_info_ori_track_info[i]['pic'] = ori_gallery_group[i]
    group_gallery_info_ori_track_info[i]['cam'] = gallery_combine_ori_track_info[ ori_gallery_group[i][0] ]['cams']
with open('group_gallery_info_by_ori_track_info.pkl','w') as fid:
    pickle.dump(group_gallery_info_ori_track_info,fid) 



    