import shutil
import pickle
import pdb
import numpy as np
import os


with open('./gallery_track_info.pkl','r') as fid:
    gallery_track_info = pickle.load(fid)
print('load gallery_track_info')


with open('gallery_cid_info.pkl','r') as fid:
    gallery_cid_track_info = pickle.load(fid)

with open('group_gallery_info_by_ori_track_info.pkl','r') as fid:
    group_gallery_info_ori_track_info = pickle.load(fid) 

new_gallery_track_info = {}
for i in range(18290):
    gallery_name = str(i+1).zfill(6)+'.jpg'
    new_gallery_track_info[gallery_name]={}
    new_gallery_track_info[gallery_name]['cam'] = ''
    new_gallery_track_info[gallery_name]['id'] =''
    new_gallery_track_info[gallery_name]['start_time'] =''
    new_gallery_track_info[gallery_name]['end_time'] =''
    new_gallery_track_info[gallery_name]['group_id'] =''
    new_gallery_track_info[gallery_name]['cam_set'] = set()

count = 0
for i in range(18290):
    gallery_name = str(i+1).zfill(6)+'.jpg'
    gallery_cam  = gallery_track_info[gallery_name]['cam']
    gallery_id = gallery_track_info[gallery_name]['id']
    gallery_stime = gallery_track_info[gallery_name]['start_time']
    gallery_etime = gallery_track_info[gallery_name]['end_time']
    gallery_frame = gallery_track_info[gallery_name]['frame']

    gallery_group_id = gallery_cid_track_info[gallery_name]['groupid']
    gallery_cam_set = gallery_cid_track_info[gallery_name]['cams']


    gallery_group_imgs = group_gallery_info_ori_track_info[gallery_group_id]['pic']

    new_stime = gallery_stime
    new_etime = gallery_etime
    for each_img in gallery_group_imgs:
        img_cam = gallery_track_info[each_img]['cam']
        if img_cam != gallery_cam:
            continue
        img_stime = gallery_track_info[each_img]['start_time']
        img_etime = gallery_track_info[each_img]['end_time']
        max_stime = max(gallery_stime, img_stime)
        min_etime = min(gallery_etime, img_etime)
        if max_stime-min_etime>1000:
            continue
        new_stime = min(new_stime, img_stime)
        new_etime = max(new_etime, img_etime)

    new_gallery_track_info[gallery_name]['cam'] = gallery_cam
    new_gallery_track_info[gallery_name]['id'] = gallery_id
    new_gallery_track_info[gallery_name]['frame'] = gallery_frame
    new_gallery_track_info[gallery_name]['start_time'] = new_stime
    new_gallery_track_info[gallery_name]['end_time'] = new_etime
    new_gallery_track_info[gallery_name]['group_id'] = gallery_group_id
    new_gallery_track_info[gallery_name]['cam_set'] = gallery_cam_set
    if (new_etime-new_stime)-(gallery_etime-gallery_stime)>100:
        count+=1
        print(gallery_name,gallery_cam,gallery_id,gallery_cam_set,new_stime,new_etime,gallery_stime,gallery_etime)
print(count)


with open('new_gallery_track_info.pkl','w') as fid:
    pickle.dump(new_gallery_track_info,fid)
pdb.set_trace()
