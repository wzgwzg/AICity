import os 
import pdb
import shutil
import numpy as np
import pickle



with open('qg_dist.pkl','r') as fid:
    qg_dist = pickle.load(fid)



with open('query_track_info.pkl','r') as fid:
    query_track_info = pickle.load(fid)
with open('new_gallery_track_info.pkl','r') as fid:
    gallery_track_info = pickle.load(fid)
print('load query_gallery track info')


cam_group = {'c006':0,
             'c007':0,
             'c008':0,
             'c009':0,

             'c029':1,

             'c033':2,
             'c034':2,
             'c035':2,
             'c036':2,

                }

def not_time_overlap(query_stime, query_etime, query_cam_groupid,
                gallery_stime, gallery_etime, gallery_cam_groupid,
                cam_move_frame, time_diff):
    if gallery_stime > query_stime:
        query_stime_cp = query_stime + cam_move_frame[query_cam_groupid, gallery_cam_groupid]
        query_etime_cp = query_etime + cam_move_frame[query_cam_groupid, gallery_cam_groupid] + time_diff[query_cam_groupid, gallery_cam_groupid]

        gallery_stime_cp = gallery_stime - time_diff[query_cam_groupid, gallery_cam_groupid]
        gallery_etime_cp = gallery_etime
    else:
        query_stime_cp = query_stime - time_diff[query_cam_groupid, gallery_cam_groupid]
        query_etime_cp = query_etime

        gallery_stime_cp = gallery_stime + cam_move_frame[query_cam_groupid, gallery_cam_groupid]
        gallery_etime_cp = gallery_etime + cam_move_frame[query_cam_groupid, gallery_cam_groupid] + time_diff[query_cam_groupid, gallery_cam_groupid]

    max_stime = max(query_stime_cp, gallery_stime_cp)
    min_etime = min(query_etime_cp, gallery_etime_cp)
    return max_stime>min_etime 


def not_time_overlap_fix_move_time(query_stime, query_etime,
                gallery_stime, gallery_etime, 
                move_frame, time_diff):
    if gallery_stime > query_stime:
        query_stime_cp = query_stime + move_frame
        query_etime_cp = query_etime + move_frame + time_diff

        gallery_stime_cp = gallery_stime - time_diff
        gallery_etime_cp = gallery_etime
    else:
        query_stime_cp = query_stime - time_diff
        query_etime_cp = query_etime

        gallery_stime_cp = gallery_stime + move_frame
        gallery_etime_cp = gallery_etime + move_frame + time_diff

    max_stime = max(query_stime_cp, gallery_stime_cp)
    min_etime = min(query_etime_cp, gallery_etime_cp)
    return max_stime>min_etime

cam_move_frame = np.array([
                [0,    0,   0],
                [0,    0, 150],
                [0,  150,   0],
               
    ])

group_dis = np.zeros((1052,18290))

time_diff = np.array([
                [300, 10000, 10000],
                [10000,   50,  300],
                [10000,  300,  300],
               
    ])
# time_diff = 300
c028_threshold = 500
c027_threshold = 800
same_cam_time_frame_thres = 35
for i in range(1052):
    query_name = str(i+1).zfill(6)+'.jpg'
    print(query_name)
    query_cam = query_track_info[query_name]['cam']

    query_frame = query_track_info[query_name]['frame']
    query_stime = query_track_info[query_name]['start_time']
    query_etime = query_track_info[query_name]['end_time']
    if query_cam not in cam_group.keys():
        continue
    query_cam_groupid = cam_group[query_cam]

    for j in range(18290):
        gallery_name = str(j+1).zfill(6)+'.jpg'
        gallery_cam = gallery_track_info[gallery_name]['cam']
        gallery_stime = gallery_track_info[gallery_name]['start_time']
        gallery_etime = gallery_track_info[gallery_name]['end_time']
        if gallery_cam == query_cam:
            max_stime = max(query_stime, gallery_stime)
            min_etime = min(query_etime, gallery_etime)
            if max_stime - min_etime > same_cam_time_frame_thres or max_stime - min_etime <0:
                group_dis[i,j]=50
        elif query_cam_groupid==2 and gallery_cam =='c028':
            to_val = not_time_overlap_fix_move_time(query_stime, query_etime, 
                                              gallery_stime, gallery_etime,
                                               c028_threshold, 400)
            if to_val:
                group_dis[i,j]=50
        elif query_cam_groupid==2 and gallery_cam =='c027':
            to_val = not_time_overlap_fix_move_time(query_stime, query_etime, 
                                              gallery_stime, gallery_etime,
                                               c027_threshold, 400)
            if to_val:
                group_dis[i,j]=50

        else: 
            if gallery_cam not in cam_group.keys():
                continue
            gallery_cam_groupid = cam_group[gallery_cam]
            time_overlap_val = not_time_overlap(query_stime, query_etime, query_cam_groupid,
                                              gallery_stime, gallery_etime, gallery_cam_groupid,
                                               cam_move_frame, time_diff)
            if time_overlap_val:
                group_dis[i,j]=50

cur_dist = qg_dist + group_dis
with open('qg_dist_mat_twf2.pkl','w') as fid:
    pickle.dump(cur_dist,fid)





   
