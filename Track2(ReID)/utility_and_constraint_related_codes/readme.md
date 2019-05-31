## Utility and Constraint Related Codes ##

**Please modify related paths before running the codes.**

* merge_feature.py
Compute the average feature of two feature files.

* concat_feature.py
Concatenate features.

* gallery_track_info_combine_official_track.py
Combine original track info with tracking algorithm results.

* get_gallery_cam_trackid_from_ori_track_info.py
Get info from test_track.txt, 

* direction_similarity_generation.py
Compute vehicle orientation similarity.

* type_similarity_generation.py
Compute vehicle type similarity.

* run_add_dist.py
Add cam_id and direction constraint punishment. Refer to our paper for more details. It contains the function for calculating euclidean distance matrix.

* type_punish.py
Add vehicle type constraint punishment.

* twf_v2_before_rerank.py
Add time windows filter (temporal constraint) before rerank.

* run_rerank.py
Run reranking algorithm.

* twf_v2_after_rerank.py
Add time windows filter(temporal constraint) after rerank. 

* group_dis_after_rerank.py
Add group distance on the original q_g_dist.

* generate_result.py
Generate the final ranking results.


In order to get the final ranking results, the user should:
1. Train all reid models and get the merged and concatenated features.
2. Calculate the original euclidean distance matrix.
3. Use run_add_dist.py, type_punish.py and twf_v2_before_rerank.py to add constraint punishment.
4. Run run_rerank.py to conduct reranking.
5. Use run_add_dist.py, type_punish.py and twf_v2_after_rerank.py to add constraint punishment again.
6. Run group_dis_after_rerank.py to add group distance.
7. Run generate_result.py to get the final ranking results.
Note: tracking and other auxiliary information will be used in this procedure.

## Some Important Pickle Files Related to This Part ##

Several important pickle files are available at [pickle_file](https://pan.baidu.com/s/1u6d6dX0uPvyrqgOB0O4Qyg)(extract code: p3fg). Some files related to this part are listed below. The remainings will be introduced by other parts.

* gallery_cid_info.pkl 
This file contains each gallery image’s cam, trackid, groupid and all (cams_trackids) in this groups.

* group_gallery_info_by_ori_track_info.pkl
This file contains each group’s gallery images and all (cam_trackids) in this group.

* query_track_info.pkl and gallery_track_info.pkl
These two files contain cam, trackid, track_start_time and track_end_time information of each query and gallery image. all these info is obtained from our tracking algorithm.  

* new_gallery_track_info.pkl
This file contains each gallery image’s cam, trackid, track_start_time, track_end_time, etc. This info is combined with the tracking algorithm results(gallery_track_info).

* gallery_track_map.pkl
This file contains the map between gallery image name and the corresponding trackid.

* may_9_gd1_twf2_rerank_type_direct.pkl
The final distance matrix used to generate ranking results.
