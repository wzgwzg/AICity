import pickle
import os
import numpy as np


if __name__ == '__main__':
    print('running...')
    root_path = os.getcwd()

    query_info_pfile = open(root_path + 'pickle_file/track_info/query_track_info.pkl', 'rb')
    query_info = pickle.load(query_info_pfile)
    query_info_pfile.close()
    gallery_info_pfile = open(root_path + 'pickle_file/track_info/gallery_track_info.pkl', 'rb')
    gallery_info = pickle.load(gallery_info_pfile)
    gallery_info_pfile.close()
    
    query_attrib_pfile = open(root_path + 'pickle_file/attrib_file/query_new_attribute.pkl', 'rb')
    query_attrib = pickle.load(query_attrib_pfile)
    query_attrib_pfile.close()
    
    gallery_attrib_pfile = open(root_path + 'pickle_file/attrib_file/gallery_new_attribute.pkl', 'rb')
    gallery_attrib = pickle.load(gallery_attrib_pfile)
    gallery_attrib_pfile.close()
    
    gallery_track_pf = open(root_path + '/pickle_file/track_info/gallery_track_map.pkl', 'rb')
    gallery_track_map = pickle.load(gallery_track_pf)
    gallery_track_pf.close()
    gallery_trackid_dict = gallery_track_map['gallery_trackid_dict']
    trackid_gallery_dict = gallery_track_map['trackid_gallery_dict']
    
    query_keys = list(query_attrib.keys())
    query_keys.sort()
    query_type_dict = {}
    for k in range(1,1053):
        tmp_list = []
        k = str(k).zfill(6)
        k_img = k + '.jpg'
        tmp_list.append(k_img)
        idx = query_keys.index(k_img)
        for j in range(idx + 1, min(len(query_keys), idx + 100)):
            if query_keys[j].split('_')[0] == k:
                tmp_list.append(query_keys[j])
            else:
                break
        x = np.concatenate([np.expand_dims(query_attrib[f], 0) for f in tmp_list], 0)
        x = np.mean(x, 0)
        query_type_dict[k_img] = x
        
    query_keys = list(query_type_dict.keys())
    query_keys.sort()
    for k in query_keys:
        if query_info[k]['cam'] != 'c035':
            continue
        item = query_type_dict[k]
        if  item[6] > 0.2 and item[6] < 0.55:
            q_rank = np.argsort(-item)
            delta = float(int(item[6]*10)) / 10
            item[6] = 0.0001
            if q_rank[0] == 6:
                item[q_rank[1]] += delta
            else:
                item[q_rank[0]] += delta
            query_type_dict[k] = item
     
    query_keys = list(query_type_dict.keys())
    query_keys.sort()
    for k in query_keys:
        query_type_dict[k][3] = query_type_dict[k][3] + query_type_dict[k][6]
        query_type_dict[k][6] = 0
    
    
    gallery_type_dict = {}
    for i in range(len(trackid_gallery_dict.keys())):
        img_list = trackid_gallery_dict[i]
        tmp_y = np.concatenate([np.expand_dims(gallery_attrib[f], 0) for f in img_list], 0)
        tmp_y = np.mean(tmp_y, 0)
        for f in img_list:
            gallery_type_dict[f] = tmp_y
        
    gallery_keys = list(gallery_type_dict.keys())
    gallery_keys.sort()
    for k in gallery_keys:
        if gallery_info[k]['cam'] != 'c035':
            continue
        item = gallery_type_dict[k]
        if  item[6] > 0.2 and item[6] < 0.55:
            q_rank = np.argsort(-item)
            delta = float(int(item[6]*10)) / 10
            item[6] = 0.0001
            if q_rank[0] == 6:
                item[q_rank[1]] += delta
            else:
                item[q_rank[0]] += delta
            gallery_type_dict[k] = item
            
    
    gallery_keys = list(gallery_type_dict.keys())
    gallery_keys.sort()
    for k in gallery_keys:
        gallery_type_dict[k][3] = gallery_type_dict[k][3] + gallery_type_dict[k][6]
        gallery_type_dict[k][6] = 0
           
    query_type_mat = np.concatenate([np.expand_dims(query_type_dict[str(f).zfill(6)+'.jpg'], 0) for f in range(1,1053)], 0)
    gallery_type_mat = np.concatenate([np.expand_dims(gallery_type_dict[str(f).zfill(6)+'.jpg'], 0) for f in range(1,18291)], 0)
    query_gallery_type_sim = np.dot(query_type_mat, np.transpose(gallery_type_mat))
    with open(root_path + 'pickle_file/type_sim.pkl', 'wb') as pf:
        pickle.dump(query_gallery_type_sim, pf, protocol=2) 
    print('Done')
	
