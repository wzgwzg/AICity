import pickle
import os
import numpy as np


if __name__ == '__main__':
    print('running...')
    root_path = os.getcwd()
    
    query_direct_pfile = open(root_path + 'pickle_file/direction_file/query_direction.pkl', 'rb')
    query_direct = pickle.load(query_direct_pfile)
    query_direct_pfile.close()
    gallery_direct_pfile = open(root_path + 'pickle_file/direction_file/gallery_direction.pkl', 'rb')
    gallery_direct = pickle.load(gallery_direct_pfile)
    gallery_direct_pfile.close()
    
    query_keys = list(query_direct.keys())
    query_keys.sort()
    query_direct = np.concatenate([np.expand_dims(query_direct[f], 0) for f in query_keys])
    gallery_keys = list(gallery_direct.keys())
    gallery_keys.sort()
    gallery_direct = np.concatenate([np.expand_dims(gallery_direct[f], 0) for f in gallery_keys])
    gallery_direct = np.transpose(gallery_direct)
    
    q_g_direct_sim = np.dot(query_direct, gallery_direct)
    with open(root_path + 'pickle_file/q_g_direct_sim.pkl', 'wb') as p_f:
        pickle.dump(q_g_direct_sim, p_f, protocol=2)
        
    q_q_direct_sim = np.dot(query_direct, np.transpose(query_direct))
    with open(root_path + 'pickle_file/q_q_direct_sim.pkl', 'wb') as p_f:
        pickle.dump(q_q_direct_sim, p_f, protocol=2)
        
    g_g_direct_sim = np.dot(gallery_direct, np.transpose(gallery_direct))
    with open(root_path + 'pickle_file/g_g_direct_sim.pkl', 'wb') as p_f:
        pickle.dump(g_g_direct_sim, p_f, protocol=2)
 
    print('Done')
	
