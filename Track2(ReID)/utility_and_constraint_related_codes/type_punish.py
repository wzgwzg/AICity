import pickle
import os
import numpy as np
import pdb


if __name__ == '__main__':
    print('running...')
    root_path = os.getcwd()
    
    q_track_file = open(root_path + '/reid/pickle_file/track_info/query_track_info.pkl', 'rb')
    g_track_file = open(root_path + '/reid/pickle_file/track_info/gallery_track_info.pkl', 'rb')
    q_track_info = pickle.load(q_track_file)
    g_track_info = pickle.load(g_track_file)
    q_track_file.close()
    g_track_file.close()
    

    type_sim_file = open(root_path + '/reid/pickle_file/track_info/type_sim.pkl', 'rb')
    type_sim = pickle.load(type_sim_file)
    type_sim_file.close()

    dist_name = 'distmat_strategy'
    distmat_pfile = open(root_path + '/reid/pickle_file/dist_file/' + dist_name + '.pkl', 'rb')
    distmat = pickle.load(distmat_pfile)
    distmat_pfile.close()
    for i in range(1, 1053):
        q_name = '%06d' %i
        q_name = q_name + '.jpg'
        q_cam = q_track_info[q_name]['cam']
        for j in range(1, 18291):
            g_name = '%06d' %(j)
            g_name = g_name + '.jpg'
            g_cam = g_track_info[g_name]['cam']
	    if q_cam == 'c035' or g_cam == 'c035':
	    	if distmat[i-1][j-1] > 3.0 and type_sim[i-1][j-1] < 0.1:
		    distmat[i-1][j-1] += 5.0
    
    dist_pf =  open(root_path + '/reid/pickle_file/distmat_with_type_punish.pkl', 'wb')
    pickle.dump(distmat, dist_pf)
    dist_pf.close()
    
    print('Done')
	
