import os
import pickle
import numpy as np

def load_features(pickle_file_path):
    pickle_file = open(pickle_file_path, 'rb')
    features = pickle.load(pickle_file)
    pickle_file.close()
    return features

if __name__ == '__main__':
    print('running...')
    root_path = os.getcwd()
    query_file_list = ['query_avg_res101_complete.pkl', 'query_kp_features_seresnext101_288by384_init1e1_lr3e2.pkl', 'query_se_res152_multi.pkl', 'query_avg_se_complete.pkl', 'query_mgn101_288_384_combine_fea_multi.pkl', 'query_hrnet_288_384_combine_fea_multi.pkl', 'query_sac101_288_384_combine_fea_multi.pkl']
    test_file_list = ['gallery_avg_res101_complete.pkl', 'gallery_kp_features_seresnext101_288by384_init1e1_lr3e2.pkl', 'gallery_se_res152.pkl', 'gallery_avg_se_complete.pkl', 'gallery_mgn101_288_384_combine_fea.pkl', 'gallery_hrnet_288_384_combine_fea.pkl', 'gallery_sac101_288_384_combine_fea.pkl']
    query_feature_list = []
    test_feature_list = []
    for i in range(len(query_file_list)):
        query_feature = load_features(root_path + '/pickle_file/pure_features/' + query_file_list[i])
        query_feature_list.append(query_feature)
        test_feature = load_features(root_path + '/pickle_file/pure_features/' + test_file_list[i])
        test_feature_list.append(test_feature)

    concat_query_features = {}
    for k in query_feature_list[0].keys():
        concat_query_features[k] = np.concatenate([query_feature_list[i][k] for i in range(len(query_feature_list))])
    
    concat_test_features = {}
    for k in test_feature_list[0].keys():
        concat_test_features[k] = np.concatenate([test_feature_list[i][k] for i in range(len(test_feature_list))])
 
    query_res_file = open(root_path + '/pickle_file/query_newbig7_cat_complete.pkl', 'wb')
    pickle.dump(concat_query_features, query_res_file)
    query_res_file.close()
    test_res_file = open(root_path + '/pickle_file/test_newbig7_cat_complete.pkl', 'wb')
    pickle.dump(concat_test_features, test_res_file)
    test_res_file.close()
    print('Done')
