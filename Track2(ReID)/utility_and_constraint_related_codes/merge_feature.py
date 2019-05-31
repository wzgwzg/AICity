import os
import pickle


def load_features(pickle_file_path):
    pickle_file = open(pickle_file_path, 'rb')
    features = pickle.load(pickle_file)
    pickle_file.close()
    return features

if __name__ == '__main__':
    print('running...')
    root_path = os.getcwd()
    res_file = open(root_path + '/pickle_file/save_file_name.pickle', 'wb')
    features1 = load_features(root_path + '/pickle_file/your_feature_pickle_file1.pickle')
    features2 = load_features(root_path + '/pickle_file/your_feature_pickle_file2.pickle')
    final_features = {}
    for k in features.keys():
        final_features[k] = (features1[k] + features2[k]) / 2
    pickle.dump(final_features, res_file)
    res_file.close()
    print('Done')
