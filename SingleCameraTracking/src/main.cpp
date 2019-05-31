#include <iostream>
#include <sstream>
#include <glog/logging.h>
#include "VideoTracker.h"
using namespace std;

#define MOTDIR "../aic19/aic19-track1-mtmc/"
#define RUNMOTCAFFE

int main(int argc, char** argv)
{
	if (argc != 5) {
		printf("usage: %s dataset sequances_idx is_visualized is_imagewrite\n", argv[0]);
		return -1;
	}
	::google::InitGoogleLogging(argv[0]);

	string dataset = argv[1];
	string seq_idx = argv[2];
	string is_feature_write = argv[3];
	string is_imagewrite = argv[4];
	bool write_feature;
	bool write_to_image;

	if(is_feature_write == "1") {
		write_feature = true;
	}
	else {
		write_feature = false;
	}
	if(is_imagewrite == "1") {
		write_to_image = true;
	}
	else {
		write_to_image = false;
	}
 
    stringstream ss_tmp;
    ss_tmp << dataset;
    vector<string> str_vec;
    string str_tmp;
    while (getline(ss_tmp, str_tmp, '/')) {
        str_vec.push_back(str_tmp);
    }

    VideoTracker* t = new VideoTracker;
    string seq_path = MOTDIR + dataset + "/" + seq_idx + "/";
    std::cout << seq_path << std::endl;
    string output_path = "../result/" + seq_idx + "_" + str_vec[0] + ".txt";

#ifdef RUNMOTFEAT
	if(t->RunTrackerWithDat(seq_path.data(), output_path.data(), write_feature, write_to_image) == false) {
		cout << t->showErrMsg() << endl;
	}
#endif

#ifdef RUNMOTCAFFE
	if(t->RunTrackerWithDets(seq_path.data(), output_path.data(), write_feature, write_to_image) == false) {
		cout << t->showErrMsg() << endl;
	}
#endif

	cout << "Finished." << endl;
	return 0;
}
