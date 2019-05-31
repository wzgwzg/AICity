#include <fstream>
#include <unordered_map>
#include "VideoTracker.h"
#include "opencv2/opencv.hpp"
#include "matching/tracker.h"
#include "feature/FeatureExtractor.h"
using namespace cv;
using namespace std;

#define args_nn_budget 100
#define args_max_cosine_distance 0.2
#define args_min_confidence 0.7
#define args_nms_max_overlap 1
#define len_tracklet 10
#define padding 0

cv::Scalar rand_color(int color) {
    return cv::Scalar(color&255, (color>>8)&255, (color>>16)&255);
}

enum DETECTBOX_IDX {IDX_X = 0, IDX_Y, IDX_W, IDX_H };

float calIOU(const DETECTBOX& bbox1, const DETECTBOX& bbox2) {
    int x1 = bbox1(IDX_X); 
    int y1 = bbox1(IDX_Y); 
    int x2 = bbox1(IDX_X) + bbox1(IDX_W);
    int y2 = bbox1(IDX_Y) + bbox1(IDX_H);
        
    int xx1 = bbox2(IDX_X) > x1?bbox2(IDX_X):x1;
    int yy1 = bbox2(IDX_Y) > y1?bbox2(IDX_Y):y1;
    int xx2 = bbox2(IDX_X) + bbox2(IDX_W);
    int yy2 = bbox2(IDX_Y) + bbox2(IDX_H);
    float area = bbox2(IDX_W) * bbox2(IDX_H) * 1.0;
    xx2 = (xx2 < x2?xx2:x2);
    yy2 = (yy2 < y2?yy2:y2);
    //standard area = w*h;
    int w = xx2-xx1; w = (w > 0?w:0);
    int h = yy2-yy1; h = (h > 0?h:0);
    float tmp_overlap = w*h*1.0/area;
    return tmp_overlap;
}

void getDetectionFeature(const DETECTBOX& bbox, const DETECTIONS& detections, FEATURE& this_feat, int& lineidx) {
    float max_iou = 0;
    float max_iou_idx = -1;
    for (int i = 0; i < detections.size(); i++) {
        DETECTBOX bbox2 = detections[i].tlwh;
        float this_iou = calIOU(bbox, bbox2);
        if (this_iou > max_iou) {
            max_iou = this_iou;
            max_iou_idx = i;
        }
    }
    this_feat = detections[max_iou_idx].feature;
    lineidx = detections[max_iou_idx].line_idx;
}

bool VideoTracker::RunTrackerWithDat(const char* motDir, const char* output_path,
	                                  bool write_feature, bool show_visualization) {
    char imgname[255];
    sprintf(imgname, "%s/img1/%06d.jpg", motDir, 1);
    Mat img = imread(imgname);
    int img_height = img.rows;
    int img_width = img.cols;
	cout << "height: " << img_height << ", width: " << img_width << endl;

    vector<cv::Scalar> color_vec;
    int num_color = 50;
    int rng[50] = {8323, 59858, 41441, 6393, 18251, 35838, 62745, 63228, 10328, 63600, 62719,
             31804, 52437, 9297, 27635, 59999, 51905, 62865, 42963, 2340, 55631, 61190,
             44466, 49642, 48684, 25695, 42940, 11215, 46251, 2086, 18140, 3025, 6363,
             53939, 45512, 20771, 62239, 2257, 28737, 24991, 50138, 52081, 12239, 32076,
             29182, 42328, 46456, 49423, 18077, 44512};
    for (int i_cl = 0; i_cl < num_color; i_cl++) {
        color_vec.push_back(rand_color(rng[i_cl]));
    }
    
	tracker mytracker(args_max_cosine_distance, args_nn_budget);
	int min_frame_idx, max_frame_idx;
	GetSeqInfo(motDir, min_frame_idx, max_frame_idx);
	ofstream output_file;
	output_file.open(output_path, ios::out|ios::trunc);

	//prepare data:
	if(ModelDetection::getInstance()->loadDataFromFile(motDir, true) == false) {
		this->errorMsg = "load detection data from MOT directory & model2.dat failed!";
		return false;
	}
	for (int i = min_frame_idx; i <= max_frame_idx; i++) {
		if (i % 100 == 0 || i == max_frame_idx) {
            std::cout << "Frame " << i << " / " << max_frame_idx << std::endl;
		}
		DETECTIONS detections;

		if(ModelDetection::getInstance()->getFrameDetections(i, detections) == false) {
			this->errorMsg = "ModelDetection not load MOT model data!";
			return false;
		}
		ModelDetection::getInstance()->dataMoreConf(args_min_confidence, detections);
		ModelDetection::getInstance()->dataPreprocessing(args_nms_max_overlap, detections);

		mytracker.predict();
		mytracker.update(detections);
		std::vector<RESULT_DATA> result;
        std::vector<FEATURE> result_feature;
        std::vector<int> result_lineidx;
		for (Track& track : mytracker.tracks) {
			if (!track.is_confirmed() || track.time_since_update >= 1) continue;
			result.push_back(std::make_pair(track.track_id, track.to_tlwh()));
            FEATURE this_feat;
            int this_lineidx;
            getDetectionFeature(track.to_tlwh(), detections, this_feat, this_lineidx);
            result_feature.push_back(this_feat);
            result_lineidx.push_back(this_lineidx);
		}
		// Visualization
		if(show_visualization) {
			char fname[255], showMsg[10];
			sprintf(fname, "%simage_%05d.jpg", motDir, i);
			Mat image = imread(fname);
			
			// Show all detection boxes in red
			for (unsigned int k = 0; k < detections.size(); k++) {
				DETECTBOX tmpbox = detections[k].tlwh;
				Rect rect(tmpbox(0), tmpbox(1), tmpbox(2), tmpbox(3));
				rectangle(image, rect, Scalar(0, 0, 255), 4);
			}
			// Show bounding boxes of tracking objects with ID in blue
			for(unsigned int k = 0; k < result.size(); k++) {
				DETECTBOX tmp = result[k].second;
				Rect rect = Rect(tmp(0), tmp(1), tmp(2), tmp(3));
				rectangle(image, rect, color_vec[result[k].first % num_color], 2);
				sprintf(showMsg, "%d", result[k].first);
				putText(image, showMsg, Point(rect.x, rect.y), CV_FONT_HERSHEY_SIMPLEX, 0.8, color_vec[result[k].first % num_color], 2);
				
				// Write results into output file
				output_file << i << "," << result[k].first << ","
				            << tmp(0) << "," << tmp(1) << "," 
				            << tmp(2) << "," << tmp(3) 
				            << ",1,-1,-1,-1" << endl;
			}
			imshow("DeepSortTracking", image);
			waitKey(10);
		}
		else {
			for(unsigned int k = 0; k < result.size(); k++) {
				DETECTBOX tmp = result[k].second;
				// Write results into output file
                float x1 = (tmp(0) - padding) < 0? 0:(tmp(0) - padding);
                float y1 = (tmp(1) - padding) < 0? 0:(tmp(1) - padding);
                float x2 = (tmp(0) + tmp(2) + padding) > img_width? img_width:(tmp(0) + tmp(2) + padding);
                float y2 = (tmp(1) + tmp(3) + padding) > img_height? img_height:(tmp(1) + tmp(3) + padding);
                output_file << i << "," << result[k].first << ","
				            << x1 << "," << y1 << "," 
				            << x2-x1 << "," << y2-y1 
				            << ",1,-1,-1,-1";
                if (write_feature) {
                    for (int idx_feat = 0; idx_feat < FEATURE_DIM; idx_feat++) {
                        output_file << "," << result_feature[k][idx_feat];
                    }
                }
                char jpgname[255];
                sprintf(jpgname, "%06d_%d", i, result_lineidx[k]);
                output_file << "," << jpgname;
                output_file << endl;
			}
		}
	}//end while;
	output_file.close();
	return true;
}

bool VideoTracker::RunTrackerWithDets(const char* motDir, const char* output_path,
	                                  bool write_feature, bool write_to_image) {
    char imgname[255];
    sprintf(imgname, "%s/img1/%06d.jpg", motDir, 1);
    Mat img = imread(imgname);
    int img_height = img.rows;
    int img_width = img.cols;
	cout << "height: " << img_height << ", width: " << img_width << endl;

	vector<cv::Scalar> color_vec;
    int num_color = 50;
    int rng[50] = {8323, 59858, 41441, 6393, 18251, 35838, 62745, 63228, 10328, 63600, 62719,
             31804, 52437, 9297, 27635, 59999, 51905, 62865, 42963, 2340, 55631, 61190,
             44466, 49642, 48684, 25695, 42940, 11215, 46251, 2086, 18140, 3025, 6363,
             53939, 45512, 20771, 62239, 2257, 28737, 24991, 50138, 52081, 12239, 32076,
             29182, 42328, 46456, 49423, 18077, 44512};
    for (int i_cl = 0; i_cl < num_color; i_cl++) {
        color_vec.push_back(rand_color(rng[i_cl]));
    }
    
	// tracker mytracker(args_max_cosine_distance, args_nn_budget);
	int min_frame_idx, max_frame_idx;
	GetSeqInfo(motDir, min_frame_idx, max_frame_idx);
	ofstream output_file;
	output_file.open(output_path, ios::out|ios::trunc);
	
	if(ModelDetection::getInstance()->loadDataFromFile(motDir, false) == false) {
		this->errorMsg = "load detection data from MOT directory failed!";
		return false;
	}

	tracker mytracker(args_max_cosine_distance, args_nn_budget);
    std::unordered_map<int, int> count_map;
    int idx_count = 0;
    // initialize vec_tracklets
    std::vector<std::vector<Point>> vec_tracklets;

	for (int i = min_frame_idx; i <= max_frame_idx; i++) {
		if (i % 100 == 0 || i == max_frame_idx) {
            std::cout << "Frame " << i << " / " << max_frame_idx << std::endl;
		}
        DETECTIONS detections;

		if(ModelDetection::getInstance()->getFrameDetections(i, detections) == false) {
			this->errorMsg = "ModelDetection not load MOT model data!";
			return false;
		}
		ModelDetection::getInstance()->dataMoreConf(args_min_confidence, detections);
		ModelDetection::getInstance()->dataPreprocessing(args_nms_max_overlap, detections);

		char fname[255];
		sprintf(fname, "%simg1/%06d.jpg", motDir, i);
		Mat image = imread(fname);
        Mat image_ori = image.clone();
		if(FeatureExtractor::getInstance()->getRectsFeature(image, detections) == false) {
			this->errorMsg = "Caffe get feature failed!";
			return false;
		}

		mytracker.predict();
		mytracker.update(detections);
		std::vector<RESULT_DATA> result;
		for (Track& track : mytracker.tracks) {
			if (!track.is_confirmed() || track.time_since_update >= 1) continue;
			result.push_back(std::make_pair(track.track_id, track.to_tlwh()));
		}

        char showMsg[10];
		if (write_to_image) {
			for (unsigned int k = 0; k < detections.size(); k++) {
				DETECTBOX tmpbox = detections[k].tlwh;
				Rect rect(tmpbox(0), tmpbox(1), tmpbox(2), tmpbox(3));
				rectangle(image, rect, Scalar(0, 0, 255), 4);
			}
        }
        std::vector<Point> vec_pos;
        float front_scale = image.rows/1000.0;
        // Show bounding boxes of tracking objects with ID in green
        for(unsigned int k = 0; k < result.size(); k++) {
            DETECTBOX tmp = result[k].second;
            Rect rect = Rect(tmp(0), tmp(1), tmp(2), tmp(3));
            if (write_to_image) {
                cv::Scalar this_color = color_vec[result[k].first % num_color];
                rectangle(image, rect, this_color, 4);
#ifdef WITH_COUNT
                if(count_map.find(result[k].first) == count_map.end()) {
                    ++idx_count;
                    count_map[result[k].first] = idx_count;
                }
                sprintf(showMsg, "%d", count_map[result[k].first]);
#else
                sprintf(showMsg, "%d", result[k].first);
#endif				
                putText(image, showMsg, Point(rect.x, rect.y), CV_FONT_HERSHEY_SIMPLEX, 0.8*front_scale, this_color, 2*front_scale);
                
                // update tracklets
                Point pos_tail;
                pos_tail.x = tmp(0) + tmp(2)/2;
                pos_tail.y = tmp(1) + tmp(3);
                vec_pos.push_back(pos_tail);
            }
            // Write results into output file
            float x1 = (tmp(0) - padding) < 0? 0:(tmp(0) - padding);
            float y1 = (tmp(1) - padding) < 0? 0:(tmp(1) - padding);
            float x2 = (tmp(0) + tmp(2) + padding) > img_width? img_width:(tmp(0) + tmp(2) + padding);
            float y2 = (tmp(1) + tmp(3) + padding) > img_height? img_height:(tmp(1) + tmp(3) + padding);
            output_file << i << "," << result[k].first << ","
                        << x1 << "," << y1 << "," 
                        << x2-x1 << "," << y2-y1 
                        << ",1,-1,-1,-1";
            sprintf(fname, "%simage_%05d.jpg", motDir, i);
            if (write_feature) {
                std::vector<float> feature(FEATURE_DIM, 0);
                if (!FeatureExtractor::getInstance()->getRectsFeature(image_ori, cv::Rect(tmp(0), tmp(1), tmp(2), tmp(3)), feature)) {
                    continue;
                }
                for (auto val : feature) {
                    output_file << "," << val;
                }
            }
            output_file << endl;
        }
		if (write_to_image) {
            vec_tracklets.push_back(vec_pos);
            if(vec_tracklets.size() > len_tracklet) {
                vec_tracklets.erase(vec_tracklets.begin());
            }
            for(int idx_t = 0; idx_t < vec_tracklets.size(); ++idx_t) {
                for(int idx_pos = 0; idx_pos < vec_tracklets[idx_t].size(); ++idx_pos) {
                    circle(image, vec_tracklets[idx_t][idx_pos], 4, cv::Scalar(255,255,0), -1);
                }
            }
            stringstream ss_fullpath;
            ss_fullpath << motDir;
            vector<string> tmp_path;
            string tmp_s;
            while (getline(ss_fullpath, tmp_s, '/')) {
                tmp_path.push_back(tmp_s);
            }
            string type_name = tmp_path[tmp_path.size() - 3];
            string camera_name = tmp_path[tmp_path.size() - 2];
            string seq_name = tmp_path[tmp_path.size() - 1];
            
            sprintf(fname, "../output/%s/%s/%s/%06d.jpg", type_name.data(), camera_name.data(), seq_name.data(), i);
            imwrite(fname, image);
        }
	}
    output_file.close();
	return true;
}

std::string VideoTracker::showErrMsg() {
	return this->errorMsg;
}

void VideoTracker::GetSeqInfo(string sequence_dir,int &min_frame_idx, int &max_frame_idx) {
	min_frame_idx = 1;
	string img_dir = sequence_dir.append("");
	
	char cmd[200];
	sprintf(cmd, "ls %s/img1/*.jpg | wc -l > ./tmp_img_count.txt", img_dir.c_str());
	system(cmd);

	std::ifstream readfile("./tmp_img_count.txt");
	string res;
	readfile >> res;
	sscanf(res.c_str(), "%d", &max_frame_idx);
}


