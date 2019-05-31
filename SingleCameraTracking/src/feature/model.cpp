#include "model.h"
#include <algorithm>

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/core/core_c.h"
#include "opencv2/core/version.hpp"

#ifndef CV_VERSION_EPOCH
#include "opencv2/videoio/videoio_c.h"
#define OPENCV_VERSION CVAUX_STR(CV_VERSION_MAJOR)""CVAUX_STR(CV_VERSION_MINOR)""CVAUX_STR(CV_VERSION_REVISION)
#pragma comment(lib, "opencv_world" OPENCV_VERSION ".lib")
#else
#define OPENCV_VERSION CVAUX_STR(CV_VERSION_EPOCH)""CVAUX_STR(CV_VERSION_MAJOR)""CVAUX_STR(CV_VERSION_MINOR)
#pragma comment(lib, "opencv_core" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_imgproc" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_highgui" OPENCV_VERSION ".lib")
#endif

#endif

enum DETECTBOX_IDX {IDX_X = 0, IDX_Y, IDX_W, IDX_H };
ModelDetection *ModelDetection::instance = NULL;

ModelDetection *ModelDetection::getInstance()
{
	if (instance == NULL) {
		instance = new ModelDetection();
	}
	return instance;
}

/**
 * load image's rects data.
 * @fpath: the mot directory.
 * model file must have \n in each line.
 */
bool ModelDetection::loadDataFromFile(const char* motDir, bool withFeature) {
	if(this->loadFromFile == true) {
		//run time program can only execute one time for this function.
		//return false;
	}

	std::string fname = motDir;
	if(withFeature) {
		// with detections and features
		fname.append("det/det_se154_csc_ms_feat.txt");
	} else {
		// with detections only
		fname.append("det/det_se154_csc_ms.txt");
	}
    std::cout << fname << std::endl;
	FILE* fp = fopen(fname.c_str(), "r");
	if(fp == NULL) return false;
    int line_idx = 0;
	while(true) {
		DETECTION_ROW tmpRow;

		int frame_idx;
		float tmpValue, x, y, w, h;
		if(fscanf(fp, "%f,", &tmpValue) != 1) break;
		frame_idx = (int)tmpValue;

		fscanf(fp, "%*f,%f,%f,%f,%f,", &x, &y, &w, &h);
		tmpRow.tlwh = DETECTBOX(x, y, w, h);
		fscanf(fp, "%f,%*f,%*f,%*f", &tmpValue);
		tmpRow.confidence = tmpValue;

		if(withFeature) {//read feature from file;
			for(int i = 0; i < FEATURE_DIM; i++) {
				fscanf(fp, ",%f", &tmpValue);
				tmpRow.feature[i] = tmpValue/2.;
			}
		}
		else {
			fscanf(fp, "%*[^\n]"); //skip the line;
		}
        tmpRow.line_idx = line_idx;
		if(data.find(frame_idx) == data.end()) {
			DETECTIONS frameData;
			frameData.push_back(tmpRow);
			data.insert(make_pair(frame_idx, frameData));
		} else data[frame_idx].push_back(tmpRow);
        
        line_idx++;
		if(fscanf(fp, "\n") == -1) break;
	}
	fclose(fp);
	this->loadFromFile = true;
	return true;
}

bool ModelDetection::getFrameDetections(int frame_idx, DETECTIONS& res)
{
	if(this->loadFromFile == true) {
		res = data[frame_idx];
		return true;
	}
	return false;
}


void ModelDetection::dataMoreConf(float min_confidence, DETECTIONS &d)
{
	DETECTIONS::iterator it;
	for(it = d.begin(); it != d.end();) {
		if((*it).confidence < min_confidence) it = d.erase(it);
		else ++it;
	}
}

void ModelDetection::dataPreprocessing(float max_bbox_overlap, DETECTIONS &d)
{
	int size = int(d.size());
	if(size == 0) return;

	//generate idx:
	std::vector<int> idxs;
	idxs.reserve(size);

	std::vector<bool> idx_status;
	idx_status.reserve(size);
	for(size_t i = 0; i < size; ++i) {
		idxs.push_back(int(i));
		idx_status.push_back(false);        
	}

	//get areas:
	std::vector<double> areas;
	areas.reserve(size);
	for(size_t i = 0; i < size; ++i) {
		double tmp = (d[i].tlwh(IDX_W)+1)*(d[i].tlwh(IDX_H)+1);
		areas.push_back(tmp);
	}

	//sort idxs by scores in ascending order ==>quick sort:
	_Qsort(d, idxs, 0, size-1);

	//get delete detections:
	std::vector<int> delIdxs;
	while(true) {//get compare idx;
		int i = -1;
		for(int j = size-1; j > 0; --j) {
			if(idx_status[j] == false) {
				i = j;
				idx_status[i] = true;
				break;
			}
		}
		if(i == -1) break; //end circle

		int x1 = d[idxs[i]].tlwh(IDX_X); 
		int y1 = d[idxs[i]].tlwh(IDX_Y); 
		int x2 = d[idxs[i]].tlwh(IDX_X) + d[idxs[i]].tlwh(IDX_W);
		int y2 = d[idxs[i]].tlwh(IDX_Y) + d[idxs[i]].tlwh(IDX_H);
		for(size_t j = 0; j < i; j++) {
			if(idx_status[j] == true) continue;
			int xx1 = int(d[idxs[j]].tlwh(IDX_X) > x1?d[idxs[j]].tlwh(IDX_X):x1);
			int yy1 = int(d[idxs[j]].tlwh(IDX_Y) > y1?d[idxs[j]].tlwh(IDX_Y):y1);
			int xx2 = d[idxs[j]].tlwh(IDX_X) + d[idxs[j]].tlwh(IDX_W);
			int yy2 = d[idxs[j]].tlwh(IDX_Y) + d[idxs[j]].tlwh(IDX_H);
			xx2 = (xx2 < x2?xx2:x2);
			yy2 = (yy2 < y2?yy2:y2);
			//standard area = w*h;
			int w = xx2-xx1+1; w = (w > 0?w:0);
			int h = yy2-yy1+1; h = (h > 0?h:0);
			//get delIdx;
			double tmp_overlap = w*h*1.0/areas[idxs[j]];
			if(tmp_overlap > max_bbox_overlap) {
				delIdxs.push_back(idxs[j]);
				idx_status[j] = true;
			}
		}
	}
	//delete from detections:
    if (delIdxs.size() == 0) return;
	for(size_t i = 0; i < delIdxs.size(); ++i) {
		DETECTIONS::iterator it = d.begin() + delIdxs[i];
		d.erase(it);
	}
}

//load detection data from outer file
ModelDetection::ModelDetection()
{
	this->loadFromFile = false;
}

void ModelDetection::_Qsort(DETECTIONS d, std::vector<int>& a, int low, int high)
{
	if(low >= high) return;
	int first = low;
	int last = high;

	int key_idx = a[first];
	while(first < last) {
		while(first < last && d[a[last]].confidence >= d[key_idx].confidence) --last;
		a[first] = a[last];
		while(first < last && d[a[first]].confidence <= d[key_idx].confidence) ++first;
		a[last] = a[first];
	}
	a[first] = key_idx;
	_Qsort(d, a, low, first-1);
	_Qsort(d, a, first+1, high);
}

DETECTBOX DETECTION_ROW::to_xyah() const
{//(centerx, centery, ration, h)
	DETECTBOX ret = tlwh;
	ret(0,IDX_X) += (ret(0, IDX_W)*0.5);
	ret(0, IDX_Y) += (ret(0, IDX_H)*0.5);
	ret(0, IDX_W) /= ret(0, IDX_H);
	return ret;
}

DETECTBOX DETECTION_ROW::to_tlbr() const
{//(x,y,xx,yy)
	DETECTBOX ret = tlwh;
	ret(0, IDX_X) += ret(0, IDX_W);
	ret(0, IDX_Y) += ret(0, IDX_H);
	return ret;
}

