#include <caffe/caffe.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "model.h"

using namespace caffe;
using std::string;

class FeatureExtractor
{
public:
	static FeatureExtractor* getInstance();
	bool getRectsFeature(const cv::Mat& img, DETECTIONS& d);
	bool getRectsFeature(const cv::Mat& img, cv::Rect rect, std::vector<float>& feature);

private:
	static FeatureExtractor* instance;
	FeatureExtractor();
	FeatureExtractor(const FeatureExtractor&);
	FeatureExtractor& operator = (const FeatureExtractor&);
	bool init();
	~FeatureExtractor();

	void WrapInputLayer(std::vector<cv::Mat>* input_channels,
						float* &input_data, int width, int height,
						int channels);
	void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);
	void SetMean(const string& mean_file);

private:
	int feature_dim;
	std::shared_ptr<Net<float> > net_;
	cv::Size input_geometry_;
	int num_channels_;
	cv::Mat mean_;	
};
