#include <fstream>
#include "FeatureExtractor.h"

#define CAFFE_MEAN_FILE ""
#define CAFFE_TRAINED_MODEL "../RUNNINGDATA/caffe_networks/vehicle.caffemodel"
#define CAFFE_MODEL_FILE "../RUNNINGDATA/caffe_networks/vehicle.prototxt"
#define CAFFE_MEAN_VALUE 0.485, 0.456, 0.406
#define CAFFE_STD_VALUE 0.229, 0.224, 0.225
#define CAFFE_SCALE_VALUE 0.00392157

FeatureExtractor *FeatureExtractor::instance = NULL;

FeatureExtractor *FeatureExtractor::getInstance() {
	if(instance == NULL) {
		instance = new FeatureExtractor();
	}
	return instance;
}

FeatureExtractor::FeatureExtractor() {
	//prepare model:
	bool status = init();
	if(status == false) exit(1);
}

FeatureExtractor::~FeatureExtractor() {
}

bool FeatureExtractor::init() {
	#ifdef CPU_ONLY
		Caffe::set_mode(Caffe::CPU);
	#else
		Caffe::set_mode(Caffe::GPU);
	#endif

	/* Load the network. */
	string model_file = CAFFE_MODEL_FILE;
	string trained_model = CAFFE_TRAINED_MODEL;
	string mean_file = CAFFE_MEAN_FILE;

	net_.reset(new Net<float>(model_file, TEST));
	net_->CopyTrainedLayersFrom(trained_model);

	Blob<float>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	if(num_channels_ != 3) {
		std::cout << "Input layer should have 1 or 3 channels." << std::endl;
		return false;
	}
	if(input_layer->width() != 222 || input_layer->height() != 222) {
		std::cout << "Input layer dimension should be 222x222." << std::endl;
		return false;
	}
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

	feature_dim = FEATURE_DIM;

	Blob<float>* output_layer = net_->output_blobs()[0];
	if(output_layer->channels() != feature_dim) {
		std::cout << "The feature dimension is different from the output layer dimension. ("
                          << feature_dim << " vs " << output_layer->channels() << ")" << std::endl;
		return false;
	}

	return true;
}

bool FeatureExtractor::getRectsFeature(const cv::Mat& img, DETECTIONS& d) {
	int batch_size = 20;
    if(d.size()==0)
		return true;

	std::vector<cv::Mat> samples;
	std::vector<float> output_total;
    for(int i = 0; i < d.size(); i++) {
        auto& dbox = d[i];
		cv::Rect rc = cv::Rect(int(dbox.tlwh(0)), int(dbox.tlwh(1)),
				int(dbox.tlwh(2)), int(dbox.tlwh(3)));
		rc.x = (rc.x >= 0 ? rc.x : 0);
		rc.y = (rc.y >= 0 ? rc.y : 0);
		rc.width = (rc.x + rc.width <= img.cols? rc.width: (img.cols-rc.x));
		rc.height = (rc.y + rc.height <= img.rows? rc.height:(img.rows - rc.y));

		cv::Mat mattmp = img(rc).clone();
		samples.push_back(mattmp);
        if (samples.size() == batch_size || i == (d.size() - 1)) {

            int count = samples.size();
            Blob<float>* input_layer = net_->input_blobs()[0];
            input_layer->Reshape(count, num_channels_,
                               input_geometry_.height, input_geometry_.width);
            net_->Reshape();

            float* input_data = input_layer->mutable_cpu_data();
            for(int idx_sample = 0; idx_sample < count; ++idx_sample) {
                std::vector<cv::Mat> input_channels;
                WrapInputLayer(&input_channels, input_data, input_geometry_.width,
                               input_geometry_.height, num_channels_);
                Preprocess(samples[idx_sample], &input_channels);
            }
            net_->ForwardPrefilled();

            Blob<float>* output_layer = net_->output_blobs()[0];
            const float* output_data = output_layer->cpu_data();
            for (int j = 0; j < feature_dim * count; j++) {
                output_total.push_back(output_data[j]);
            }
            samples.clear();
        }
    }
	int i = 0;
	for(DETECTION_ROW& dbox : d) {
		for(int j = 0; j < feature_dim; j++) {
			 dbox.feature[j] = output_total[i*feature_dim+j];
		}
		i++;
	}
	return true;
}

bool FeatureExtractor::getRectsFeature(const cv::Mat& img, cv::Rect rc, std::vector<float>& feature) {
	std::vector<cv::Mat> samples;
    rc.x = (rc.x >= 0 ? rc.x : 0);
    rc.y = (rc.y >= 0 ? rc.y : 0);
    rc.width = (rc.x + rc.width <= img.cols? rc.width: (img.cols - rc.x));
    rc.height = (rc.y + rc.height <= img.rows? rc.height:(img.rows - rc.y));
    if (rc.width <= 0 || rc.height <= 0)
        return false;
    cv::Mat mattmp = img(rc).clone();
    samples.push_back(mattmp);
	int count = samples.size();
	Blob<float>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape(count, num_channels_,
	                   input_geometry_.height, input_geometry_.width);
	net_->Reshape();

	float* input_data = input_layer->mutable_cpu_data();
    for(int idx_sample = 0; idx_sample < count; ++idx_sample) {
		std::vector<cv::Mat> input_channels;
		WrapInputLayer(&input_channels, input_data, input_geometry_.width,
			           input_geometry_.height, num_channels_);
		Preprocess(samples[idx_sample], &input_channels);
	}
	net_->ForwardPrefilled();

	/* Copy the output layer to a std::vector */
	Blob<float>* output_layer = net_->output_blobs()[0];
	const float* output_data = output_layer->cpu_data();
    feature.reserve(feature_dim);
    for(int j = 0; j < feature_dim; j++) {
         feature[j] = output_data[j];
    }
	return true;
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void FeatureExtractor::WrapInputLayer(std::vector<cv::Mat>* input_channels,
	                                  float* &input_data, int width, int height,
	                                  int channels) {
	for (int i = 0; i < channels; ++i) {
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
}
void FeatureExtractor::Preprocess(const cv::Mat& img,
                            	  std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample.convertTo(sample_float, CV_32FC3);
  else
    sample.convertTo(sample_float, CV_32FC1);
  
  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample_float, sample_resized, input_geometry_);
  else
    sample_resized = sample_float;

  for (int y = 0; y < sample_float.rows; y++) {
    for(int x = 0; x < sample_float.cols; x++) {
        cv::Vec3f val = sample_float.at<cv::Vec3f>(y,x);
    }
  }

  float scale = 1;
#ifdef CAFFE_SCALE_VALUE
  scale = CAFFE_SCALE_VALUE;
#endif
#ifdef CAFFE_MEAN_VALUE
  std::vector<float> mean = {CAFFE_MEAN_VALUE};
#else
  std::vector<float> mean = {0, 0, 0};
#endif
#ifdef CAFFE_STD_VALUE
  std::vector<float> std = {CAFFE_STD_VALUE};
#else
  std::vector<float> std = {1, 1, 1};
#endif
  cv::Mat sample_prep = sample_resized.clone();
  int height = sample_resized.rows;
  int width = sample_resized.cols;
  for (int y = 0; y < height; y++) {
    const float *ps = sample_resized.ptr<float>(y);
    float *pd = sample_prep.ptr<float>(y);
    for (int x = 0; x < width; x++) {
      pd[x * 3 + 0] = (ps[x * 3 + 2] * scale - mean[0]) / std[0];
      pd[x * 3 + 1] = (ps[x * 3 + 1] * scale - mean[1]) / std[1];
      pd[x * 3 + 2] = (ps[x * 3 + 0] * scale - mean[2]) / std[2];
    }
  }
  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_prep, *input_channels);

  // if(reinterpret_cast<float*>(input_channels->at(0).data)
  //       != net_->input_blobs()[0]->cpu_data())
  //   std::cout << "Input channels are not wrapping the input layer of the network." 
  //             << std::endl;
}
/* Load the mean file in binaryproto format. */
void FeatureExtractor::SetMean(const string& mean_file) {
	BlobProto blob_proto;
	ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

	/* Convert from BlobProto to Blob<float> */
	Blob<float> mean_blob;
	mean_blob.FromProto(blob_proto);
	if(mean_blob.channels() != num_channels_) {
		std::cout << "Number of channels of mean file doesn't match input layer." << std::endl;
	}

	/* The format of the mean file is planar 32-bit float BGR or grayscale. */
	std::vector<cv::Mat> channels;
	float* data = mean_blob.mutable_cpu_data();
	for (int i = 0; i < num_channels_; ++i) {
	/* Extract an individual channel. */
		cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
		channels.push_back(channel);
		data += mean_blob.height() * mean_blob.width();
	}

	/* Merge the separate channels into a single image. */
	cv::Mat mean;
	cv::merge(channels, mean);

	/* Compute the global mean pixel value and create a mean image
	* filled with this value. */
	cv::Scalar channel_mean = cv::mean(mean);
	mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}
