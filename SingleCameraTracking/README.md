# Single Camera Tracking
This part is based on bitzy's implementation of DeepSORT: https://github.com/bitzy/DeepSort

## Dependencies
- opencv 2.4
- caffe
- cuda 7.5
- cudnn 5.0 
Refer to our source codes for other dependencies  

## Preparing data and models
Before running the tracker, you need:
1. download aic19-track1-mtmc data sets from [TRACK1-DOWNLOAD](http://www.aicitychallenge.org/track1-download/), and release them into the **aic19**.
2. download detection results from [detections](https://pan.baidu.com/s/1dE2n1f0qKICRbLFm7EWxHA)(extract code: dlqy) and put them into **detection_res**. These pickle files contain detection results of our detector.
3. download the vehicle reid model from [reid-model](https://pan.baidu.com/s/1smsCRuQaQ4O3jMmOZfXNLg)(extract code: fc3h) and put it into **RUNNINGDATA/caffe_networks**. This model is implemented by pytorch and converted to caffe model.

## Running
```
cd tools
python convert_coco_detection_to_mot.py
cd ..
mkdir build && cd build
cmake ..
make -j
./run_tracker train/S01 c001 0 0
```

