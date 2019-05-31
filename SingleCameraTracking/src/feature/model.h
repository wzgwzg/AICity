#ifndef MODEL_H
#define MODEL_H
#include "dataType.h"
#include <map>
#include "opencv2/opencv.hpp"

/**
 * Each rect's data structure.
 * tlwh: topleft point & (w,h)
 * confidence: detection confidence.
 * feature: the rect's reid feature.
 */
class DETECTION_ROW {
public:
    DETECTBOX tlwh;
    float confidence;
    FEATURE feature;
    DETECTBOX to_xyah() const;
    DETECTBOX to_tlbr() const;
    int line_idx;
};

typedef std::vector<DETECTION_ROW> DETECTIONS;

/**
 * Get each image's rects & corresponding features.
 * Method of filter conf.
 * Method of preprocessing.
 */
class ModelDetection
{

public:
    static ModelDetection* getInstance();
    bool loadDataFromFile(const char* motDir, bool withFeature);
    bool getFrameDetections(int frame_idx, DETECTIONS& res);
    void dataMoreConf(float min_confidence, DETECTIONS& d);
    void dataPreprocessing(float max_bbox_overlap, DETECTIONS& d);

private:
    ModelDetection();
    ModelDetection(const ModelDetection&);
    ModelDetection& operator =(const ModelDetection&);
    static ModelDetection* instance;

    using AREAPAIR = std::pair<int, double>;
    struct cmp {
        bool operator()(const AREAPAIR a, const AREAPAIR b) {
            return a.second < b.second;
        }
    };
    std::map<int, DETECTIONS> data;
    void _Qsort(DETECTIONS d, std::vector<int>& a, int low, int high);
    bool loadFromFile;
};

#endif // MODEL_H
