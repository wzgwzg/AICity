#ifndef VIDEOTRACKER_H_
#define VIDEOTRACKER_H_
#include <string>

/*
 * VideoTracker
 *
 * run:
 * -vpath: Path to the video.
 * -show_visualization: Decide whether to show the figure.
 *
 * RunTrackerWithDat:
 * -motDir: Path to the tracking sequences.
 * -show_visualization: Decide whether to show the figure.
 *
 * RunTrackerWithDets:
 * -motDir: Path to the tracking sequences.
 * -show_visualization: Decide whether to show the figure.
 */
class VideoTracker {
public:
	bool run(const char* vpath, bool show_visualization);
	bool RunTrackerWithDat(const char* motDir, const char* output_path,
                            bool write_feature, bool show_visualization);
	bool RunTrackerWithDets(const char* motDir, const char* output_path,
		                    bool show_visualization, bool write_to_image);
	std::string showErrMsg();

private:
	bool vShowFlag;
	std::string errorMsg;

	void GetSeqInfo(std::string sequence_dir,int &min_frame_idx, int &max_frame_idx);
	std::string loadFromBUFFERFILE();
};


#endif /* VIDEOTRACKER_H_ */

